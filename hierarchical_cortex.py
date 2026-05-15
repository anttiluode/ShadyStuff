import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hierarchical Cortex (V1→V2) booting on {device}...")

# Layout constants
GRID_H, GRID_W = 16, 8          # V1: 128 neurons on a physical 2D sheet
N1 = GRID_H * GRID_W            # 128
N2 = 64                          # V2: temporal pattern detectors
V2_TAUS = [1, 2, 3, 5, 8, 13, 21]  # Fibonacci TEMPORAL delays (frames)


# =====================================================================
# V1 — SPATIAL GEOMETRIC CORTEX
# Receives raw visual input. Detects geometry via Fibonacci spatial
# dilations. Arranged on a 2D retinotopic sheet. Receives top-down
# feedback from V2 as threshold modulation.
# =====================================================================
class V1Cortex(nn.Module):
    def __init__(self, in_channels=2, num_neurons=N1):
        super().__init__()
        self.num_neurons = num_neurons
        self.taus = [1, 2, 3, 5, 8, 13, 21]
        self.patch_dim = in_channels * 9

        m_init = torch.randn(num_neurons, len(self.taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)

        self.ais_pool  = None
        self.prediction = None
        self.pre_trace  = torch.zeros(num_neurons, device=device)
        self.post_trace = torch.zeros(num_neurons, device=device)
        self.neuron_rate = torch.zeros(num_neurons, device=device)

        # Physical 2D layout — precompute pairwise distances
        positions = [[i / GRID_H, j / GRID_W]
                     for i in range(GRID_H) for j in range(GRID_W)]
        pos = torch.tensor(positions[:num_neurons], dtype=torch.float32)
        self.register_buffer('positions', pos)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        self.register_buffer('dist2_matrix', (diff ** 2).sum(-1))

        # Physics parameters (exposed to sliders)
        self.ais_integration   = 0.30
        self.base_thresh       = 0.05
        self.cemi_coupling     = 0.60
        self.cemi_sigma        = 0.30
        self.sparsity          = 0.05
        self.stdp_Aplus        = 0.10
        self.stdp_Aminus_ratio = 0.60
        self.stdp_tau          = 0.85
        self.pred_strength     = 0.60

        # V2 top-down feedback (injected externally each frame)
        self.v2_feedback          = torch.zeros(num_neurons, device=device)
        self.v2_feedback_strength = 0.40

    def _local_cemi(self):
        sigma2   = max(self.cemi_sigma ** 2, 1e-4)
        w        = torch.exp(-self.dist2_matrix / (2 * sigma2))
        w        = w / (w.sum(1, keepdim=True) + 1e-6)
        return (w * self.neuron_rate.unsqueeze(0)).sum(1)  # (N1,)

    def step(self, image_tensor):
        B, C, H, W = image_tensor.shape
        if self.ais_pool is None or self.ais_pool.shape[-2:] != (H, W):
            self.ais_pool   = torch.zeros(1, self.num_neurons, H, W, device=device)
            self.prediction = torch.zeros(B, C, H, W, device=device)

        # Predictive coding — drive on upward surprise
        pred_error = torch.relu(image_tensor - self.prediction)
        driving    = (1.0 - self.pred_strength) * image_tensor \
                   +        self.pred_strength  * pred_error

        # Fibonacci spatial resonance
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        for i, tau in enumerate(self.taus):
            pad      = tau
            padded   = F.pad(driving, (pad, pad, pad, pad), mode='reflect')
            v        = F.normalize(
                F.unfold(padded, kernel_size=3, dilation=tau).view(B, self.patch_dim, H, W),
                p=2, dim=1)
            dot      = torch.einsum('b c h w, n c -> b n h w', v, self.m[:, i, :])
            total_res += dot ** 2
        avg_res = total_res / len(self.taus)

        # AIS integration
        self.ais_pool = (self.ais_pool * (1.0 - self.ais_integration)
                       + avg_res        *        self.ais_integration)

        # Local CEMI + V2 top-down feedback modulates threshold
        local_cemi  = self._local_cemi()
        v2_fb       = self.v2_feedback * self.v2_feedback_strength
        cemi_total  = (local_cemi + v2_fb).clamp(0, 1)
        dyn_thresh  = self.base_thresh * torch.clamp(
            1.0 - cemi_total.view(1, self.num_neurons, 1, 1) * self.cemi_coupling,
            0.02, 1.0)

        # Top-k sparse firing
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_idx   = torch.topk(self.ais_pool, k, dim=1)
        sparse_mask   = torch.zeros_like(self.ais_pool, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_idx, True)
        spikes        = ((self.ais_pool > dyn_thresh) & sparse_mask).float()
        self.ais_pool = self.ais_pool * (1.0 - spikes)  # refractory reset

        self.neuron_rate = self.neuron_rate * 0.92 + spikes.mean(dim=(0, 2, 3)) * 0.08

        # STDP
        res_per_n      = avg_res.mean(dim=(0, 2, 3))
        self.pre_trace = self.pre_trace * self.stdp_tau + res_per_n * (1.0 - self.stdp_tau)
        spikes_flat    = spikes.view(self.num_neurons, -1)
        active         = spikes_flat.sum(dim=1) > 0

        if self.stdp_Aplus > 0 and active.any():
            for i, tau in enumerate(self.taus):
                pad    = tau
                padded = F.pad(driving, (pad, pad, pad, pad), mode='reflect')
                v_norm = F.normalize(
                    F.unfold(padded, kernel_size=3, dilation=tau).view(B, self.patch_dim, H, W),
                    p=2, dim=1)[0]
                for n in torch.where(active)[0]:
                    mask = spikes[0, n:n+1, :, :]
                    if mask.sum() == 0: continue
                    v_m   = (v_norm * mask).sum(dim=(1, 2)) / mask.sum()
                    m_cur = self.m[n, i]
                    delta = v_m - torch.dot(v_m, m_cur) * m_cur
                    self.m.data[n, i] += delta * (
                        self.stdp_Aplus * self.pre_trace[n]
                      - self.stdp_Aplus * self.stdp_Aminus_ratio * self.post_trace[n])
            self.m.data = F.normalize(self.m.data, p=2, dim=2)

        self.post_trace = self.post_trace * self.stdp_tau + active.float() * (1.0 - self.stdp_tau)
        self.prediction = self._reconstruct(spikes)

        return spikes, pred_error, local_cemi

    def _reconstruct(self, spikes):
        B, N, H, W  = spikes.shape
        recon       = torch.zeros(B, 2, H, W, device=device)
        spikes_flat = spikes.view(B, N, H * W)
        for i, tau in enumerate(self.taus):
            patches = torch.bmm(spikes_flat.transpose(1, 2),
                                self.m[:, i, :].unsqueeze(0).expand(B, -1, -1)).transpose(1, 2)
            pad     = tau
            folded  = F.fold(patches, output_size=(H+pad*2, W+pad*2), kernel_size=3, dilation=tau)
            recon  += folded[:, :, pad:-pad, pad:-pad]
        mn = recon.amin(dim=(2, 3), keepdim=True)
        mx = recon.amax(dim=(2, 3), keepdim=True)
        return (recon - mn) / (mx - mn + 1e-8)


# =====================================================================
# V2 — TEMPORAL PATTERN CORTEX
# Receives V1 spike rates as input. Detects patterns ACROSS TIME using
# Fibonacci temporal delays (Takens embedding of V1 dynamics).
# Learns: "which sequences of V1 states recur?"
# Outputs: predicted V1 activity → feeds back to V1 thresholds.
# =====================================================================
class V2Cortex(nn.Module):
    def __init__(self, n_v1=N1, n_v2=N2):
        super().__init__()
        self.n_v1 = n_v1
        self.n_v2 = n_v2
        self.taus = V2_TAUS  # Fibonacci delays in FRAMES (not pixels)

        # Templates: m_V2[neuron, tau_idx, v1_neuron]
        m_init = torch.randn(n_v2, len(self.taus), n_v1) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)

        self.ais_pool   = torch.zeros(n_v2, device=device)
        self.pre_trace  = torch.zeros(n_v2, device=device)
        self.post_trace = torch.zeros(n_v2, device=device)
        self.neuron_rate = torch.zeros(n_v2, device=device)
        self.global_cemi = 0.0

        # Ring buffer of V1 spike rates (stored on CPU to save GPU memory)
        self.v1_buffer = deque(maxlen=max(self.taus) + 3)

        # Physics parameters
        self.ais_integration   = 0.20
        self.base_thresh       = 0.01
        self.cemi_coupling     = 0.50
        self.sparsity          = 0.10
        self.stdp_Aplus        = 0.08
        self.stdp_Aminus_ratio = 0.50
        self.stdp_tau          = 0.90

    def step(self, v1_spike_rate):
        # v1_spike_rate: (N1,) — spatial mean activity per V1 neuron this frame
        self.v1_buffer.append(v1_spike_rate.detach().cpu())

        if len(self.v1_buffer) < max(self.taus) + 1:
            # Buffer not full yet — return zeros
            zero = torch.zeros(self.n_v1, device=device)
            return torch.zeros(self.n_v2, device=device), zero, self.ais_pool.clone()

        # --- Fibonacci temporal resonance ---
        # For each delay tau, retrieve V1 state from tau frames ago.
        # This IS Takens embedding: V2 sees a trajectory window of V1.
        total_res = torch.zeros(self.n_v2, device=device)
        for i, tau in enumerate(self.taus):
            idx = -(tau + 1)
            if abs(idx) > len(self.v1_buffer):
                continue
            v_t = F.normalize(self.v1_buffer[idx].to(device), p=2, dim=0)  # (N1,)
            dot = (v_t.unsqueeze(0) * self.m[:, i, :]).sum(1)               # (N2,)
            total_res += dot ** 2
        avg_res = total_res / len(self.taus)

        # --- AIS integration (1D — no spatial structure in V2) ---
        self.ais_pool = (self.ais_pool * (1.0 - self.ais_integration)
                       + avg_res        *        self.ais_integration)

        pre_spike_pool = self.ais_pool.clone()  # save for raster visualization

        # --- Threshold with global CEMI ---
        thresh = max(0.002, self.base_thresh * (1.0 - self.global_cemi * self.cemi_coupling))

        # --- Top-k sparse firing ---
        k = max(1, int(self.n_v2 * self.sparsity))
        _, topk_idx  = torch.topk(self.ais_pool, k)
        sparse_mask  = torch.zeros(self.n_v2, device=device, dtype=torch.bool)
        sparse_mask[topk_idx] = True
        spikes       = ((self.ais_pool > thresh) & sparse_mask).float()
        self.ais_pool = self.ais_pool * (1.0 - spikes)

        self.neuron_rate  = self.neuron_rate * 0.92 + spikes * 0.08
        self.global_cemi  = self.global_cemi * 0.85 + (spikes.mean().item() / self.sparsity) * 0.15

        # --- STDP ---
        self.pre_trace = self.pre_trace * self.stdp_tau + avg_res * (1.0 - self.stdp_tau)
        active_v2      = spikes > 0

        if self.stdp_Aplus > 0 and active_v2.any():
            for i, tau in enumerate(self.taus):
                idx = -(tau + 1)
                if abs(idx) > len(self.v1_buffer): continue
                v_t = F.normalize(self.v1_buffer[idx].to(device), p=2, dim=0)
                for n in torch.where(active_v2)[0]:
                    m_cur = self.m[n, i]
                    delta = v_t - torch.dot(v_t, m_cur) * m_cur
                    self.m.data[n, i] += delta * (
                        self.stdp_Aplus * self.pre_trace[n]
                      - self.stdp_Aplus * self.stdp_Aminus_ratio * self.post_trace[n])
            self.m.data = F.normalize(self.m.data, p=2, dim=2)

        self.post_trace = self.post_trace * self.stdp_tau + active_v2.float() * (1.0 - self.stdp_tau)

        # --- V2 → V1 top-down reconstruction ---
        # V2 says: "given the temporal patterns I just recognized,
        # which V1 neurons do I predict should be active?"
        predicted_v1 = torch.zeros(self.n_v1, device=device)
        n_active = spikes.sum().item()
        if n_active > 0:
            for i in range(len(self.taus)):
                predicted_v1 += (spikes.unsqueeze(1) * self.m[:, i, :]).sum(0)
            predicted_v1 = predicted_v1 / (n_active * len(self.taus))
            mn, mx = predicted_v1.min(), predicted_v1.max()
            predicted_v1 = (predicted_v1 - mn) / (mx - mn + 1e-8)

        return spikes, predicted_v1, pre_spike_pool


# =====================================================================
# HELPERS
# =====================================================================
def extract_frequency_streams(frame_gray):
    low  = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    high = cv2.equalizeHist(cv2.subtract(frame_gray, low))
    return torch.stack([
        torch.from_numpy(low ).float() / 255.0,
        torch.from_numpy(high).float() / 255.0,
    ], dim=0).unsqueeze(0).to(device)


def render_v1_sheet(v1_local_cemi, v1_rate, v2_feedback, res):
    """V1 neural sheet: PLASMA CEMI heatmap + cyan V2-feedback glow + white dots."""
    cg = v1_local_cemi.reshape(GRID_H, GRID_W)
    rg = v1_rate.reshape(GRID_H, GRID_W)
    fg = v2_feedback.reshape(GRID_H, GRID_W)

    base = cv2.applyColorMap(
        cv2.resize((np.clip(cg / (cg.max()+1e-6), 0, 1)*255).astype(np.uint8),
                   (res, res), interpolation=cv2.INTER_CUBIC),
        cv2.COLORMAP_PLASMA)

    # Cyan glow = V2 top-down feedback
    fb_up = cv2.resize((np.clip(fg / (fg.max()+1e-6), 0, 1)*255).astype(np.uint8),
                       (res, res), interpolation=cv2.INTER_CUBIC)
    glow  = np.zeros((res, res, 3), dtype=np.uint8)
    glow[:,:,1] = fb_up           # green channel → cyan
    glow[:,:,0] = fb_up // 3
    base = cv2.addWeighted(base, 0.65, glow, 0.35, 0)

    # White dots = active V1 neurons
    rmx = rg.max() + 1e-6
    for i in range(GRID_H):
        for j in range(GRID_W):
            if rg[i, j] / rmx > 0.20:
                px, py = int(j/GRID_W*res), int(i/GRID_H*res)
                br = int(rg[i, j] / rmx * 255)
                cv2.circle(base, (px, py), 3, (br, br, br), -1)
    return base


def render_v2_raster(history, width, height):
    """Scrolling V2 AIS pool heatmap — reveals integrate-and-fire dynamics."""
    if len(history) < 2:
        return np.zeros((height, width, 3), dtype=np.uint8)
    arr      = np.array(history)                   # (T, N2)
    arr_norm = arr / (arr.max() + 1e-6)
    img      = cv2.resize((arr_norm.T * 255).astype(np.uint8),
                          (min(len(history), width), height),
                          interpolation=cv2.INTER_NEAREST)
    if img.shape[1] < width:
        pad  = np.zeros((height, width - img.shape[1]), dtype=np.uint8)
        img  = np.hstack((pad, img))
    return cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)


# =====================================================================
# APPLICATION
# =====================================================================
class HierarchicalCortexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hierarchical Cortex — V1 → V2")
        self.root.configure(bg="#0e0e12")

        self.RES = 128
        self.v1  = V1Cortex().to(device)
        self.v2  = V2Cortex().to(device)
        self.cap = cv2.VideoCapture(0)

        # Rolling buffer for V2 raster (stores pre-spike AIS pool values)
        self.v2_raster_buf = deque(maxlen=300)

        self._setup_ui()
        self._loop()

    # ------------------------------------------------------------------
    def _slider(self, parent, label, lo, hi, default, cb, color="#00ffcc"):
        f = tk.Frame(parent, bg=parent.cget('bg'))
        f.pack(side=tk.LEFT, padx=5)
        tk.Label(f, text=label, bg=parent.cget('bg'), fg=color,
                 font=("Courier", 8, "bold")).pack()
        s = tk.Scale(f, from_=lo, to=hi, resolution=0.01, orient=tk.HORIZONTAL,
                     bg="#2a2a30", fg="white", length=108,
                     highlightthickness=0, troughcolor="#333340")
        s.set(default)
        s.pack()
        s.config(command=lambda v: cb(float(v)))

    def _setup_ui(self):
        # --- V1 sliders ---
        r1 = tk.Frame(self.root, bg="#1a1a20", pady=4)
        r1.pack(fill=tk.X)
        tk.Label(r1, text="V1 ▸", bg="#1a1a20", fg="#aaaaaa",
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT, padx=6)

        def sv1(attr): return lambda v: setattr(self.v1, attr, v)
        self._slider(r1, "AIS",       0.01, 1.0,  0.30, sv1('ais_integration'))
        self._slider(r1, "Rajapinta", 0.01, 0.5,  0.05, sv1('base_thresh'))
        self._slider(r1, "CEMI Coup", 0.0,  1.0,  0.60, sv1('cemi_coupling'))
        self._slider(r1, "CEMI σ",   0.05, 1.0,  0.30, sv1('cemi_sigma'))
        self._slider(r1, "STDP A+",   0.0,  0.3,  0.10, sv1('stdp_Aplus'))
        self._slider(r1, "LTD/LTP",   0.0,  2.0,  0.60, sv1('stdp_Aminus_ratio'))
        self._slider(r1, "Pred Str",  0.0,  1.0,  0.60, sv1('pred_strength'))

        # --- V2 sliders ---
        r2 = tk.Frame(self.root, bg="#12121a", pady=4)
        r2.pack(fill=tk.X)
        tk.Label(r2, text="V2 ▸", bg="#12121a", fg="#aaaaaa",
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT, padx=6)

        def sv2(attr): return lambda v: setattr(self.v2, attr, v)
        self._slider(r2, "AIS",       0.01, 1.0,  0.20, sv2('ais_integration'),   "#ffaa44")
        self._slider(r2, "Threshold", 0.01, 0.3,  0.01, sv2('base_thresh'),        "#ffaa44")
        self._slider(r2, "STDP A+",   0.0,  0.3,  0.08, sv2('stdp_Aplus'),         "#ffaa44")
        self._slider(r2, "LTD/LTP",   0.0,  2.0,  0.50, sv2('stdp_Aminus_ratio'),  "#ffaa44")
        self._slider(r2, "Sparsity",  0.02, 0.3,  0.10, sv2('sparsity'),           "#ffaa44")
        self._slider(r2, "V2→V1 Fbk", 0.0, 1.0,  0.40,
                     lambda v: setattr(self.v1, 'v2_feedback_strength', v), "#ff88ff")

        self.info = tk.Label(r2, text="V1:0.000  V2:0.000",
                             bg="#12121a", fg="#ff5555", font=("Courier", 11, "bold"))
        self.info.pack(side=tk.RIGHT, padx=10)

        # --- Main 4-panel display ---
        self.disp = tk.Label(self.root, bg="#0e0e12")
        self.disp.pack(fill=tk.BOTH, expand=True, pady=2)

        # --- V2 temporal raster ---
        tk.Label(self.root,
                 text="V2 TEMPORAL RASTER  [ time → | V2 neurons ↑ | "
                      "bright = spike, glow = charging ]",
                 bg="#0e0e12", fg="#ffaa44", font=("Courier", 8)).pack()
        self.raster_lbl = tk.Label(self.root, bg="#0e0e12")
        self.raster_lbl.pack(pady=2)

        # --- Legend ---
        leg = tk.Frame(self.root, bg="#1a1a20", pady=3)
        leg.pack(fill=tk.X)
        items = [
            ("■ cyan glow = V2 predicts this V1 neuron active",  "#00ffcc"),
            ("■ V2 raster = temporal memory / sequence learning", "#ffaa44"),
            ("■ V2 AIS slow → learns long rhythms",              "#ff88ff"),
            ("■ V2→V1 feedback = anticipation",                   "#88aaff"),
        ]
        for txt, col in items:
            tk.Label(leg, text=txt + "   ", bg="#1a1a20", fg=col,
                     font=("Courier", 8)).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    def _loop(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              (self.RES, self.RES))
            x_t = extract_frequency_streams(gray)

            # ---- V1 step ----
            v1_spikes, v1_error, v1_local_cemi = self.v1.step(x_t)
            v1_rate_vec = v1_spikes.mean(dim=(0, 2, 3))  # (N1,) — feed to V2

            # ---- V2 step ----
            v2_spikes, pred_v1, v2_ais = self.v2.step(v1_rate_vec)

            # Inject V2 prediction into V1 for next frame
            self.v1.v2_feedback = pred_v1

            # ---- Info bar ----
            self.info.config(
                text=f"V1 rate:{self.v1.neuron_rate.mean().item():.3f}  "
                     f"V2 cemi:{self.v2.global_cemi:.3f}  "
                     f"V1 spk:{v1_spikes.mean()*100:.1f}%  "
                     f"V2 spk:{v2_spikes.mean()*100:.1f}%")

            # ---- Panel 1: Reality ----
            p1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # ---- Panel 2: V1 Prediction Error ----
            err = v1_error[0].mean(0).cpu().numpy()
            p2  = cv2.applyColorMap(
                    (np.clip(err, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)

            # ---- Panel 3: V1 Neural Sheet + V2 feedback glow ----
            p3 = render_v1_sheet(
                    v1_local_cemi.cpu().numpy(),
                    self.v1.neuron_rate.cpu().numpy(),
                    pred_v1.cpu().numpy(),
                    self.RES)

            # ---- Panel 4: V2→V1 top-down image ----
            # Weight V1's last spike pattern by V2's predicted V1 activity.
            # This shows WHAT V2 expects V1 to currently be processing.
            pred_spikes = v1_spikes * pred_v1.view(1, N1, 1, 1)
            pred_img    = self.v1._reconstruct(pred_spikes)
            rl, rh = pred_img[0, 0].cpu().numpy(), pred_img[0, 1].cpu().numpy()
            mind   = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            mind[:, :, 0] = rh * 2.0
            mind[:, :, 1] = (rh + rl) * 0.5
            mind[:, :, 2] = rl * 1.5
            p4 = (np.clip(mind, 0, 1) * 255).astype(np.uint8)

            # ---- Assemble main row ----
            row = np.hstack((p1, p2, p3, p4))
            row = cv2.resize(row, (1200, 300), interpolation=cv2.INTER_NEAREST)
            for txt, x in [("1. REALITY",              8),
                           ("2. V1 PREDICTION ERROR",  308),
                           ("3. V1 SHEET + V2 FEEDBACK", 608),
                           ("4. V2→V1 PREDICTION",     908)]:
                cv2.putText(row, txt, (x, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)))
            self.disp.imgtk = img
            self.disp.configure(image=img)

            # ---- V2 raster ----
            self.v2_raster_buf.append(v2_ais.cpu().numpy())
            if len(self.v2_raster_buf) > 4:
                raster = render_v2_raster(list(self.v2_raster_buf), 1200, 80)
                rim = ImageTk.PhotoImage(
                        Image.fromarray(cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)))
                self.raster_lbl.imgtk = rim
                self.raster_lbl.configure(image=rim)

        self.root.after(20, self._loop)


if __name__ == "__main__":
    root = tk.Tk()
    HierarchicalCortexApp(root)
    root.mainloop()
