import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predictive Cortex booting on {device}...")

GRID_H, GRID_W = 16, 8  # 128 neurons arranged on a physical 2D sheet

# =====================================================================
# 1. THE PREDICTIVE CORTEX
#    Three additions beyond ZetaGratingCortex:
#    (a) Predictive coding  — neurons respond to prediction ERROR, not raw input
#    (b) Local spatial CEMI — ephaptic field decays with distance → traveling waves
#    (c) STDP               — asymmetric temporal plasticity (pre→post ≠ post→pre)
# =====================================================================

class PredictiveCortex(nn.Module):
    def __init__(self, in_channels=2, num_neurons=128):
        super().__init__()
        self.num_neurons = num_neurons
        self.taus        = [1, 2, 3, 5, 8, 13, 21]   # Fibonacci dilation scales
        self.patch_dim   = in_channels * 9

        # Grating templates
        m_init = torch.randn(num_neurons, len(self.taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)

        # State
        self.ais_pool   = None
        self.prediction = None                                    # top-down prediction from t-1

        # STDP traces (one scalar per neuron)
        self.pre_trace  = torch.zeros(num_neurons, device=device) # decaying resonance trace
        self.post_trace = torch.zeros(num_neurons, device=device) # decaying spike trace

        # Per-neuron firing rate (for local CEMI)
        self.neuron_rate = torch.zeros(num_neurons, device=device)

        # Neuron positions on the 2D sheet
        positions = [[i / GRID_H, j / GRID_W]
                     for i in range(GRID_H) for j in range(GRID_W)]
        pos = torch.tensor(positions[:num_neurons], dtype=torch.float32)
        self.register_buffer('positions', pos)

        # Precompute pairwise squared distances (N×N) — used every step
        diff  = pos.unsqueeze(0) - pos.unsqueeze(1)   # N×N×2
        dist2 = (diff ** 2).sum(-1)                   # N×N
        self.register_buffer('dist2_matrix', dist2)

        # --- EXPOSED PHYSICS PARAMETERS ---
        self.ais_integration    = 0.30   # AIS fill rate
        self.base_thresh        = 0.05   # Static Rajapinta boundary
        self.cemi_coupling      = 0.60   # How much local CEMI lowers threshold
        self.cemi_sigma         = 0.30   # Spatial reach of ephaptic field
        self.sparsity           = 0.05   # k/N winner-take-all fraction
        self.stdp_Aplus         = 0.10   # LTP rate
        self.stdp_Aminus_ratio  = 0.60   # LTD/LTP asymmetry (>1 → forget fast)
        self.stdp_tau           = 0.85   # Trace decay (both pre and post)
        self.pred_strength      = 0.70   # Weight on prediction error vs raw input

    # ------------------------------------------------------------------
    # Local CEMI: weighted average of neighbours' firing rates
    # ------------------------------------------------------------------
    def compute_local_cemi(self):
        sigma2  = max(self.cemi_sigma ** 2, 1e-4)
        weights = torch.exp(-self.dist2_matrix / (2 * sigma2))   # N×N
        weights = weights / (weights.sum(1, keepdim=True) + 1e-6)
        return (weights * self.neuron_rate.unsqueeze(0)).sum(1)   # (N,)

    # ------------------------------------------------------------------
    # One step of the full predictive cortex
    # ------------------------------------------------------------------
    def step(self, image_tensor):
        B, C, H, W = image_tensor.shape

        if self.ais_pool is None or self.ais_pool.shape[-2:] != (H, W):
            self.ais_pool   = torch.zeros(1, self.num_neurons, H, W, device=device)
            self.prediction = torch.zeros(B, C, H, W, device=device)

        # ---- (a) PREDICTIVE CODING ----
        # The cortex drives itself on the UPWARD surprise, not the raw image.
        # pred_strength=0 → pure raw input  (original ZetaGrating behaviour)
        # pred_strength=1 → pure prediction error
        pred_error = torch.relu(image_tensor - self.prediction)
        driving    = (1.0 - self.pred_strength) * image_tensor \
                   +        self.pred_strength  * pred_error

        # ---- Fibonacci resonance on driving signal ----
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        for i, tau in enumerate(self.taus):
            pad      = tau
            padded   = F.pad(driving, (pad, pad, pad, pad), mode='reflect')
            unfolded = F.unfold(padded, kernel_size=3, dilation=tau)
            v        = F.normalize(unfolded.view(B, self.patch_dim, H, W), p=2, dim=1)
            dot      = torch.einsum('b c h w, n c -> b n h w', v, self.m[:, i, :])
            total_res += dot ** 2
        avg_res = total_res / len(self.taus)

        # ---- AIS integration ----
        self.ais_pool = (self.ais_pool * (1.0 - self.ais_integration)
                       + avg_res       *        self.ais_integration)

        # ---- (b) LOCAL SPATIAL CEMI ----
        # Each neuron's threshold is lowered by its neighbours' activity,
        # weighted by spatial distance → traveling excitability waves.
        local_cemi = self.compute_local_cemi()                          # (N,)
        cemi_bc    = local_cemi.view(1, self.num_neurons, 1, 1)
        dyn_thresh = self.base_thresh * torch.clamp(
                        1.0 - cemi_bc * self.cemi_coupling, 0.02, 1.0)

        # ---- Sparse firing ----
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_idx = torch.topk(self.ais_pool, k, dim=1)
        sparse_mask = torch.zeros_like(self.ais_pool, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_idx, True)
        spikes = ((self.ais_pool > dyn_thresh) & sparse_mask).float()
        self.ais_pool = self.ais_pool * (1.0 - spikes)           # refractory reset

        # Update per-neuron firing rate (slow average)
        self.neuron_rate = (self.neuron_rate * 0.92
                           + spikes.mean(dim=(0, 2, 3)) * 0.08)

        # ---- (c) STDP — asymmetric temporal plasticity ----
        # pre_trace  : decaying trace of resonance evidence (pre-synaptic)
        # post_trace : decaying trace of spike history     (post-synaptic)
        #
        # On spike  → LTP proportional to pre_trace  (causal: input → spike)
        # On input  → LTD proportional to post_trace (acausal: spike → input)
        # When LTD/LTP ratio > 1 the system unlearns recent associations fast.

        res_per_neuron  = avg_res.mean(dim=(0, 2, 3))
        self.pre_trace  = (self.pre_trace  * self.stdp_tau
                          + res_per_neuron * (1.0 - self.stdp_tau))

        spikes_flat    = spikes.view(self.num_neurons, -1)
        active_neurons = spikes_flat.sum(dim=1) > 0

        if self.stdp_Aplus > 0 and active_neurons.any():
            for i, tau in enumerate(self.taus):
                pad      = tau
                padded   = F.pad(driving, (pad, pad, pad, pad), mode='reflect')
                v_norm   = F.normalize(
                    F.unfold(padded, kernel_size=3, dilation=tau)
                     .view(B, self.patch_dim, H, W), p=2, dim=1)[0]

                for n in torch.where(active_neurons)[0]:
                    mask = spikes[0, n:n+1, :, :]
                    if mask.sum() == 0:
                        continue
                    v_m    = (v_norm * mask).sum(dim=(1, 2)) / mask.sum()
                    m_curr = self.m[n, i]
                    delta  = v_m - torch.dot(v_m, m_curr) * m_curr  # Oja direction

                    ltp = self.stdp_Aplus                             * self.pre_trace[n]  * delta
                    ltd = self.stdp_Aplus * self.stdp_Aminus_ratio    * self.post_trace[n] * delta
                    self.m.data[n, i] += ltp - ltd

            self.m.data = F.normalize(self.m.data, p=2, dim=2)

        # Update post-trace after weight update
        self.post_trace = (self.post_trace    * self.stdp_tau
                          + active_neurons.float() * (1.0 - self.stdp_tau))

        # ---- Generate new top-down prediction ----
        self.prediction = self._reconstruct(spikes)

        return avg_res.mean(dim=1, keepdim=True), spikes, pred_error, local_cemi

    def _reconstruct(self, spikes):
        B, N, H, W = spikes.shape
        recon      = torch.zeros(B, 2, H, W, device=device)
        spikes_flat = spikes.view(B, N, H * W)
        for i, tau in enumerate(self.taus):
            m_tau   = self.m[:, i, :]
            patches = torch.bmm(spikes_flat.transpose(1, 2),
                                m_tau.unsqueeze(0).expand(B, -1, -1)).transpose(1, 2)
            pad     = tau
            folded  = F.fold(patches,
                             output_size=(H + pad*2, W + pad*2),
                             kernel_size=3, dilation=tau)
            recon  += folded[:, :, pad:-pad, pad:-pad]
        rmin = recon.amin(dim=(2, 3), keepdim=True)
        rmax = recon.amax(dim=(2, 3), keepdim=True)
        return (recon - rmin) / (rmax - rmin + 1e-8)


# =====================================================================
# 2. FRONT-END
# =====================================================================

def extract_frequency_streams(frame_gray):
    low_freq  = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    high_freq = cv2.equalizeHist(cv2.subtract(frame_gray, low_freq))
    return torch.stack([
        torch.from_numpy(low_freq ).float() / 255.0,
        torch.from_numpy(high_freq).float() / 255.0
    ], dim=0).unsqueeze(0).to(device)


def render_neural_sheet(local_cemi_np, rate_np, res):
    """16×8 neuron sheet → coloured CEMI heatmap with active-neuron overlay."""
    cemi_grid = local_cemi_np.reshape(GRID_H, GRID_W)
    rate_grid = rate_np.reshape(GRID_H, GRID_W)

    mx = cemi_grid.max() + 1e-6
    cemi_up = cv2.resize((np.clip(cemi_grid / mx, 0, 1) * 255).astype(np.uint8),
                         (res, res), interpolation=cv2.INTER_CUBIC)
    panel = cv2.applyColorMap(cemi_up, cv2.COLORMAP_PLASMA)

    rate_mx = rate_grid.max() + 1e-6
    for i in range(GRID_H):
        for j in range(GRID_W):
            if rate_grid[i, j] / rate_mx > 0.25:
                px = int(j / GRID_W * res)
                py = int(i / GRID_H * res)
                br = int(rate_grid[i, j] / rate_mx * 255)
                cv2.circle(panel, (px, py), 3, (br, br, br), -1)
    return panel


# =====================================================================
# 3. GUI
# =====================================================================

class PredictiveCortexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Cortex — GAIT Full Architecture")
        self.root.configure(bg="#0e0e12")

        self.RES    = 128
        self.cortex = PredictiveCortex(num_neurons=128).to(device)
        self.cap    = cv2.VideoCapture(0)
        self._setup_ui()
        self._loop()

    def _make_slider(self, parent, label, lo, hi, default, attr, color="#00ffcc"):
        f = tk.Frame(parent, bg="#1a1a20")
        f.pack(side=tk.LEFT, padx=8)
        tk.Label(f, text=label, bg="#1a1a20", fg=color,
                 font=("Courier", 8, "bold")).pack()
        s = tk.Scale(f, from_=lo, to=hi, resolution=0.01,
                     orient=tk.HORIZONTAL, bg="#2a2a30", fg="white",
                     length=120, highlightthickness=0, troughcolor="#333340")
        s.set(default)
        s.pack()
        s.config(command=lambda v, a=attr: setattr(self.cortex, a, float(v)))

    def _setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#1a1a20", pady=6)
        ctrl.pack(fill=tk.X)

        self._make_slider(ctrl, "AIS Integ.",    0.01, 1.0,  0.30, "ais_integration")
        self._make_slider(ctrl, "Rajapinta",     0.01, 0.5,  0.05, "base_thresh")
        self._make_slider(ctrl, "CEMI Coupling", 0.0,  1.0,  0.60, "cemi_coupling")
        self._make_slider(ctrl, "CEMI σ (local)",0.05, 1.0,  0.30, "cemi_sigma",
                          color="#ff88ff")
        self._make_slider(ctrl, "STDP A+",       0.0,  0.3,  0.10, "stdp_Aplus",
                          color="#88ffaa")
        self._make_slider(ctrl, "LTD/LTP ratio", 0.0,  2.0,  0.60, "stdp_Aminus_ratio",
                          color="#88ffaa")
        self._make_slider(ctrl, "Pred Strength", 0.0,  1.0,  0.70, "pred_strength",
                          color="#ffaa44")

        self.info = tk.Label(ctrl, text="CEMI: 0.000", bg="#1a1a20",
                             fg="#ff5555", font=("Courier", 12, "bold"))
        self.info.pack(side=tk.RIGHT, padx=12)

        self.disp = tk.Label(self.root, bg="#0e0e12")
        self.disp.pack(fill=tk.BOTH, expand=True, pady=4)

        # Legend strip
        legend = tk.Frame(self.root, bg="#1a1a20", pady=3)
        legend.pack(fill=tk.X)
        notes = [
            ("■ CEMI σ small → traveling waves   ", "#ff88ff"),
            ("■ pred=1.0 → only novelty drives   ", "#ffaa44"),
            ("■ LTD/LTP>1 → fast forgetting   ",    "#88ffaa"),
            ("■ low AIS → slow metronome   ",        "#00ffcc"),
        ]
        for txt, col in notes:
            tk.Label(legend, text=txt, bg="#1a1a20", fg=col,
                     font=("Courier", 8)).pack(side=tk.LEFT, padx=6)

    def _loop(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              (self.RES, self.RES))
            x_t  = extract_frequency_streams(gray)

            _, spikes, pred_error, local_cemi = self.cortex.step(x_t)

            global_cemi = self.cortex.neuron_rate.mean().item()
            active_pct  = spikes.mean().item() * 100
            self.info.config(text=f"CEMI:{global_cemi:.3f}  spikes:{active_pct:.1f}%")

            # Panel 1 — Reality
            p1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Panel 2 — Prediction Error (what surprised the cortex this frame)
            err = pred_error[0].mean(0).cpu().numpy()
            p2  = cv2.applyColorMap(
                    (np.clip(err, 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT)

            # Panel 3 — Neural sheet with local CEMI traveling waves
            p3 = render_neural_sheet(
                    local_cemi.cpu().numpy(),
                    self.cortex.neuron_rate.cpu().numpy(),
                    self.RES)

            # Panel 4 — Top-down prediction (what the cortex expects)
            pred  = self.cortex.prediction
            rl, rh = pred[0, 0].cpu().numpy(), pred[0, 1].cpu().numpy()
            mind  = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            mind[:, :, 0] = rh * 2.0
            mind[:, :, 1] = (rh + rl) * 0.5
            mind[:, :, 2] = rl * 1.5
            p4 = (np.clip(mind, 0, 1) * 255).astype(np.uint8)

            # Assemble
            row = np.hstack((p1, p2, p3, p4))
            row = cv2.resize(row, (1200, 320), interpolation=cv2.INTER_NEAREST)

            for txt, x in [("1. REALITY",          8),
                           ("2. PREDICTION ERROR", 308),
                           ("3. NEURAL SHEET",     608),
                           ("4. TOP-DOWN PRED",    908)]:
                cv2.putText(row, txt, (x, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)))
            self.disp.imgtk = img
            self.disp.configure(image=img)

        self.root.after(20, self._loop)


if __name__ == "__main__":
    root = tk.Tk()
    PredictiveCortexApp(root)
    root.mainloop()