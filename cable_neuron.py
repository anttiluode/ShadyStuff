"""
BiologicalCableNeuron  —  clean, correct, runnable
===================================================
Fixes every crash from the previous attempts:

BUG 1 (silent, fatal): L[1:-1, 1:-1] = -(...)
  Sets a 2D BLOCK, not the diagonal. Laplacian was completely wrong.
  Fix: build diagonal explicitly with torch.diag().

BUG 2: AIS center as raw nn.Parameter with hard index
  Gradient detaches because integer indexing is not differentiable.
  Fix: ais_pos → soft Gaussian over all compartments (always in graph).

BUG 3: Fixed spike temperature
  Sigmoid near 0 or 1 → gradient ≈ 0 → stuck.
  Fix: anneal spike_temp 10→1 over training (smooth early, sharp late).

BUG 4 (v2 100%→56% oscillation):
  BCELoss on sigmoid output → gradient flat at boundaries.
  Fix: return logits, use BCEWithLogitsLoss throughout.

Architecture
  N=32 cable compartments  (soma = compartment 0)
  IMEX semi-implicit PDE   (diffusion+leak implicit, ion channels explicit)
  AIS grating              (Nav at cosine peaks, Kv at troughs, Gaussian envelope)
  Gamma gate               (40 Hz fixed, phase learnable)
  Temporal XOR benchmark   (requires physical wave interference, linear model fails)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Physical constants (SI-normalised, mV / ms) ─────────────────────────────
E_NA  =  55.0   # mV  sodium reversal
E_K   = -90.0   # mV  potassium reversal
CM    =   1.0   # μF/cm²  membrane capacitance
RA    =   0.5   # Ω·cm    axial resistance  (lower = faster signal spread)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core neuron model                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CableNeuron(nn.Module):
    """
    Dendrite  →  AIS grating  →  Gamma-gated soma  →  logit output

    Learnable parameters
    --------------------
    log_d        diameter profile     (N,)    – controls axial conductance
    log_gl       leak conductance     (N,)    – controls membrane time constant
    ais_pos      AIS soft centre      scalar  – sigmoid→[0,N-1]
    ais_width    AIS half-width       scalar  – in compartments
    log_g_nav    Nav amplitude        scalar
    log_g_kv     Kv amplitude         scalar
    gamma_phase  gamma phase offset   scalar
    gamma_alpha  gamma coupling       scalar  – [0,1] gating strength
    log_thr      spike threshold      scalar
    """

    def __init__(self, N: int = 32, dt: float = 0.5, dx: float = 2.0):
        super().__init__()
        self.N  = N
        self.dt = dt     # ms per step
        self.dx = dx     # μm per compartment

        # Cable geometry
        self.log_d   = nn.Parameter(torch.ones(N)  * np.log(2.0))
        self.log_gl  = nn.Parameter(torch.ones(N)  * np.log(0.008))   # τ_m ≈ 125 ms

        # AIS
        self.ais_pos    = nn.Parameter(torch.tensor(0.0))              # will be sigmoid→[0,N-1]
        self.ais_width  = nn.Parameter(torch.tensor(5.0))              # compartments
        self.log_g_nav  = nn.Parameter(torch.tensor(np.log(0.5)))
        self.log_g_kv   = nn.Parameter(torch.tensor(np.log(0.2)))
        # 190 nm periodicity → coarse grating at compartment scale
        self.register_buffer("k_grating", torch.tensor(2.0 * np.pi / 4.0))

        # Gamma gate  (40 Hz fixed; phase + coupling learned)
        self.gamma_phase = nn.Parameter(torch.tensor(0.0))
        self.gamma_alpha = nn.Parameter(torch.tensor(0.5))
        freq_rad = 2.0 * np.pi * 40.0 * (dt / 1000.0)                # rad / step
        self.register_buffer("gamma_freq", torch.tensor(freq_rad))

        # Threshold
        self.log_thr = nn.Parameter(torch.tensor(np.log(8.0)))

        # Spike temperature  (annealed externally during training)
        self.spike_temp = 8.0

        # Compartment positions (constant)
        self.register_buffer("xpos", torch.arange(N, dtype=torch.float32))

    # ── Cable operators ──────────────────────────────────────────────────────

    def _cable_matrices(self):
        """
        Build the IMEX (semi-implicit) matrices A, B for one step:
            A V_new = B V_old + I_active + I_ext

        A = (Cm/dt) I  –  ½ L  +  ½ G_L
        B = (Cm/dt) I  +  ½ L  –  ½ G_L

        L is the cable Laplacian (tridiagonal, sealed-end BC).
        G_L = diag(leak conductances).

        CRITICAL FIX: diagonal built explicitly via torch.diag(),
        NOT via 2D slice L[1:-1, 1:-1] which sets a block, not a diagonal.
        """
        d  = torch.exp(self.log_d)          # (N,)
        gl = torch.exp(self.log_gl)          # (N,)

        # Axial conductance between neighbouring compartments
        # g_ax[i] = conductance between compartment i and i+1
        d_mid = 0.5 * (d[:-1] + d[1:])                       # (N-1,)
        g_ax  = d_mid / (4.0 * RA * self.dx ** 2)             # (N-1,)

        # Laplacian diagonal: sealed ends, interior = −(left + right)
        diag_L = torch.zeros(self.N, device=d.device)
        diag_L[0]     = -g_ax[0]
        diag_L[-1]    = -g_ax[-1]
        diag_L[1:-1]  = -(g_ax[:-1] + g_ax[1:])              # ← FIX: 1D, not 2D

        # Assemble L (tridiagonal)
        L = (torch.diag(g_ax,  diagonal=+1) +
             torch.diag(g_ax,  diagonal=-1) +
             torch.diag(diag_L))

        I_N  = torch.eye(self.N, device=d.device)
        scl  = CM / self.dt
        GL   = torch.diag(gl)

        A = scl * I_N - 0.5 * L + 0.5 * GL
        B = scl * I_N + 0.5 * L - 0.5 * GL
        return A, B

    # ── AIS grating ──────────────────────────────────────────────────────────

    def _ais_conductances(self):
        """
        Soft Gaussian AIS envelope × interleaved Nav/Kv grating.

        CRITICAL FIX: ais_pos is a continuous parameter → sigmoid maps
        it smoothly to [0, N-1].  No integer indexing, always in the graph.
        """
        center   = torch.sigmoid(self.ais_pos) * (self.N - 1)
        width    = self.ais_width.clamp(min=2.0, max=14.0)
        envelope = torch.exp(-0.5 * ((self.xpos - center) / width) ** 2)

        phase   = self.k_grating * self.xpos
        g_nav   = torch.exp(self.log_g_nav) * envelope * (0.5 + 0.5 * torch.cos(phase))
        g_kv    = torch.exp(self.log_g_kv)  * envelope * (0.5 - 0.5 * torch.cos(phase))
        return g_nav, g_kv

    # ── Ion channel kinetics (steady-state) ──────────────────────────────────

    @staticmethod
    def m_inf(V):
        """Nav activation gate – sigmoid around 15 mV"""
        return torch.sigmoid((V - 15.0) / 5.0)

    @staticmethod
    def n_inf(V):
        """Kv activation gate – sigmoid around 20 mV (slower)"""
        return torch.sigmoid((V - 20.0) / 8.0)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, I_ext, return_traces: bool = False):
        """
        Parameters
        ----------
        I_ext        (batch, T, N) – external current injections [μA/cm²]
        return_traces  bool – also return voltage traces (for visualisation)

        Returns
        -------
        logits       (batch,) – pre-sigmoid class score; positive = class 1
        """
        batch, T, N = I_ext.shape
        assert N == self.N, f"I_ext last dim {N} ≠ N={self.N}"

        A, B        = self._cable_matrices()          # (N, N)
        g_nav, g_kv = self._ais_conductances()        # (N,)

        V           = torch.zeros(batch, N, device=I_ext.device)
        soma_trace  = torch.zeros(batch, T, device=I_ext.device)
        V_list      = [] if return_traces else None

        for t in range(T):
            # ── Active currents (explicit at current V) ──────────────────
            I_Na  = g_nav * self.m_inf(V) * (E_NA - V)    # (batch, N) inward
            I_Kv  = g_kv  * self.n_inf(V) * (V   - E_K)   # (batch, N) outward
            # Hard clamp to prevent numerical explosion
            I_act = torch.clamp(I_Na - I_Kv, -30.0, 30.0)

            # ── Semi-implicit step: A V_new = B V_old + I_act + I_ext ────
            RHS   = V @ B.T + I_act + I_ext[:, t, :]      # (batch, N)
            # solve: A (N×N), RHS.T (N×batch) → result (N×batch) → .T
            V     = torch.linalg.solve(A, RHS.T).T         # (batch, N)

            # Voltage clamp (biological realism + gradient stability)
            V     = V.clamp(-80.0, 80.0)

            soma_trace[:, t] = V[:, 0]                     # soma = compartment 0

            if return_traces:
                V_list.append(V.detach().clone())

        # ── Gamma phase gate ──────────────────────────────────────────────
        t_ax    = torch.arange(T, device=I_ext.device, dtype=torch.float32)
        gamma   = 0.5 + 0.5 * torch.cos(self.gamma_freq * t_ax + self.gamma_phase)
        alpha   = self.gamma_alpha.clamp(0.0, 1.0)
        gated   = soma_trace * (1.0 - alpha + alpha * gamma.unsqueeze(0))

        # ── Logit readout: top-k mean of gated soma voltage ───────────────
        k       = max(1, T // 12)
        peak_v  = torch.topk(gated, k=k, dim=1).values.mean(dim=1)   # (batch,)
        thr     = torch.exp(self.log_thr)
        logits  = (peak_v - thr) / self.spike_temp                    # (batch,)

        if return_traces:
            V_traces = torch.stack(V_list, dim=1)   # (batch, T, N)
            return logits, V_traces, g_nav.detach(), g_kv.detach(), gated.detach()
        return logits


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Temporal XOR dataset                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_xor_dataset(N=32, T=200, dt=0.5, n=200,
                     pos_a=6, pos_b=24,
                     t_first=30, gap=100,
                     amp=4.0, dur=15, device=DEVICE):
    """
    Class 1: input A fires at t_first, then B fires at t_first+gap
    Class 0: input B fires at t_first, then A fires at t_first+gap

    Gap ~ 50 ms  (= gap*dt),  τ_m ≈ 125 ms  →  first pulse lingers when second arrives.
    The cable creates different interference patterns for the two orderings.
    A linear model cannot separate them: physical nonlinearity is required.
    """
    I = torch.zeros(n, T, N, device=device)
    y = torch.zeros(n, device=device)

    for i in range(n):
        cls = np.random.randint(2)
        y[i] = float(cls)
        t0, t1 = t_first, t_first + gap
        if cls == 1:                          # A first
            I[i, t0:t0+dur, pos_a] = amp
            I[i, t1:t1+dur, pos_b] = amp
        else:                                 # B first
            I[i, t0:t0+dur, pos_b] = amp
            I[i, t1:t1+dur, pos_a] = amp

    return I, y


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Training                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(model, epochs=500, batch=32, lr=4e-3, device=DEVICE):
    I_all, y_all = make_xor_dataset(N=model.N, device=device)

    n_tr    = 160
    I_tr, y_tr = I_all[:n_tr], y_all[:n_tr]
    I_te, y_te = I_all[n_tr:], y_all[n_tr:]

    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    history = {"loss": [], "acc": [], "ais_c": []}
    best    = 0.0

    for ep in range(epochs):
        # Temperature annealing: smooth early → sharp late
        model.spike_temp = 8.0 * float(np.exp(-ep * 4.0 / epochs)) + 0.8

        model.train()
        perm  = torch.randperm(n_tr)
        ep_loss = 0.0
        n_b     = 0

        for s in range(0, n_tr, batch):
            idx   = perm[s:s+batch]
            logit = model(I_tr[idx])
            loss  = F.binary_cross_entropy_with_logits(logit, y_tr[idx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            opt.step()

            # keep parameters in physical range
            with torch.no_grad():
                model.gamma_alpha.clamp_(0.0, 1.0)
                model.ais_width.clamp_(2.0, 12.0)
                model.log_gl.clamp_(max=-1.0)  # gl ≤ 1  →  τ_m ≥ 1 ms

            ep_loss += loss.item()
            n_b     += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            logit_te = model(I_te)
            acc = ((logit_te > 0) == y_te).float().mean().item()

        ais_c = torch.sigmoid(model.ais_pos).item() * (model.N - 1)
        history["loss"].append(ep_loss / n_b)
        history["acc"].append(acc)
        history["ais_c"].append(ais_c)

        if acc > best:
            best = acc

        if ep % 50 == 0 or (ep > 200 and acc > 0.90):
            print(f"ep {ep:04d}  loss {ep_loss/n_b:.4f}  acc {acc*100:.1f}%  "
                  f"best {best*100:.1f}%  AIS@{ais_c:.1f}  T={model.spike_temp:.2f}")

    print(f"\n{'='*55}")
    print(f"Final acc {history['acc'][-1]*100:.1f}%   Best acc {best*100:.1f}%")
    print(f"{'='*55}")
    return history


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Visualisation                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def visualize(model, history, out="cable_neuron.png", device=DEVICE):
    T = 200
    I_vis, y_vis = make_xor_dataset(N=model.N, T=T, n=8, device=device)

    model.eval()
    with torch.no_grad():
        logits, V_tr, g_nav, g_kv, gated = model(I_vis, return_traces=True)

    dark = "#080810"
    fig  = plt.figure(figsize=(18, 11), facecolor=dark)
    gs   = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.32,
                    left=0.05, right=0.97, top=0.93, bottom=0.05)

    t_ms = np.arange(T) * model.dt
    xc   = np.arange(model.N)

    def dark_ax(ax):
        ax.set_facecolor(dark)
        ax.tick_params(colors="#555")
        for sp in ax.spines.values():
            sp.set_color("#1a1a2e")
        return ax

    # ── (0,0) AIS grating profile ───────────────────────────────────────────
    ax = dark_ax(fig.add_subplot(gs[0, 0]))
    ax.bar(xc, g_nav.cpu(), color="#28c8e0", alpha=0.8, label="Nav  (excitatory)", width=0.8)
    ax.bar(xc, -g_kv.cpu(), color="#e06060", alpha=0.8, label="Kv   (inhibitory)", width=0.8)
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_title("AIS Grating — Nav / Kv", color="#28c8e0", fontsize=9, pad=4)
    ax.legend(fontsize=7, facecolor=dark, labelcolor="white", framealpha=0)
    ax.set_xlabel("Compartment", color="#555", fontsize=7)

    # ── (0,1) Net grating ───────────────────────────────────────────────────
    ax = dark_ax(fig.add_subplot(gs[0, 1]))
    net = (g_nav - g_kv).cpu().numpy()
    ax.plot(xc, net, color="#00e87a", lw=1.5)
    ax.fill_between(xc, 0, net, where=net > 0, color="#00e87a", alpha=0.25)
    ax.fill_between(xc, 0, net, where=net < 0, color="#e06060", alpha=0.25)
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_title("Net grating  (Nav − Kv)", color="#00e87a", fontsize=9, pad=4)
    ax.set_xlabel("Compartment", color="#555", fontsize=7)

    # ── (0,2-3) Training curves ──────────────────────────────────────────────
    ax = dark_ax(fig.add_subplot(gs[0, 2:]))
    ep = np.arange(len(history["acc"]))
    ax.plot(ep, [a * 100 for a in history["acc"]], color="#9060c8", lw=1.5, label="Test acc %")
    ax.plot(ep, history["loss"], color="#e08030", lw=1.2, alpha=0.7, label="Train loss")
    ax.axhline(100, color="#00e87a", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Training history", color="white", fontsize=9, pad=4)
    ax.legend(fontsize=7, facecolor=dark, labelcolor="white", framealpha=0)
    ax.set_xlabel("Epoch", color="#555", fontsize=7)

    # ── Rows 1–2: spatiotemporal voltage for one example of each class ────────
    for row, (label_want, tag) in enumerate([(1.0, "Class 1 — A→B"), (0.0, "Class 0 — B→A")]):
        mask = (y_vis == label_want).nonzero(as_tuple=True)[0]
        if len(mask) == 0:
            continue
        si = mask[0].item()

        # Spatiotemporal heatmap
        ax = dark_ax(fig.add_subplot(gs[1 + row, :2]))
        Vm = V_tr[si].cpu().numpy().T   # (N, T)
        im = ax.imshow(Vm, aspect="auto", origin="lower",
                       cmap="magma", vmin=-15, vmax=45,
                       extent=[0, T * model.dt, 0, model.N])
        ax.set_title(f"Spatiotemporal V  —  {tag}", color="white", fontsize=9, pad=4)
        ax.set_xlabel("Time (ms)", color="#555", fontsize=7)
        ax.set_ylabel("Compartment", color="#555", fontsize=7)
        cb = plt.colorbar(im, ax=ax)
        cb.ax.tick_params(colors="#555", labelsize=7)
        cb.set_label("mV", color="#555", fontsize=7)

        # Soma + gated trace
        ax2 = dark_ax(fig.add_subplot(gs[1 + row, 2:]))
        ax2.plot(t_ms, V_tr[si, :, 0].cpu(),  color="#9060c8", lw=1.5, label="Soma V")
        ax2.plot(t_ms, gated[si].cpu(),        color="#28c8e0", lw=1.2, alpha=0.8,
                 label="Gated soma")
        thr_val = float(torch.exp(model.log_thr).item())
        ax2.axhline(thr_val, color="red", ls="--", lw=1.0, label=f"Thr {thr_val:.1f} mV")
        ax2.set_title(f"Soma trace  —  {tag}", color="white", fontsize=9, pad=4)
        ax2.set_xlabel("Time (ms)", color="#555", fontsize=7)
        ax2.set_ylabel("V (mV)",    color="#555", fontsize=7)
        ax2.legend(fontsize=7, facecolor=dark, labelcolor="white", framealpha=0)

    pred_cls = (logits.cpu() > 0).int().numpy()
    true_cls = y_vis.cpu().int().numpy()
    acc_vis  = (pred_cls == true_cls).mean() * 100
    fig.suptitle(
        f"BiologicalCableNeuron — Temporal XOR via Physical Wave Interference\n"
        f"Accuracy on 8 visualisation samples: {acc_vis:.0f}%",
        color="#00e87a", fontsize=11, y=0.97)

    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=dark)
    print(f"Figure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Entry point                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 55)
    print("BiologicalCableNeuron — semi-implicit PDE + AIS grating")
    print("=" * 55)
    print(f"Device : {DEVICE}")

    model = CableNeuron(N=32, dt=0.5, dx=2.0).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params}  (cable geometry + AIS + gamma + threshold)")
    print(f"Trial  : 200 steps × 0.5 ms = 100 ms")
    print(f"τ_m    ≈ {1.0 / float(torch.exp(model.log_gl).mean()):.0f} ms  "
          f"(inter-pulse gap = 50 ms)")
    print()
    print("Temporal XOR requires physical wave interference.")
    print("A linear Fourier synthesizer cannot solve this task.")
    print("The cable nonlinearity is the computation.")
    print()

    history = train(model, epochs=380, batch=32, lr=4e-3, device=DEVICE)

    visualize(model, history, out="cable_neuron.png")
