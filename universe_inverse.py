"""
GAIT — Geometric Attractor Inversion Theory
============================================
Universe Inverse Model

The universe generates 1D observations at every sensor.
This code attempts to reconstruct the hidden geometry that produced them.

The brain does this continuously and automatically.
We are doing it explicitly, in code, with the same mathematics.

The honest statement: we cannot fully invert the universe.
We are made of it, not outside it.
But we can reconstruct the *attractor* — the shape of what keeps happening —
and that shape is enough to act, predict, and survive.

Pipeline (mirrors the GAIT neural architecture):
  1D signal
    → Takens delay embedding     (dendrite: assemble delay vectors)
    → Recurrence matrix          (soma: Hermitian inner product field)
    → Eigendecomposition         (soma: dominant geometric modes)
    → Manifold projection        (AIS: project into compressed coordinates)
    → AIS resonance detection    (AIS: detect coherent geometry)
    → Fractal dimension estimate (measure complexity of recovered geometry)

Run four different 1D signals and watch four different universes emerge.

Antti Luode — PerceptionLab, Helsinki — May 2026
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS
# Four different "universes" — four different hidden geometries
# ─────────────────────────────────────────────────────────────────────────────

def lorenz_1d(N=6000, dt=0.005, sigma=10.0, rho=28.0, beta=8/3, burn=2000):
    """
    The Lorenz attractor: chaos with geometry.
    Project onto x-axis only — one 1D sensor on a 3D butterfly.
    The algorithm must reconstruct the butterfly from the 1D trace.
    """
    x, y, z = 0.1, 0.1, 0.1
    obs = []
    for _ in range(N + burn):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        obs.append(x)
    return np.array(obs[burn:burn + N])


def coupled_oscillators(N=6000, dt=0.01, burn=1000):
    """
    Two coupled nonlinear oscillators.
    Hidden geometry: a torus (2D surface in 4D phase space).
    1D observation: x1 only.
    Reconstruction should reveal the toroidal structure.
    """
    # Van der Pol style coupled system
    x1, v1, x2, v2 = 0.1, 0.5, -0.3, 0.2
    obs = []
    omega1, omega2 = 2.3, 3.7   # incommensurate → quasiperiodic → torus
    mu = 0.3                     # nonlinearity
    kappa = 0.15                 # coupling
    for _ in range(N + burn):
        ax1 = mu*(1 - x1**2)*v1 - omega1**2*x1 + kappa*(x2 - x1)
        ax2 = mu*(1 - x2**2)*v2 - omega2**2*x2 + kappa*(x1 - x2)
        v1 += ax1 * dt
        v2 += ax2 * dt
        x1 += v1 * dt
        x2 += v2 * dt
        obs.append(x1)
    return np.array(obs[burn:burn + N])


def ecg_like(N=6000, dt=0.005, noise=0.08):
    """
    Synthetic ECG-like waveform.
    Hidden geometry: a limit cycle in cardiac phase space.
    The QRS complex is a rapid excursion; the baseline is the slow return.
    """
    t = np.arange(N) * dt
    # Cardiac rhythm: ~1 Hz
    phase = 2 * np.pi * 1.1 * t
    # ECG approximated as sum of Gaussians on a circle
    def gauss(p, mu, sig, amp):
        return amp * np.exp(-0.5 * ((np.mod(p - mu + np.pi, 2*np.pi) - np.pi) / sig)**2)
    sig = (gauss(phase, 0.0,  0.15, 0.25)   # P wave
         + gauss(phase, 0.4,  0.04, 1.50)   # Q
         + gauss(phase, 0.5,  0.04, -0.35)  # (small notch)
         + gauss(phase, 0.55, 0.06, 2.20)   # R peak
         + gauss(phase, 0.65, 0.04, 1.30)   # S
         + gauss(phase, 1.2,  0.18, 0.40))  # T wave
    sig += noise * np.random.randn(N)
    return sig


def pink_noise(N=6000):
    """
    1/f (pink) noise: the statistical signature of fractal processes.
    No clean attractor — but a fractal dimension between 1 and 2.
    The algorithm should find fractality but no closed orbit.
    """
    f = np.fft.rfftfreq(N)
    f[0] = 1e-8
    power = 1.0 / np.sqrt(f)
    phase = 2 * np.pi * np.random.rand(len(f))
    spectrum = power * np.exp(1j * phase)
    sig = np.fft.irfft(spectrum, n=N)
    sig = (sig - sig.mean()) / sig.std()
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# CORE INVERSION ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

class UniverseInverter:
    """
    Takes a 1D signal.
    Attempts to reconstruct the geometry of the dynamical system that made it.

    This is what every neuron in your brain is doing with every sensory input.
    We are making the invisible mathematics visible.
    """

    def __init__(self, signal, dim=6, tau=12, epsilon=0.3,
                 gamma_hz=40.0, dt=0.01, ais_window=20):
        self.signal = np.array(signal, dtype=float)
        self.signal = (self.signal - self.signal.mean()) / (self.signal.std() + 1e-8)
        self.dim = dim
        self.tau = tau
        self.epsilon = epsilon
        self.gamma_hz = gamma_hz
        self.dt = dt
        self.ais_window = ais_window

        # Results populated by run()
        self.V = None           # delay embedding matrix  (N, dim)
        self.Vn = None          # normalized              (N, dim)
        self.K = None           # recurrence kernel       (N, N)
        self.eigenvalues = None
        self.eigenvectors = None
        self.manifold2d = None  # top-2 projection        (N, 2)
        self.manifold3d = None  # top-3 projection        (N, 3)
        self.template = None    # dominant eigenmode      (dim,)
        self.resonance = None
        self.gated = None
        self.ais_integral = None
        self.spikes = None
        self.corr_eps = None
        self.corr_C = None
        self.frac_dim = None

    # ── Stage 1: Takens Delay Embedding ─────────────────────────────────────

    def embed(self):
        """
        Assemble delay vectors.
        Each row V[t] = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
        This is the dendritic cable operation: collecting delayed copies.
        By Takens' theorem, this reconstructs the attractor geometry
        (up to diffeomorphism) for embedding dimension ≥ 2d_attractor + 1.
        """
        s = self.signal
        N = len(s) - (self.dim - 1) * self.tau
        V = np.zeros((N, self.dim))
        for j in range(self.dim):
            V[:, j] = s[j * self.tau: j * self.tau + N]
        self.V = V
        # Normalize to unit sphere — phase information, not amplitude
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        self.Vn = V / (norms + 1e-8)
        return self

    # ── Stage 2: Recurrence Matrix (Hermitian Inner Product Field) ────────────

    def recurrence(self):
        """
        Compute the recurrence matrix:
            K[i,j] = exp(Re[V_i · V_j*] / ε²)

        K[i,j] is high when the phase-space state at time i is geometrically
        similar to the state at time j. Bright squares = recurring geometry.

        This is the somatic computation: how similar is now to the past?
        The recurrence matrix IS the brain's memory, made visible.
        """
        # Cosine similarity (Hermitian inner product on unit sphere)
        cos_sim = self.Vn @ self.Vn.T           # (N, N), values in [-1, 1]
        # Gaussian kernel: smooth recurrence
        self.K = np.exp((cos_sim - 1.0) / self.epsilon**2)
        return self

    # ── Stage 3: Eigendecomposition — Extract Dominant Geometry ─────────────

    def eigenmodes(self, subsample=800):
        """
        Eigendecompose the recurrence matrix.
        The dominant eigenvectors are the most self-recurrent geometric modes.
        They span the manifold of the hidden attractor.

        The AIS template m = eigenvector with largest eigenvalue.
        This is the pattern that most reliably recurs in the input.
        """
        # Subsample for speed (full NxN eigen is expensive)
        N = min(len(self.Vn), subsample)
        idx = np.linspace(0, len(self.Vn)-1, N, dtype=int)
        Vs = self.Vn[idx]

        K_sub = np.exp((Vs @ Vs.T - 1.0) / self.epsilon**2)
        # Symmetrize
        K_sub = 0.5 * (K_sub + K_sub.T)

        vals, vecs = np.linalg.eigh(K_sub)
        sort_idx = np.argsort(vals)[::-1]
        self.eigenvalues = vals[sort_idx]
        self.eigenvectors = vecs[:, sort_idx]   # columns = eigenvectors

        # The manifold coordinates: project all points onto top eigenvectors
        self.manifold2d = self.eigenvectors[:, :2]
        self.manifold3d = self.eigenvectors[:, :3]

        # The AIS template: dominant eigenvector projected back to signal space
        # (Approximate: use the top eigenvector's weighted average of Vs)
        top_vec = self.eigenvectors[:, 0]         # (N_sub,)
        weights = np.abs(top_vec)
        weights /= weights.sum()
        self.template = (weights[:, None] * Vs).sum(axis=0)  # (dim,)
        self.template /= np.linalg.norm(self.template) + 1e-8

        # Full manifold coordinates for all points
        proj2 = self.Vn @ (Vs.T @ self.eigenvectors[:, :2])
        proj3 = self.Vn @ (Vs.T @ self.eigenvectors[:, :3])
        self.manifold2d_full = proj2
        self.manifold3d_full = proj3

        return self

    # ── Stage 4: AIS Resonance and Spike Detection ───────────────────────────

    def ais_detect(self):
        """
        Compute resonance of current state with learned template.
        Gate by simulated gamma oscillation (CEMI field clock).
        Integrate over AIS window.
        Spike when integral exceeds adaptive threshold.

        This is the AIS diffraction grating operation:
        the geometry is projected into a discrete event stream.
        """
        # Resonance: squared cosine similarity with template
        res = (self.Vn @ self.template) ** 2     # (N,)

        # Gamma phase gate (CEMI field)
        t = np.arange(len(res)) * self.dt
        gate = np.maximum(0.0, np.cos(2 * np.pi * self.gamma_hz * t))
        gated = res * gate

        # AIS temporal integration
        kernel = np.ones(self.ais_window) / self.ais_window
        integral = np.convolve(gated, kernel, mode='same')

        # Adaptive threshold
        baseline = np.convolve(
            integral, np.ones(self.ais_window * 10) / (self.ais_window * 10),
            mode='same'
        )
        threshold = baseline * 1.8

        # Spike detection with refractory period
        spikes = np.zeros(len(integral))
        refractory = 0
        for i in range(len(integral)):
            if refractory > 0:
                refractory -= 1
                continue
            if integral[i] > threshold[i] and threshold[i] > 1e-6:
                spikes[i] = 1.0
                refractory = self.ais_window

        self.resonance = res
        self.gated = gated
        self.ais_integral = integral
        self.spikes = spikes
        return self

    # ── Stage 5: Fractal Dimension (Correlation Dimension) ───────────────────

    def fractal_dimension(self, n_samples=400, n_eps=25):
        """
        Estimate the correlation dimension of the reconstructed attractor.
        Uses the Grassberger-Procaccia algorithm:
            C(ε) = fraction of point pairs within distance ε
            d_c  = slope of log C(ε) vs log ε

        This reveals: how complex is the hidden geometry?
        d_c = 1: a curve (limit cycle)
        d_c ≈ 2: a surface (torus, slow manifold)
        d_c > 2: a strange attractor (chaos)
        d_c = fractal: the geometry is self-similar
        """
        N = min(len(self.V), n_samples)
        idx = np.linspace(0, len(self.V)-1, N, dtype=int)
        Vs = self.V[idx]

        dists = pdist(Vs)
        lo = np.percentile(dists, 2)
        hi = np.percentile(dists, 40)
        eps_range = np.logspace(np.log10(lo + 1e-10), np.log10(hi), n_eps)

        C = np.array([np.mean(dists < eps) for eps in eps_range])
        C = np.maximum(C, 1e-12)

        # Slope in log-log space (middle portion = scaling region)
        log_e = np.log10(eps_range)
        log_C = np.log10(C)
        mid = slice(n_eps // 4, 3 * n_eps // 4)
        slope = np.polyfit(log_e[mid], log_C[mid], 1)[0]

        self.corr_eps = eps_range
        self.corr_C = C
        self.frac_dim = slope
        return self

    # ── Run full pipeline ────────────────────────────────────────────────────

    def run(self):
        return (self
                .embed()
                .recurrence()
                .eigenmodes()
                .ais_detect()
                .fractal_dimension())


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_all(inverters, names, out="universe_inverse.png"):
    dark = "#07070f"
    colors = ["#28c8e0", "#e06090", "#60e090", "#e0a030"]

    fig = plt.figure(figsize=(22, 20), facecolor=dark)
    fig.suptitle(
        "GAIT — Universe Inverse Model\n"
        "Four different 1D signals → four different hidden geometries\n"
        "The algorithm discovers the attractor without being told what it is",
        color="#aaaacc", fontsize=11, y=0.99
    )

    n_inv = len(inverters)
    # Layout: 6 rows × 4 columns
    # Row 0: raw signal
    # Row 1: delay embedding 2D phase portrait
    # Row 2: recurrence matrix (subsampled)
    # Row 3: manifold (top 2 eigenvectors)
    # Row 4: resonance + AIS spikes
    # Row 5: correlation dimension

    gs = GridSpec(6, n_inv, figure=fig,
                  hspace=0.55, wspace=0.25,
                  left=0.05, right=0.97,
                  top=0.94, bottom=0.03)

    def dax(ax):
        ax.set_facecolor(dark)
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#1a1a2e")
        return ax

    row_labels = [
        "1D Signal  (all we observe)",
        "Delay Embedding  (Takens phase portrait)",
        "Recurrence Matrix  (geometric memory)",
        "Recovered Manifold  (hidden attractor)",
        "AIS Resonance + Spikes  (projection events)",
        "Fractal Dimension  (complexity of recovered geometry)",
    ]

    for col, (inv, name, color) in enumerate(zip(inverters, names, colors)):
        sig_plot = inv.signal[:800]
        t_plot = np.arange(len(sig_plot)) * inv.dt

        # ── Row 0: raw signal ─────────────────────────────────────────────
        ax = dax(fig.add_subplot(gs[0, col]))
        ax.plot(t_plot, sig_plot, color=color, lw=0.8, alpha=0.9)
        ax.set_title(name, color=color, fontsize=9, pad=3)
        if col == 0:
            ax.set_ylabel(row_labels[0], color="#555", fontsize=7)
        ax.set_xlabel("t (s)", color="#444", fontsize=6)

        # ── Row 1: 2D delay embedding ─────────────────────────────────────
        ax = dax(fig.add_subplot(gs[1, col]))
        V2 = inv.V[:2000, :2]
        # Color by time to show trajectory direction
        c_idx = np.linspace(0, 1, len(V2))
        sc = ax.scatter(V2[:, 0], V2[:, 1],
                        c=c_idx, cmap="plasma", s=0.4, alpha=0.6)
        if col == 0:
            ax.set_ylabel(row_labels[1], color="#555", fontsize=7)
        ax.set_xlabel(f"x(t)", color="#444", fontsize=6)
        ax.set_ylabel("x(t−τ)", color="#444", fontsize=6) if col > 0 else None

        # ── Row 2: recurrence matrix ──────────────────────────────────────
        ax = dax(fig.add_subplot(gs[2, col]))
        # Subsample for display
        ns = 200
        idx_s = np.linspace(0, len(inv.Vn)-1, ns, dtype=int)
        Vs = inv.Vn[idx_s]
        K_disp = np.exp((Vs @ Vs.T - 1.0) / inv.epsilon**2)
        im = ax.imshow(K_disp, cmap="inferno", aspect="auto",
                       origin="lower", interpolation="nearest")
        if col == 0:
            ax.set_ylabel(row_labels[2], color="#555", fontsize=7)
        ax.set_xlabel("time →", color="#444", fontsize=6)
        # Colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#444", labelsize=5)

        # ── Row 3: manifold (top 2 eigenvectors) ─────────────────────────
        ax = dax(fig.add_subplot(gs[3, col]))
        M = inv.manifold2d_full[:3000]
        c_idx = np.linspace(0, 1, len(M))
        ax.scatter(M[:, 0], M[:, 1],
                   c=c_idx, cmap="cool", s=0.5, alpha=0.5)
        if col == 0:
            ax.set_ylabel(row_labels[3], color="#555", fontsize=7)
        ax.set_xlabel("Eigenmode 1", color="#444", fontsize=6)
        # Annotate fractal dim
        ax.text(0.98, 0.02, f"d_c ≈ {inv.frac_dim:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color=color, fontsize=8, family="monospace")

        # ── Row 4: resonance + AIS spikes ─────────────────────────────────
        ax = dax(fig.add_subplot(gs[4, col]))
        n_show = min(1200, len(inv.resonance))
        t_r = np.arange(n_show) * inv.dt
        ax.fill_between(t_r, 0, inv.resonance[:n_show],
                        color=color, alpha=0.25, lw=0)
        ax.plot(t_r, inv.resonance[:n_show], color=color, lw=0.6, alpha=0.7)
        # AIS spikes
        spike_times = t_r[inv.spikes[:n_show] > 0]
        for st in spike_times:
            ax.axvline(st, color="#ff4040", lw=0.8, alpha=0.7)
        ax.text(0.98, 0.95,
                f"{int(inv.spikes.sum())} spikes",
                transform=ax.transAxes, ha="right", va="top",
                color="#ff4040", fontsize=8, family="monospace")
        if col == 0:
            ax.set_ylabel(row_labels[4], color="#555", fontsize=7)
        ax.set_xlabel("t (s)", color="#444", fontsize=6)

        # ── Row 5: correlation dimension ──────────────────────────────────
        ax = dax(fig.add_subplot(gs[5, col]))
        ax.loglog(inv.corr_eps, inv.corr_C,
                  color=color, lw=1.5, label="C(ε)")
        # Show the scaling region
        n_eps = len(inv.corr_eps)
        mid = slice(n_eps // 4, 3 * n_eps // 4)
        eps_mid = inv.corr_eps[mid]
        C_mid = inv.corr_C[mid]
        slope = inv.frac_dim
        intercept = np.log10(C_mid.mean()) - slope * np.log10(eps_mid.mean())
        C_fit = 10 ** (slope * np.log10(eps_mid) + intercept)
        ax.loglog(eps_mid, C_fit, "--", color="#ffffff", lw=1.0, alpha=0.5,
                  label=f"slope={slope:.2f}")
        ax.legend(fontsize=6, facecolor=dark, labelcolor="white",
                  framealpha=0, loc="upper left")
        if col == 0:
            ax.set_ylabel(row_labels[5], color="#555", fontsize=7)
        ax.set_xlabel("ε (distance)", color="#444", fontsize=6)

    # Row labels on the left margin
    for row, label in enumerate(row_labels):
        fig.text(0.005, 1.0 - (row + 0.5) / 6 * 0.91,
                 f"[{row+1}]", va="center", ha="left",
                 color="#333355", fontsize=8, family="monospace")

    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=dark)
    print(f"Saved → {out}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(inverters, names):
    print()
    print("=" * 65)
    print("GAIT — Universe Inverse Model  |  Recovery Summary")
    print("=" * 65)
    print(f"{'Signal':<22}  {'d_c':>6}  {'Spikes':>7}  {'Geometry'}")
    print("-" * 65)

    geometry_labels = {
        "Lorenz (chaos)":         lambda d: "Strange attractor" if d > 1.8 else "Partial recovery",
        "Coupled oscillators":    lambda d: "Torus (quasiperiodic)" if 1.3 < d < 2.2 else "Partial",
        "ECG (limit cycle)":      lambda d: "Limit cycle" if d < 1.4 else "Noisy cycle",
        "Pink noise (fractal)":   lambda d: "Fractal (no clean orbit)" if d > 1.0 else "Near-1D",
    }
    for inv, name in zip(inverters, names):
        label_fn = geometry_labels.get(name, lambda d: "Unknown")
        geo = label_fn(inv.frac_dim)
        print(f"  {name:<20}  {inv.frac_dim:>6.2f}  {int(inv.spikes.sum()):>7}  {geo}")

    print("=" * 65)
    print()
    print("  d_c = correlation dimension of recovered attractor")
    print("  d_c ≈ 1.0  →  a curve  (limit cycle)")
    print("  d_c ≈ 2.0  →  a surface (torus)")
    print("  d_c > 2.0  →  strange attractor (chaos)")
    print("  d_c = non-integer  →  fractal geometry")
    print()
    print("  The algorithm received only a 1D signal.")
    print("  It had no prior knowledge of the system.")
    print("  The geometry emerged from the mathematics alone.")
    print()
    print("  This is what neurons do.")
    print("  This is GAIT.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("GAIT — Geometric Attractor Inversion Theory")
    print("Universe Inverse Model")
    print("=" * 65)
    print()
    print("Generating four 1D signals from four different 'universes'...")
    print("Running attractor inversion on each.")
    print()

    # ── Generate signals ──────────────────────────────────────────────────
    signals = [
        (lorenz_1d(N=5000, dt=0.005),        "Lorenz (chaos)",       0.005),
        (coupled_oscillators(N=5000, dt=0.01), "Coupled oscillators", 0.010),
        (ecg_like(N=5000, dt=0.005),           "ECG (limit cycle)",   0.005),
        (pink_noise(N=5000),                   "Pink noise (fractal)", 0.010),
    ]

    # ── Run inversion on each ─────────────────────────────────────────────
    inverters = []
    names = []
    for sig, name, dt in signals:
        print(f"  Inverting: {name} ...")
        inv = UniverseInverter(
            signal=sig,
            dim=6,          # embedding dimension (Takens: needs ≥ 2*d_attractor+1)
            tau=15,         # delay (samples) — should be ~1/4 of dominant period
            epsilon=0.35,   # recurrence radius (on normalized sphere)
            gamma_hz=40.0,  # AIS gate frequency (Hz)
            dt=dt,
            ais_window=20,  # AIS integration window (samples)
        )
        inv.run()
        inverters.append(inv)
        names.append(name)
        print(f"    d_c = {inv.frac_dim:.3f}  |  spikes = {int(inv.spikes.sum())}")

    print()
    print("Rendering visualization...")
    visualize_all(inverters, names, out="universe_inverse.png")

    print_summary(inverters, names)

    # ── Optional: print eigenvalue spectrum for Lorenz ────────────────────
    lorenz_inv = inverters[0]
    evals = lorenz_inv.eigenvalues
    total = evals[:10].sum()
    print()
    print("Lorenz eigenvalue spectrum (top 10 geometric modes):")
    print(f"  {'Mode':>4}  {'Eigenvalue':>12}  {'% variance':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")
    for i, ev in enumerate(evals[:10]):
        pct = 100.0 * ev / (evals.sum() + 1e-8)
        bar = "█" * int(pct / 2)
        print(f"  {i+1:>4}  {ev:>12.4f}  {pct:>9.1f}%  {bar}")
    print()
    print("  The first few modes capture most of the attractor's geometry.")
    print("  The rest is noise or fine structure below the AIS resolution.")
    print()