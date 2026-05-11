import os
import threading
import numpy as np
import mne
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import the core physics engine from your existing file
from universe_inverse import UniverseInverter

class GAIT_EEG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAIT — Clinical EEG Topological Explorer")
        self.root.geometry("1400x850")
        self.root.configure(bg="#07070f")
        
        self.raw_eeg = None
        self.channels = []
        self.sfreq = 1.0
        
        self.setup_ui()

    def setup_ui(self):
        # --- LEFT CONTROL PANEL ---
        ctrl_frame = tk.Frame(self.root, width=300, bg="#1a1a2e", padx=20, pady=20)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(ctrl_frame, text="GAIT EEG EXPLORER", fg="#28c8e0", bg="#1a1a2e", 
                 font=("Courier", 16, "bold")).pack(pady=(0, 20))
        
        # Load Button
        tk.Button(ctrl_frame, text="1. Load .EDF File", command=self.load_edf, 
                  bg="#28c8e0", fg="black", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=10)
        
        self.lbl_file = tk.Label(ctrl_frame, text="No file loaded", fg="#888", bg="#1a1a2e")
        self.lbl_file.pack(pady=5)

        # Channel Selector
        tk.Label(ctrl_frame, text="2. Select Electrode:", fg="white", bg="#1a1a2e").pack(anchor=tk.W, pady=(20, 5))
        self.combo_ch = ttk.Combobox(ctrl_frame, state="readonly")
        self.combo_ch.pack(fill=tk.X)

        # Time Window
        tk.Label(ctrl_frame, text="3. Start Time (seconds):", fg="white", bg="#1a1a2e").pack(anchor=tk.W, pady=(20, 5))
        self.ent_start = tk.Entry(ctrl_frame)
        self.ent_start.insert(0, "0")
        self.ent_start.pack(fill=tk.X)

        tk.Label(ctrl_frame, text="Window Length (seconds):", fg="white", bg="#1a1a2e").pack(anchor=tk.W, pady=(10, 5))
        self.ent_duration = tk.Entry(ctrl_frame)
        self.ent_duration.insert(0, "4")  # 4 seconds is usually a good manifold size
        self.ent_duration.pack(fill=tk.X)
        
        tk.Label(ctrl_frame, text="(Max ~2000 samples for UI speed)", fg="#888", bg="#1a1a2e", font=("Arial", 8)).pack(anchor=tk.W)

        # Analyze Button
        self.btn_analyze = tk.Button(ctrl_frame, text="4. Extract Geometry", command=self.run_analysis, 
                                     bg="#e06090", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.btn_analyze.pack(fill=tk.X, pady=30)
        
        # Results Readout
        self.lbl_results = tk.Label(ctrl_frame, text="", fg="#60e090", bg="#1a1a2e", font=("Courier", 11), justify=tk.LEFT)
        self.lbl_results.pack(fill=tk.X, pady=20)

        # --- RIGHT PLOT PANEL ---
        self.fig = plt.Figure(figsize=(10, 8), facecolor="#07070f")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def load_edf(self):
        filepath = filedialog.askopenfilename(filetypes=[("EDF Files", "*.edf"), ("All Files", "*.*")])
        if not filepath:
            return
            
        try:
            self.root.config(cursor="watch")
            self.root.update()
            
            # Load the EDF using MNE
            self.raw_eeg = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # Apply standard biological bandpass (1Hz to 45Hz) to remove drift and line noise
            self.raw_eeg.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            
            self.sfreq = self.raw_eeg.info['sfreq']
            self.channels = self.raw_eeg.ch_names
            
            self.lbl_file.config(text=os.path.basename(filepath) + f"\n({self.sfreq} Hz)")
            self.combo_ch['values'] = self.channels
            if self.channels:
                self.combo_ch.current(0)
            
            self.btn_analyze.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("EDF Load Error", str(e))
        finally:
            self.root.config(cursor="")

    def run_analysis(self):
        ch_name = self.combo_ch.get()
        try:
            start_t = float(self.ent_start.get())
            duration = float(self.ent_duration.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for time.")
            return

        # Extract the 1D signal array
        start_samp = int(start_t * self.sfreq)
        end_samp = int((start_t + duration) * self.sfreq)
        
        if end_samp > self.raw_eeg.n_times:
            messagebox.showwarning("Warning", "Window exceeds file length.")
            return
            
        signal, _ = self.raw_eeg[ch_name, start_samp:end_samp]
        signal_1d = signal[0] # Take the first (and only) channel

        # Disable UI and run physics on thread to keep UI responsive
        self.btn_analyze.config(text="Processing...", state=tk.DISABLED)
        threading.Thread(target=self._process_and_plot, args=(signal_1d, ch_name), daemon=True).start()

    def _process_and_plot(self, signal, ch_name):
        try:
            # 1. RUN GAIT PIPELINE
            # We use tau = sfreq/10 (roughly 100ms delay, standard for alpha rhythms)
            delay = int(self.sfreq / 10) 
            if delay < 1: delay = 1
            
            inv = UniverseInverter(
                signal=signal, 
                dim=6, 
                tau=delay, 
                epsilon=0.4, 
                gamma_hz=40.0, 
                dt=1.0/self.sfreq, 
                ais_window=int(self.sfreq/10) # 100ms integration window
            )
            inv.run()

            # 2. UPDATE UI SAFELY
            self.root.after(0, self._render_plots, inv, signal, ch_name)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Analysis Error", str(e))
            self.root.after(0, self.btn_analyze.config, {"text": "4. Extract Geometry", "state": tk.NORMAL})

    def _render_plots(self, inv, signal, ch_name):
        self.fig.clf()
        
        # Color palette
        c_sig = "#28c8e0"
        c_att = "#e06090"
        dark = "#07070f"
        
        # Create 3x1 layout
        ax1 = self.fig.add_subplot(311)
        ax2 = self.fig.add_subplot(323)
        ax3 = self.fig.add_subplot(324)
        ax4 = self.fig.add_subplot(313)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(dark)
            ax.tick_params(colors="#555")
            for sp in ax.spines.values():
                sp.set_color("#222")

        # Top: Raw EEG Signal
        time_axis = np.arange(len(signal)) / self.sfreq
        ax1.plot(time_axis, signal, color=c_sig, lw=1)
        ax1.set_title(f"Electrode: {ch_name} (Raw 1D Signal)", color=c_sig)
        ax1.set_ylabel("Amplitude")

        # Bottom Left: Recovered Attractor (Top 2 Eigenmodes)
        if inv.manifold2d_full is not None:
            M = inv.manifold2d_full
            ax2.scatter(M[:, 0], M[:, 1], c=np.linspace(0, 1, len(M)), cmap="plasma", s=2, alpha=0.6)
            ax2.set_title(f"Hidden Geometry (d_c ≈ {inv.frac_dim:.2f})", color=c_att)

        # Bottom Right: Recurrence Matrix (Somatic Memory)
        if inv.Vn is not None:
            ns = min(400, len(inv.Vn)) # Subsample for plot speed
            idx = np.linspace(0, len(inv.Vn)-1, ns, dtype=int)
            Vs = inv.Vn[idx]
            K = np.exp((Vs @ Vs.T - 1.0) / inv.epsilon**2)
            ax3.imshow(K, cmap="inferno", origin="lower", aspect="auto")
            ax3.set_title("Recurrence Field", color="#e0a030")

        # Bottom: Resonance and AIS Spikes
        if inv.resonance is not None:
            ax4.fill_between(time_axis[:len(inv.resonance)], 0, inv.resonance, color="#60e090", alpha=0.3)
            ax4.plot(time_axis[:len(inv.resonance)], inv.resonance, color="#60e090", lw=1)
            
            # Plot Spikes
            spike_times = time_axis[:len(inv.spikes)][inv.spikes > 0]
            for st in spike_times:
                ax4.axvline(st, color="#ff4040", lw=1, alpha=0.8)
            ax4.set_title(f"AIS Phase-Gated Projection ({int(inv.spikes.sum())} Spikes)", color="#60e090")

        self.fig.tight_layout()
        self.canvas.draw()

        # Update Readout text
        stats = (
            f"--- TOPOLOGICAL FINGERPRINT ---\n\n"
            f"Fractal Dim (d_c): {inv.frac_dim:.3f}\n"
            f"Total Spikes     : {int(inv.spikes.sum())}\n"
            f"Dom Eigenvalue   : {inv.eigenvalues[0]:.2f}\n"
        )
        if inv.frac_dim > 2.5:
            stats += "\nState: Chaotic / Desynchronized"
        elif inv.frac_dim < 1.5:
            stats += "\nState: Limit Cycle / Seizure-like"
        else:
            stats += "\nState: Healthy Complex / Pink"
            
        self.lbl_results.config(text=stats)
        
        # Reset Button
        self.btn_analyze.config(text="4. Extract Geometry", state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = GAIT_EEG_GUI(root)
    root.mainloop()