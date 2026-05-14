import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns

print("Loading Synthetic EEG Data...")
try:
    data = np.load("synthetic_cortex_eeg.npz")
    cemi = data['cemi']     # Shape: (Time)
    spikes = data['spikes'] # Shape: (Time, Neurons)
    ais = data['ais']       # Shape: (Time, Neurons)
    res = data['res']       # Shape: (Time, Neurons)
except FileNotFoundError:
    print("Error: Run gait_eeg_cortex.py and click 'RECORD EEG' first to generate the data.")
    exit()

time_steps = len(cemi)
neurons = spikes.shape[1]
fps = 30.0 # Approximate framerate of the simulation

# Create the Clinical Dashboard
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"Topological Cortex Clinical EEG Analysis ({time_steps} Frames | {neurons} Neurons)", fontsize=16)

# --- 1. MACRO EEG (Scalp Electrode / Global CEMI) ---
ax1 = plt.subplot(2, 2, 1)
time_axis = np.arange(time_steps) / fps
ax1.plot(time_axis, cemi, color='darkorange', linewidth=1.5)
ax1.set_title("Global CEMI Field (Macro Scalp EEG)")
ax1.set_xlabel("Time (Seconds)")
ax1.set_ylabel("CEMI Dipole Strength")
ax1.grid(True, alpha=0.3)

# --- 2. MULTI-UNIT ACTIVITY (Spike Raster Plot) ---
ax2 = plt.subplot(2, 2, 2)
# Find where spikes occurred (values > 0)
time_idx, neuron_idx = np.where(spikes > 0.05)
ax2.scatter(time_idx / fps, neuron_idx, s=1, color='teal', alpha=0.5)
ax2.set_title("Multi-Unit Activity (MUA / Raster Plot)")
ax2.set_xlabel("Time (Seconds)")
ax2.set_ylabel("Neuron ID")
ax2.set_ylim(0, neurons)

# --- 3. POWER SPECTRAL DENSITY (Brainwave Frequencies) ---
ax3 = plt.subplot(2, 2, 3)
# Calculate PSD using Welch's method
freqs, psd = signal.welch(cemi, fs=fps, nperseg=min(256, time_steps))
ax3.semilogy(freqs, psd, color='darkred', linewidth=2)
ax3.set_title("Power Spectral Density (Brainwave Frequencies)")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Power / Hz")
ax3.grid(True, alpha=0.3)
ax3.axvspan(0.5, 4, color='yellow', alpha=0.1, label='Delta (0.5-4 Hz)')
ax3.axvspan(4, 8, color='green', alpha=0.1, label='Theta (4-8 Hz)')
ax3.axvspan(8, 12, color='blue', alpha=0.1, label='Alpha (8-12 Hz)')
ax3.legend(loc='upper right')

# --- 4. FUNCTIONAL CONNECTIVITY (Correlation Matrix) ---
ax4 = plt.subplot(2, 2, 4)
# Calculate Pearson correlation between the AIS integration pools of the first 30 neurons
corr_matrix = np.corrcoef(ais[:, :30].T)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax4, cbar=False)
ax4.set_title("Functional Connectivity (AIS Cross-Correlation, First 30 Neurons)")
ax4.set_xlabel("Neuron ID")
ax4.set_ylabel("Neuron ID")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()