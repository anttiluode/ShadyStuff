import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting EEG-Instrumented Cortex on {device}...")

# =====================================================================
# 1. SPATIAL CORTEX WITH LESIONS
# =====================================================================
class LesionableCortex(nn.Module):
    def __init__(self, in_channels=2, num_neurons=128, taus=[1, 2, 3, 5, 8], eta=0.15, sparsity=0.05):
        super().__init__()
        self.num_neurons = num_neurons
        self.taus = taus
        self.eta = eta
        self.sparsity = sparsity
        self.base_thresh = 0.05
        
        self.patch_dim = in_channels * 9 
        m_init = torch.randn(num_neurons, len(taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)
        self.ais = None 
        
        self.register_buffer("lesion_mask", torch.ones(1, num_neurons, 1, 1))
        self.learning_active = True

    def step(self, image_tensor, current_cemi):
        B, C, H, W = image_tensor.shape
        if self.ais is None or self.ais.shape[-2:] != (H, W):
            self.ais = torch.zeros(1, self.num_neurons, H, W, device=device)
            
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        
        for i, tau in enumerate(self.taus):
            pad = tau
            padded = F.pad(image_tensor, (pad, pad, pad, pad), mode='reflect')
            unfolded = F.unfold(padded, kernel_size=3, dilation=tau)
            v = unfolded.view(B, self.patch_dim, H, W)
            v = F.normalize(v, p=2, dim=1)
            
            m_tau = self.m[:, i, :] 
            dot = torch.einsum('b c h w, n c -> b n h w', v, m_tau)
            total_res += dot ** 2
            
        avg_res = total_res / len(self.taus)
        avg_res = avg_res * self.lesion_mask
        
        self.ais = self.ais * 0.70 + avg_res * 0.30
        thresh = self.base_thresh * (1.0 - current_cemi * 0.6)
        
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_indices = torch.topk(self.ais, k, dim=1)
        sparse_mask = torch.zeros_like(self.ais, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_indices, True)
        
        spikes = ((self.ais > thresh) & sparse_mask).float()
        self.ais = self.ais * (1.0 - spikes)
        
        if self.learning_active and spikes.sum() > 0:
            spikes_flat = spikes.view(self.num_neurons, -1)
            active_neurons = spikes_flat.sum(dim=1) > 0
            if active_neurons.any():
                for i, tau in enumerate(self.taus):
                    pad = tau
                    padded = F.pad(image_tensor, (pad, pad, pad, pad), mode='reflect')
                    unfolded = F.unfold(padded, kernel_size=3, dilation=tau).view(B, self.patch_dim, H, W)
                    v_norm = F.normalize(unfolded, p=2, dim=1)[0]
                    for n in torch.where(active_neurons)[0]:
                        if self.lesion_mask[0, n, 0, 0] == 0: continue
                        mask = spikes[0, n:n+1, :, :]
                        if mask.sum() == 0: continue
                        v_masked = (v_norm * mask).sum(dim=(1,2)) / mask.sum()
                        m_curr = self.m[n, i]
                        self.m.data[n, i] += self.eta * (v_masked - torch.dot(v_masked, m_curr) * m_curr)
                self.m.data = F.normalize(self.m.data, p=2, dim=2)
        
        return avg_res.mean(dim=1, keepdim=True), spikes

    def reconstruct_mind_eye(self, spikes):
        B, N, H, W = spikes.shape
        reconstruction = torch.zeros(B, 2, H, W, device=device)
        spikes_flat = spikes.view(B, N, H*W) 
        for i, tau in enumerate(self.taus):
            m_tau = self.m[:, i, :]
            patches = torch.bmm(spikes_flat.transpose(1, 2), m_tau.unsqueeze(0).expand(B, -1, -1))
            patches = patches.transpose(1, 2)
            pad = tau
            folded = F.fold(patches, output_size=(H + pad*2, W + pad*2), kernel_size=3, dilation=tau)
            reconstruction += folded[:, :, pad:-pad, pad:-pad]
        recon_min = reconstruction.amin(dim=(2,3), keepdim=True)
        recon_max = reconstruction.amax(dim=(2,3), keepdim=True)
        reconstruction = (reconstruction - recon_min) / (recon_max - recon_min + 1e-8)
        return reconstruction

# =====================================================================
# 2. FREQUENCY DECOMPOSITION
# =====================================================================
def extract_frequency_streams(frame_gray):
    low_freq = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    high_freq = cv2.subtract(frame_gray, low_freq)
    high_freq = cv2.equalizeHist(high_freq)
    tensor_lf = torch.from_numpy(low_freq).float() / 255.0
    tensor_hf = torch.from_numpy(high_freq).float() / 255.0
    stacked = torch.stack((tensor_lf, tensor_hf), dim=0).unsqueeze(0)
    return stacked.to(device)

# =====================================================================
# 3. EEG LAB GUI
# =====================================================================
class EEG_LabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Topological EEG & Lesion Lab")
        self.root.configure(bg="#111115")
        
        self.RES = 128
        self.N_NEURONS = 128
        self.cortex = LesionableCortex(in_channels=2, num_neurons=self.N_NEURONS).to(device)
        self.cemi = 0.0
        
        # --- EEG Recording State ---
        self.is_recording = False
        self.record_frames = 0
        self.MAX_FRAMES = 1200 # ~10 seconds at 30fps
        self.eeg_data = {'cemi': [], 'spikes': [], 'ais': [], 'res': []}
        
        self.cap = cv2.VideoCapture(0)
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        ctrl_frame = tk.Frame(self.root, bg="#222", pady=10)
        ctrl_frame.pack(fill=tk.X)
        
        self.btn_freeze = tk.Button(ctrl_frame, text="🧠 FREEZE LEARNING", bg="#cc5555", fg="white", font=("Courier", 10, "bold"), command=self.toggle_learning)
        self.btn_freeze.pack(side=tk.LEFT, padx=10)
        
        self.btn_record = tk.Button(ctrl_frame, text="⏺ RECORD EEG", bg="#5555cc", fg="white", font=("Courier", 10, "bold"), command=self.start_recording)
        self.btn_record.pack(side=tk.LEFT, padx=10)
        
        self.status_lbl = tk.Label(ctrl_frame, text="Status: LEARNING ACTIVE", bg="#222", fg="#55cc55", font=("Courier", 12, "bold"))
        self.status_lbl.pack(side=tk.RIGHT, padx=20)
        
        self.display_lbl = tk.Label(self.root, bg="#111")
        self.display_lbl.pack(fill=tk.BOTH, expand=True, pady=5)
        
        grid_frame = tk.Frame(self.root, bg="#111", pady=10)
        grid_frame.pack(fill=tk.X)
        tk.Label(grid_frame, text="NEURON TISSUE (Click to Lesion):", bg="#111", fg="white", font=("Courier", 10)).pack()
        self.tissue_canvas = tk.Canvas(grid_frame, width=1024, height=40, bg="#222", highlightthickness=0)
        self.tissue_canvas.pack(pady=5)
        self.tissue_canvas.bind("<Button-1>", self.on_tissue_click)
        self.draw_tissue_bar()

    def toggle_learning(self):
        self.cortex.learning_active = not self.cortex.learning_active
        self.btn_freeze.config(text="🧠 FREEZE LEARNING" if self.cortex.learning_active else "🔓 UNFREEZE", bg="#cc5555" if self.cortex.learning_active else "#55cc55")
        self.status_lbl.config(text="Status: LEARNING ACTIVE" if self.cortex.learning_active else "Status: FROZEN", fg="#55cc55" if self.cortex.learning_active else "#cc5555")

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_frames = 0
            self.eeg_data = {'cemi': [], 'spikes': [], 'ais': [], 'res': []}
            self.btn_record.config(text="RECORDING... DO NOT CLOSE", bg="#cc0000")
            self.status_lbl.config(text="Status: GATHERING EEG DATA...", fg="#cccc55")

    def save_recording(self):
        self.is_recording = False
        self.btn_record.config(text="⏺ RECORD EEG (10s)", bg="#5555cc")
        # Save to compressed NumPy dictionary
        np.savez("synthetic_cortex_eeg.npz", 
                 cemi=np.array(self.eeg_data['cemi']),
                 spikes=np.array(self.eeg_data['spikes']),
                 ais=np.array(self.eeg_data['ais']),
                 res=np.array(self.eeg_data['res']))
        self.status_lbl.config(text="Status: SAVED to synthetic_cortex_eeg.npz", fg="#55cc55")

    def draw_tissue_bar(self):
        self.tissue_canvas.delete("all")
        mask = self.cortex.lesion_mask.squeeze().cpu().numpy()
        block_w = 1024 / self.N_NEURONS
        for i in range(self.N_NEURONS):
            color = "#00ff88" if mask[i] > 0.5 else "#333333"
            self.tissue_canvas.create_rectangle(i*block_w, 0, (i+1)*block_w, 40, fill=color, outline="#111")

    def on_tissue_click(self, event):
        block_w = 1024 / self.N_NEURONS
        n_idx = int(event.x / block_w)
        if 0 <= n_idx < self.N_NEURONS:
            current = self.cortex.lesion_mask[0, n_idx, 0, 0].item()
            new_state = 0.0 if current > 0.5 else 1.0
            for offset in range(-4, 5):
                t = n_idx + offset
                if 0 <= t < self.N_NEURONS: self.cortex.lesion_mask[0, t, 0, 0] = new_state
            self.draw_tissue_bar()

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.RES, self.RES))
            x_t = extract_frequency_streams(gray)
            
            heatmap, spikes = self.cortex.step(x_t, self.cemi)
            
            # --- DATA LOGGING (THE EEG) ---
            if self.is_recording:
                # We save spatial means for AIS and Res to keep files sizes manageable (acting like single macro-electrodes per neuron)
                self.eeg_data['cemi'].append(self.cemi)
                self.eeg_data['spikes'].append(spikes.mean(dim=(2,3)).squeeze().cpu().numpy().copy())
                self.eeg_data['ais'].append(self.cortex.ais.mean(dim=(2,3)).squeeze().cpu().numpy().copy())
                self.eeg_data['res'].append(heatmap.mean(dim=(2,3)).squeeze().cpu().numpy().copy())
                
                self.record_frames += 1
                if self.record_frames >= self.MAX_FRAMES:
                    self.save_recording()
            
            spike_rate = spikes.mean().item() / self.cortex.sparsity
            self.cemi = self.cemi * 0.8 + spike_rate * 0.2
            
            mind_eye = self.cortex.reconstruct_mind_eye(spikes)
            
            real_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            low_f, high_f = x_t[0, 0].cpu().numpy(), x_t[0, 1].cpu().numpy()
            stream_vis = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            stream_vis[:,:,0] = high_f * 2.0 
            stream_vis[:,:,1] = (high_f + low_f) * 0.5 
            stream_vis[:,:,2] = low_f * 1.5  
            stream_vis = (np.clip(stream_vis, 0, 1) * 255).astype(np.uint8)
            
            recon_low, recon_high = mind_eye[0, 0].cpu().numpy(), mind_eye[0, 1].cpu().numpy()
            mind_vis = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            mind_vis[:,:,0] = recon_high * 2.0
            mind_vis[:,:,1] = (recon_high + recon_low) * 0.5
            mind_vis[:,:,2] = recon_low * 1.5
            mind_vis = (np.clip(mind_vis, 0, 1) * 255).astype(np.uint8)
            
            display = np.hstack((real_bgr, stream_vis, mind_vis))
            display = cv2.resize(display, (1200, 400), interpolation=cv2.INTER_NEAREST)
            
            # Flash red border if recording
            if self.is_recording: cv2.rectangle(display, (0,0), (1199, 399), (0,0,255), 6)
            
            cv2.putText(display, "1. REALITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "2. FREQUENCY STREAMS", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "3. MIND'S EYE RECONSTRUCTION", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            self.display_lbl.imgtk = img
            self.display_lbl.configure(image=img)
            
        self.root.after(20, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEG_LabApp(root)
    root.mainloop()
