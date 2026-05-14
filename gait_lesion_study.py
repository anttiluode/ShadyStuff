import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting Topological Lesion Simulator on {device}...")

# =====================================================================
# 1. SPATIAL CORTEX WITH LESION CAPABILITY
# =====================================================================
class LesionableCortex(nn.Module):
    def __init__(self, in_channels=2, num_neurons=128, taus=[1, 2, 3, 5, 8], eta=0.15, sparsity=0.05):
        super().__init__()
        self.num_neurons = num_neurons
        self.taus = taus
        self.eta = eta
        self.sparsity = sparsity
        self.base_thresh = 0.05
        
        # We use 2 channels: Low Frequency (Blobs) and High Frequency (Edges)
        self.patch_dim = in_channels * 9 
        
        m_init = torch.randn(num_neurons, len(taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)
        self.ais = None 
        
        # LESION MASK: 1.0 = Healthy, 0.0 = Dead
        self.register_buffer("lesion_mask", torch.ones(1, num_neurons, 1, 1))
        self.learning_active = True

    def step(self, image_tensor, current_cemi):
        B, C, H, W = image_tensor.shape
        if self.ais is None or self.ais.shape[-2:] != (H, W):
            self.ais = torch.zeros(1, self.num_neurons, H, W, device=device)
            
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        
        # Forward pass (extract Takens delays and compute resonance)
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
        
        # Apply Lesion Mask (Dead neurons cannot resonate)
        avg_res = avg_res * self.lesion_mask
        
        self.ais = self.ais * 0.70 + avg_res * 0.30
        thresh = self.base_thresh * (1.0 - current_cemi * 0.6)
        
        # Lateral Inhibition
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_indices = torch.topk(self.ais, k, dim=1)
        sparse_mask = torch.zeros_like(self.ais, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_indices, True)
        
        spikes = ((self.ais > thresh) & sparse_mask).float()
        self.ais = self.ais * (1.0 - spikes)
        
        # Oja's Rule (Only if learning is active and neuron is healthy)
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
                        # Skip dead neurons
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
        # Reconstruct into 2 channels (Low/High Freq)
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
# 2. FREQUENCY DECOMPOSITION (Visual Front-End)
# =====================================================================
def extract_frequency_streams(frame_gray):
    """Breaks a grayscale image into Low-Pass (Blobs) and High-Pass (Edges)"""
    # 1. Low Frequency Stream (Blurred)
    low_freq = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    
    # 2. High Frequency Stream (Edges/Details)
    high_freq = cv2.subtract(frame_gray, low_freq)
    high_freq = cv2.equalizeHist(high_freq) # Boost contrast of edges
    
    # Stack into a 2-channel tensor
    tensor_lf = torch.from_numpy(low_freq).float() / 255.0
    tensor_hf = torch.from_numpy(high_freq).float() / 255.0
    stacked = torch.stack((tensor_lf, tensor_hf), dim=0).unsqueeze(0)
    
    return stacked.to(device)


# =====================================================================
# 3. LESION LAB GUI (Tkinter + OpenCV)
# =====================================================================
class LesionLabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Topological Lesion Study Lab")
        self.root.configure(bg="#111115")
        
        self.RES = 128
        self.N_NEURONS = 128
        self.cortex = LesionableCortex(in_channels=2, num_neurons=self.N_NEURONS).to(device)
        self.cemi = 0.0
        
        self.cap = cv2.VideoCapture(0)
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        # --- Control Panel ---
        ctrl_frame = tk.Frame(self.root, bg="#222", pady=10)
        ctrl_frame.pack(fill=tk.X)
        
        self.btn_freeze = tk.Button(ctrl_frame, text="🧠 FREEZE LEARNING (Oja OFF)", 
                                    bg="#cc5555", fg="white", font=("Courier", 10, "bold"), command=self.toggle_learning)
        self.btn_freeze.pack(side=tk.LEFT, padx=10)
        
        self.btn_reset_lesions = tk.Button(ctrl_frame, text="⚕️ HEAL ALL LESIONS", 
                                          bg="#55cc55", fg="black", font=("Courier", 10, "bold"), command=self.heal_all)
        self.btn_reset_lesions.pack(side=tk.LEFT, padx=10)
        
        self.status_lbl = tk.Label(ctrl_frame, text="Status: LEARNING ACTIVE", bg="#222", fg="#55cc55", font=("Courier", 12, "bold"))
        self.status_lbl.pack(side=tk.RIGHT, padx=20)
        
        # --- Main Display ---
        self.display_lbl = tk.Label(self.root, bg="#111")
        self.display_lbl.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- Lesion Grid (Clickable) ---
        grid_frame = tk.Frame(self.root, bg="#111", pady=10)
        grid_frame.pack(fill=tk.X)
        tk.Label(grid_frame, text="NEURON TISSUE (Click to Lesion/Heal Blocks):", bg="#111", fg="white", font=("Courier", 10)).pack()
        
        self.tissue_canvas = tk.Canvas(grid_frame, width=1024, height=40, bg="#222", highlightthickness=0)
        self.tissue_canvas.pack(pady=5)
        self.tissue_canvas.bind("<Button-1>", self.on_tissue_click)
        self.draw_tissue_bar()

    def toggle_learning(self):
        self.cortex.learning_active = not self.cortex.learning_active
        if self.cortex.learning_active:
            self.btn_freeze.config(text="🧠 FREEZE LEARNING (Oja OFF)", bg="#cc5555")
            self.status_lbl.config(text="Status: LEARNING ACTIVE", fg="#55cc55")
        else:
            self.btn_freeze.config(text="🔓 UNFREEZE (Oja ON)", bg="#55cc55")
            self.status_lbl.config(text="Status: FROZEN (Ready for Lesion Study)", fg="#cc5555")

    def heal_all(self):
        self.cortex.lesion_mask.fill_(1.0)
        self.draw_tissue_bar()

    def draw_tissue_bar(self):
        self.tissue_canvas.delete("all")
        mask = self.cortex.lesion_mask.squeeze().cpu().numpy()
        block_w = 1024 / self.N_NEURONS
        for i in range(self.N_NEURONS):
            color = "#00ff88" if mask[i] > 0.5 else "#333333"
            self.tissue_canvas.create_rectangle(i*block_w, 0, (i+1)*block_w, 40, fill=color, outline="#111")

    def on_tissue_click(self, event):
        # Determine which block was clicked
        block_w = 1024 / self.N_NEURONS
        n_idx = int(event.x / block_w)
        if 0 <= n_idx < self.N_NEURONS:
            # Toggle lesion state (silence a block of 8 neurons around the click for impact)
            current_state = self.cortex.lesion_mask[0, n_idx, 0, 0].item()
            new_state = 0.0 if current_state > 0.5 else 1.0
            
            # Apply to a cluster to make the scotoma visible
            for offset in range(-4, 5):
                target = n_idx + offset
                if 0 <= target < self.N_NEURONS:
                    self.cortex.lesion_mask[0, target, 0, 0] = new_state
                    
            self.draw_tissue_bar()

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Front-End: Frequency Decomposition
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.RES, self.RES))
            x_t = extract_frequency_streams(gray) # (1, 2, H, W)
            
            # 2. The Cortex
            heatmap, spikes = self.cortex.step(x_t, self.cemi)
            
            spike_rate = spikes.mean().item() / self.cortex.sparsity
            self.cemi = self.cemi * 0.8 + spike_rate * 0.2
            
            # 3. Reconstruct Mind's Eye
            mind_eye = self.cortex.reconstruct_mind_eye(spikes) # (1, 2, H, W)
            
            # --- Rendering ---
            # Panel 1: Reality
            real_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Panel 2: The Two Frequency Streams (Cyan = High/Edges, Orange = Low/Blobs)
            low_f = x_t[0, 0].cpu().numpy()
            high_f = x_t[0, 1].cpu().numpy()
            stream_vis = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            stream_vis[:,:,0] = high_f * 2.0 # Blue/Cyan for edges
            stream_vis[:,:,1] = (high_f + low_f) * 0.5 # Green shared
            stream_vis[:,:,2] = low_f * 1.5  # Red/Orange for blobs
            stream_vis = (np.clip(stream_vis, 0, 1) * 255).astype(np.uint8)
            
            # Panel 3: Mind's Eye (Reconstructed Streams)
            recon_low = mind_eye[0, 0].cpu().numpy()
            recon_high = mind_eye[0, 1].cpu().numpy()
            mind_vis = np.zeros((self.RES, self.RES, 3), dtype=np.float32)
            mind_vis[:,:,0] = recon_high * 2.0
            mind_vis[:,:,1] = (recon_high + recon_low) * 0.5
            mind_vis[:,:,2] = recon_low * 1.5
            mind_vis = (np.clip(mind_vis, 0, 1) * 255).astype(np.uint8)
            
            # Stack and scale
            display = np.hstack((real_bgr, stream_vis, mind_vis))
            display = cv2.resize(display, (1200, 400), interpolation=cv2.INTER_NEAREST)
            
            cv2.putText(display, "1. REALITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "2. FREQUENCY STREAMS (Orange=Low, Cyan=High)", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "3. MIND'S EYE RECONSTRUCTION", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            self.display_lbl.imgtk = img
            self.display_lbl.configure(image=img)
            
        self.root.after(20, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = LesionLabApp(root)
    root.mainloop()