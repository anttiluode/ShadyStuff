import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting Zeta Grating Cortex Lab on {device}...")

# =====================================================================
# 1. THE ZETA GRATING CORTEX (Explicit Parameterization)
# =====================================================================
class ZetaGratingCortex(nn.Module):
    def __init__(self, in_channels=2, num_neurons=128):
        super().__init__()
        self.num_neurons = num_neurons
        # The Fibonacci Diffraction Grating
        self.taus = [1, 2, 3, 5, 8, 13, 21] 
        self.patch_dim = in_channels * 9 
        
        # The Learned Grating Templates (m)
        m_init = torch.randn(num_neurons, len(self.taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)
        
        # The Continuous Analog Pool
        self.ais_pool = None 
        
        # --- EXPOSED PHYSICS PARAMETERS ---
        self.eta = 0.15                # Oja Plasticity
        self.ais_integration = 0.30    # How fast resonance fills the AIS
        self.base_thresh = 0.05        # The static Rajapinta boundary
        self.cemi_coupling = 0.60      # How much CEMI lowers the boundary
        self.sparsity = 0.05           # Lateral inhibition strictness

    def step(self, image_tensor, current_cemi):
        B, C, H, W = image_tensor.shape
        if self.ais_pool is None or self.ais_pool.shape[-2:] != (H, W):
            self.ais_pool = torch.zeros(1, self.num_neurons, H, W, device=device)
            
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        
        # 1. Sample the Fibonacci Delay Manifold
        for i, tau in enumerate(self.taus):
            pad = tau
            padded = F.pad(image_tensor, (pad, pad, pad, pad), mode='reflect')
            unfolded = F.unfold(padded, kernel_size=3, dilation=tau)
            v = unfolded.view(B, self.patch_dim, H, W)
            v = F.normalize(v, p=2, dim=1)
            
            m_tau = self.m[:, i, :] 
            dot = torch.einsum('b c h w, n c -> b n h w', v, m_tau)
            total_res += dot ** 2 # Moiré Resonance
            
        avg_res = total_res / len(self.taus)
        
        # 2. The Analog AIS Integration (The buildup to the Rajapinta)
        # Using the slider value to determine the leak vs integration
        self.ais_pool = self.ais_pool * (1.0 - self.ais_integration) + avg_res * self.ais_integration
        
        # 3. The Dynamic Rajapinta (Thresholding)
        # CEMI field acts as an ephaptic bridge, lowering the barrier when active
        current_thresh = self.base_thresh * (1.0 - current_cemi * self.cemi_coupling)
        current_thresh = max(0.001, current_thresh) # Prevent absolute zero collapse
        
        # 4. Lateral Inhibition (Winners take all)
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_indices = torch.topk(self.ais_pool, k, dim=1)
        sparse_mask = torch.zeros_like(self.ais_pool, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_indices, True)
        
        # 5. The Collapse (Digital Spike)
        spikes = ((self.ais_pool > current_thresh) & sparse_mask).float()
        
        # Reset the AIS pool for neurons that spiked (Refractory discharge)
        self.ais_pool = self.ais_pool * (1.0 - spikes)
        
        # 6. Oja's Physical Rotation (Learning)
        if self.eta > 0 and spikes.sum() > 0:
            spikes_flat = spikes.view(self.num_neurons, -1)
            active_neurons = spikes_flat.sum(dim=1) > 0
            
            if active_neurons.any():
                for i, tau in enumerate(self.taus):
                    pad = tau
                    padded = F.pad(image_tensor, (pad, pad, pad, pad), mode='reflect')
                    unfolded = F.unfold(padded, kernel_size=3, dilation=tau).view(B, self.patch_dim, H, W)
                    v_norm = F.normalize(unfolded, p=2, dim=1)[0]
                    
                    for n in torch.where(active_neurons)[0]:
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
# 2. FRONT-END DECOMPOSITION
# =====================================================================
def extract_frequency_streams(frame_gray):
    low_freq = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    high_freq = cv2.subtract(frame_gray, low_freq)
    high_freq = cv2.equalizeHist(high_freq)
    
    tensor_lf = torch.from_numpy(low_freq).float() / 255.0
    tensor_hf = torch.from_numpy(high_freq).float() / 255.0
    return torch.stack((tensor_lf, tensor_hf), dim=0).unsqueeze(0).to(device)

# =====================================================================
# 3. INTERACTIVE LAB GUI
# =====================================================================
class ZetaGratingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zeta Grating Cortex: Rajapinta Lab")
        self.root.configure(bg="#111115")
        
        self.RES = 128
        self.cortex = ZetaGratingCortex(num_neurons=128).to(device)
        self.cemi = 0.0
        
        self.cap = cv2.VideoCapture(0)
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        # --- TOP CONTROLS (The Physics Parameters) ---
        ctrl_frame = tk.Frame(self.root, bg="#222", pady=10)
        ctrl_frame.pack(fill=tk.X)
        
        def make_slider(parent, label, min_v, max_v, default, attr_name):
            frame = tk.Frame(parent, bg="#222")
            frame.pack(side=tk.LEFT, padx=15)
            tk.Label(frame, text=label, bg="#222", fg="#00ffcc", font=("Courier", 9, "bold")).pack()
            s = tk.Scale(frame, from_=min_v, to=max_v, resolution=0.01, orient=tk.HORIZONTAL, 
                         bg="#333", fg="white", length=150, highlightthickness=0)
            s.set(default)
            s.pack()
            # Bind slider directly to the cortex attribute
            s.config(command=lambda val, a=attr_name: setattr(self.cortex, a, float(val)))
            return s

        self.sl_ais = make_slider(ctrl_frame, "AIS Integration Rate", 0.01, 1.0, 0.30, "ais_integration")
        self.sl_thresh = make_slider(ctrl_frame, "Rajapinta Threshold", 0.01, 0.5, 0.05, "base_thresh")
        self.sl_cemi = make_slider(ctrl_frame, "CEMI Coupling", 0.0, 1.0, 0.60, "cemi_coupling")
        self.sl_eta = make_slider(ctrl_frame, "Oja Plasticity (eta)", 0.0, 0.5, 0.15, "eta")
        
        self.cemi_lbl = tk.Label(ctrl_frame, text="GLOBAL CEMI: 0.00", bg="#222", fg="#ff5555", font=("Courier", 14, "bold"))
        self.cemi_lbl.pack(side=tk.RIGHT, padx=20)
        
        # --- DISPLAY ---
        self.display_lbl = tk.Label(self.root, bg="#111")
        self.display_lbl.pack(fill=tk.BOTH, expand=True, pady=10)

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.RES, self.RES))
            x_t = extract_frequency_streams(gray)
            
            # Step the Physics
            heatmap, spikes = self.cortex.step(x_t, self.cemi)
            
            # Update Global CEMI Field
            spike_rate = spikes.mean().item() / self.cortex.sparsity
            self.cemi = self.cemi * 0.8 + spike_rate * 0.2
            self.cemi_lbl.config(text=f"GLOBAL CEMI: {self.cemi:.3f}")
            
            # Reconstruct the Inverse Manifold
            mind_eye = self.cortex.reconstruct_mind_eye(spikes)
            
            # Rendering
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
            
            cv2.putText(display, "1. OPTICAL REALITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "2. DIFFRACTION GRATING (In)", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display, "3. INVERSE RAJAPINTA (Out)", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            self.display_lbl.imgtk = img
            self.display_lbl.configure(image=img)
            
        self.root.after(20, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ZetaGratingApp(root)
    root.mainloop()