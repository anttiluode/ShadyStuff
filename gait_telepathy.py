import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting Visual Cortex on {device}...")

# =====================================================================
# 1. SPATIAL GAIT CORTEX (With Inverse Rajapinta Engine)
# =====================================================================
class SpatialCortex(nn.Module):
    def __init__(self, in_channels=1, num_neurons=64, taus=[1, 2, 3, 5, 8], eta=0.15, sparsity=0.05):
        super().__init__()
        self.num_neurons = num_neurons
        self.taus = taus
        self.eta = eta
        self.sparsity = sparsity
        self.base_thresh = 0.05
        
        # 3x3 patch for each tau
        self.patch_dim = in_channels * 9 
        
        # The Geometric Templates (m)
        m_init = torch.randn(num_neurons, len(taus), self.patch_dim) * 0.3
        self.m = nn.Parameter(F.normalize(m_init, p=2, dim=2), requires_grad=False)
        
        self.ais = None 

    def step(self, image_tensor, current_cemi):
        B, C, H, W = image_tensor.shape
        if self.ais is None or self.ais.shape[-2:] != (H, W):
            self.ais = torch.zeros(1, self.num_neurons, H, W, device=device)
            
        total_res = torch.zeros(1, self.num_neurons, H, W, device=device)
        
        # --- FORWARD PASS (Encode Reality into Spikes) ---
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
        
        # Integration & CEMI Threshold
        self.ais = self.ais * 0.70 + avg_res * 0.30
        thresh = self.base_thresh * (1.0 - current_cemi * 0.6)
        
        # Lateral Inhibition (Sparsity)
        k = max(1, int(self.num_neurons * self.sparsity))
        _, topk_indices = torch.topk(self.ais, k, dim=1)
        sparse_mask = torch.zeros_like(self.ais, dtype=torch.bool)
        sparse_mask.scatter_(1, topk_indices, True)
        
        spikes = ((self.ais > thresh) & sparse_mask).float()
        self.ais = self.ais * (1.0 - spikes)
        
        # Oja's Rule (Learn the Geometry)
        if spikes.sum() > 0:
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
        
        global_heatmap = avg_res.mean(dim=1, keepdim=True)
        return global_heatmap, spikes

    # --- INVERSE RAJAPINTA ENGINE (Decode Spikes into Mind's Eye) ---
    def reconstruct_mind_eye(self, spikes):
        """
        Takes purely the binary spikes and the learned templates (m).
        Re-inflates them into a continuous 2D image using F.fold.
        """
        B, N, H, W = spikes.shape
        reconstruction = torch.zeros(B, 1, H, W, device=device)
        
        # Flatten spatial dimensions for matrix math
        spikes_flat = spikes.view(B, N, H*W) # (1, 64, H*W)
        
        for i, tau in enumerate(self.taus):
            # 1. Weight the templates by the spikes
            # spikes_transpose: (1, H*W, 64) | m_tau: (64, 9) -> Output: (1, H*W, 9)
            m_tau = self.m[:, i, :]
            patches = torch.bmm(spikes_flat.transpose(1, 2), m_tau.unsqueeze(0).expand(B, -1, -1))
            
            # Transpose to expected unfold shape: (1, 9, H*W)
            patches = patches.transpose(1, 2)
            
            # 2. FOLD the patches back into continuous 2D spatial overlap
            pad = tau
            folded = F.fold(patches, output_size=(H + pad*2, W + pad*2), 
                            kernel_size=3, dilation=tau)
            
            # Crop padding
            folded_cropped = folded[:, :, pad:-pad, pad:-pad]
            reconstruction += folded_cropped
            
        # Normalize the reconstruction to visible range
        recon_min = reconstruction.min()
        recon_max = reconstruction.max()
        if recon_max > recon_min:
            reconstruction = (reconstruction - recon_min) / (recon_max - recon_min)
            
        return reconstruction


# =====================================================================
# 2. THE TELEPATHY GUI (Webcam Loop)
# =====================================================================
# Settings
RES = 160 # Internal resolution (Keep low for 60fps)
cortex = SpatialCortex(num_neurons=64, taus=[1, 2, 3, 5, 8, 13]).to(device)
cemi = 0.0

print("Connecting to Eye (Webcam)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam failed.")
    exit()

cv2.namedWindow("GAIT Telepathy - Mind's Eye", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GAIT Telepathy - Mind's Eye", 1200, 400)

print("Running. Show the camera objects. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Preprocess Reality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (RES, RES))
    x_t = torch.from_numpy(gray_small).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 1. Forward Pass (Encode)
    heatmap, spikes = cortex.step(x_t, cemi)
    
    # Update CEMI
    spike_rate = spikes.mean().item() / cortex.sparsity
    cemi = cemi * 0.8 + spike_rate * 0.2
    
    # 2. INVERSE PASS (Decode)
    # We pass ONLY the discrete spikes. The cortex reconstructs the reality.
    mind_eye = cortex.reconstruct_mind_eye(spikes)
    
    # --- Rendering ---
    # Reality Panel
    real_render = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2BGR)
    
    # Heatmap Panel (Where it is paying attention)
    heat_np = heatmap[0, 0].cpu().numpy()
    heat_norm = np.clip((heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-5) * 255, 0, 255).astype(np.uint8)
    heat_render = cv2.applyColorMap(heat_norm, cv2.COLORMAP_INFERNO)
    
    # Mind's Eye Panel (What it is actually seeing/feeling)
    mind_np = (mind_eye[0, 0].cpu().numpy() * 255).astype(np.uint8)
    mind_render = cv2.applyColorMap(mind_np, cv2.COLORMAP_OCEAN) # Blue/Green ghost map
    
    # Combine panels horizontally
    display = np.hstack((real_render, heat_render, mind_render))
    display = cv2.resize(display, (1200, 400), interpolation=cv2.INTER_NEAREST)
    
    # Labels
    cv2.putText(display, "1. REALITY (INPUT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(display, f"2. RESONANCE HEATMAP (CEMI: {cemi:.2f})", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(display, "3. INVERSE MANIFOLD (MIND'S EYE)", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("GAIT Telepathy - Mind's Eye", display)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()