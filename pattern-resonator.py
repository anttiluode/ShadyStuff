import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# ---------------------------------------------------------
# THE RESONATOR (Coordinate Network)
# ---------------------------------------------------------
class ResonatorNet(nn.Module):
    """A continuous mathematical representation of the physical pattern."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Fourier feature mapping (helps learn high-frequency fractal detail)
        self.mapping = nn.Linear(2, 64) 
        self.net = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project inputs to higher frequencies
        x_proj = (2.0 * np.pi * self.mapping(x))
        x_mapped = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(x_mapped)

# ---------------------------------------------------------
# THE GUI APPLICATION
# ---------------------------------------------------------
class PatternResonatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAIT — Pattern Resonator & Dream Engine")
        self.root.geometry("1400x800")
        self.root.configure(bg="#07070f")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.original_img = None
        self.target_tensor = None
        self.coords_tensor = None
        self.is_training = False
        self.loss_history = []
        
        self.setup_ui()

    def setup_ui(self):
        # --- LEFT CONTROL PANEL ---
        ctrl_frame = tk.Frame(self.root, width=350, bg="#1a1a2e", padx=20, pady=20)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(ctrl_frame, text="PATTERN RESONATOR", fg="#28c8e0", bg="#1a1a2e", 
                 font=("Courier", 16, "bold")).pack(pady=(0, 20))
        
        tk.Button(ctrl_frame, text="1. Load Pattern Image", command=self.load_image, 
                  bg="#28c8e0", fg="black", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=10)
        
        self.btn_train = tk.Button(ctrl_frame, text="2. Internalize Geometry", command=self.start_training, 
                                   bg="#e06090", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.btn_train.pack(fill=tk.X, pady=10)

        # Resonance Controls (The Dream Sliders)
        tk.Label(ctrl_frame, text="--- RESONANCE CONTROLS ---", fg="#888", bg="#1a1a2e", font=("Courier", 10)).pack(pady=(30, 10))
        
        tk.Label(ctrl_frame, text="Phase Shift (Morphing)", fg="white", bg="#1a1a2e").pack(anchor=tk.W)
        self.slide_phase = ttk.Scale(ctrl_frame, from_=-2.0, to=2.0, value=0.0, orient=tk.HORIZONTAL, command=self.update_dream)
        self.slide_phase.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(ctrl_frame, text="Frequency Zoom (Fractal Scale)", fg="white", bg="#1a1a2e").pack(anchor=tk.W)
        self.slide_zoom = ttk.Scale(ctrl_frame, from_=0.1, to=5.0, value=1.0, orient=tk.HORIZONTAL, command=self.update_dream)
        self.slide_zoom.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(ctrl_frame, text="Nonlinear Twist", fg="white", bg="#1a1a2e").pack(anchor=tk.W)
        self.slide_twist = ttk.Scale(ctrl_frame, from_=-1.0, to=1.0, value=0.0, orient=tk.HORIZONTAL, command=self.update_dream)
        self.slide_twist.pack(fill=tk.X, pady=(0, 15))

        self.lbl_status = tk.Label(ctrl_frame, text="Ready.", fg="#60e090", bg="#1a1a2e", justify=tk.LEFT)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        # --- RIGHT PLOT PANEL ---
        self.fig = plt.Figure(figsize=(12, 8), facecolor="#07070f")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 3 Subplots: Original, Dream, Loss
        self.ax_orig = self.fig.add_subplot(221)
        self.ax_dream = self.fig.add_subplot(222)
        self.ax_loss = self.fig.add_subplot(212)
        
        for ax in [self.ax_orig, self.ax_dream, self.ax_loss]:
            ax.set_facecolor("#07070f")
            ax.tick_params(colors="#555")
            for sp in ax.spines.values(): sp.set_color("#222")
            
        self.ax_orig.set_title("Original Spatial Pattern", color="#28c8e0")
        self.ax_dream.set_title("Resonator Dream State", color="#e06090")
        self.ax_loss.set_title("Internalization Error (Loss)", color="#60e090")
        self.fig.tight_layout()

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not filepath: return
        
        # Load and process image into target tensors
        img = Image.open(filepath).convert('L')
        img = img.resize((128, 128)) # Standardize size for fast training
        self.original_img = np.array(img) / 255.0
        
        # Build standard coordinate grid [-1, 1]
        h, w = self.original_img.shape
        y, x = np.mgrid[-1:1:complex(0,h), -1:1:complex(0,w)]
        self.base_coords = np.stack([x.flatten(), y.flatten()], axis=-1)
        
        self.coords_tensor = torch.tensor(self.base_coords, dtype=torch.float32).to(self.device)
        self.target_tensor = torch.tensor(self.original_img.flatten(), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Display original
        self.ax_orig.clear()
        self.ax_orig.imshow(self.original_img, cmap='gray')
        self.ax_orig.set_title("Original Spatial Pattern", color="#28c8e0")
        self.ax_orig.axis('off')
        self.canvas.draw()
        
        self.btn_train.config(state=tk.NORMAL)
        self.lbl_status.config(text="Pattern loaded. Ready to internalize.")

    def start_training(self):
        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.model = ResonatorNet().to(self.device)
        self.loss_history = []
        
        threading.Thread(target=self._train_loop, daemon=True).start()

    def _train_loop(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        epochs = 1500
        for epoch in range(epochs):
            if not self.is_training: break
            
            optimizer.zero_grad()
            output = self.model(self.coords_tensor)
            loss = criterion(output, self.target_tensor)
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            # Update UI periodically
            if epoch % 50 == 0:
                self.root.after(0, self._update_plots, epoch, epochs, loss.item(), output.detach())
                
        self.is_training = False
        self.root.after(0, self.lbl_status.config, {"text": "Geometry Fully Internalized! Use sliders."})

    def _update_plots(self, epoch, total, loss_val, current_dream):
        # Update Loss Curve
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_history, color="#60e090", lw=2)
        self.ax_loss.set_title(f"Internalization: Epoch {epoch}/{total} | Loss: {loss_val:.5f}", color="#60e090")
        self.ax_loss.set_facecolor("#07070f")
        for sp in self.ax_loss.spines.values(): sp.set_color("#222")
        self.ax_loss.tick_params(colors="#555")
        
        # Update Dream image with base parameters
        h, w = self.original_img.shape
        img_out = current_dream.cpu().numpy().reshape(h, w)
        self.ax_dream.clear()
        self.ax_dream.imshow(img_out, cmap='viridis')
        self.ax_dream.set_title("Resonator Dream State", color="#e06090")
        self.ax_dream.axis('off')
        
        self.canvas.draw()

    def update_dream(self, event=None):
        if self.model is None or self.is_training: return
        
        # Get slider values
        phase = self.slide_phase.get()
        zoom = self.slide_zoom.get()
        twist = self.slide_twist.get()
        
        # Alter the physical geometry coordinates before feeding to network
        x = self.base_coords[:, 0] * zoom + phase
        y = self.base_coords[:, 1] * zoom + phase
        
        # Apply nonlinear twist
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x) + (twist * radius)
        
        x_new = radius * np.cos(angle)
        y_new = radius * np.sin(angle)
        
        altered_coords = np.stack([x_new, y_new], axis=-1)
        coords_t = torch.tensor(altered_coords, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            dream = self.model(coords_t).cpu().numpy()
            
        h, w = self.original_img.shape
        img_out = dream.reshape(h, w)
        
        self.ax_dream.clear()
        self.ax_dream.imshow(img_out, cmap='viridis')
        self.ax_dream.set_title(f"Shift: {phase:.2f} | Zoom: {zoom:.2f} | Twist: {twist:.2f}", color="#e06090")
        self.ax_dream.axis('off')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PatternResonatorGUI(root)
    root.mainloop()