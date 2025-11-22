import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

from modules import UNet1D
from diffusion import Diffusion

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = UNet1D(seq_len=args.seq_len, in_channels=2, cond_channels=2, base_channels=64)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    
    dx = args.end_x - args.start_x
    dy = args.end_y - args.start_y
    scale = max(abs(dx), abs(dy), 1.0)
    
    target = torch.tensor([[dx/scale, dy/scale]], dtype=torch.float32).to(device) # [1, 2]
    
    print(f"Generating trace from ({args.start_x}, {args.start_y}) to ({args.end_x}, {args.end_y})...")
    shape = (1, 2, args.seq_len)

    samples = diffusion.sample_ddim(target, shape, steps=50)
    
    # denorm
    path = samples[0].cpu().numpy() # [2, L]
    path_x = path[0] * scale + args.start_x
    path_y = path[1] * scale + args.start_y
    
    # smoothing paths
    def smooth_path(px, py, window_size=5):
        if len(px) < window_size: 
            return px, py
        kernel = np.ones(window_size) / window_size
        pad = window_size // 2
        sx = np.convolve(np.pad(px, (pad,pad), 'edge'), kernel, 'valid')
        sy = np.convolve(np.pad(py, (pad,pad), 'edge'), kernel, 'valid')
        sx[0], sx[-1] = px[0], px[-1]
        sy[0], sy[-1] = py[0], py[-1]
        return sx, sy

    path_x, path_y = smooth_path(path_x, path_y)

    # out
    result = []
    for x, y in zip(path_x, path_y):
        result.append({"x": float(x), "y": float(y)})
        
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
        
    if args.plot:
        plt.figure(figsize=(10, 6))
        plt.plot(path_x, path_y, label='Generated Path')
        plt.scatter([args.start_x], [args.start_y], c='green', label='Start')
        plt.scatter([args.end_x], [args.end_y], c='red', label='End')
        plt.legend()
        plt.grid(True)
        plt.title("Generated Trace")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/diffusion_final.pt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--start_x", type=float, required=True)
    parser.add_argument("--start_y", type=float, required=True)
    parser.add_argument("--end_x", type=float, required=True)
    parser.add_argument("--end_y", type=float, required=True)
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    generate(args)
