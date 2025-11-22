import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import json
import time
from pathlib import Path

from modules import UNet1D
from diffusion import Diffusion

app = Flask(__name__)

model = None
diffusion = None
device = None

def load_model():
    global model, diffusion, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    print(f"Loading model on {device}...")
    model = UNet1D(seq_len=32, in_channels=2, cond_channels=2, base_channels=64)
    
    model_path = Path("models/diffusion_best.pt")
    if not model_path.exists():
        model_path = Path("models/diffusion_final.pt")
        
    if not model_path.exists():
        print("Warning: Model not found. Please train first.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    print("Model loaded.")

def smooth_path(path_x, path_y, window_size=5):
    if len(path_x) < window_size:
        return path_x, path_y
    
    kernel = np.ones(window_size) / window_size
    pad = window_size // 2
    
    px = np.pad(path_x, (pad, pad), mode='edge')
    py = np.pad(path_y, (pad, pad), mode='edge')
    
    smooth_x = np.convolve(px, kernel, mode='valid')
    smooth_y = np.convolve(py, kernel, mode='valid')
    
    smooth_x[0] = path_x[0]
    smooth_x[-1] = path_x[-1]
    smooth_y[0] = path_y[0]
    smooth_y[-1] = path_y[-1]
    
    return smooth_x, smooth_y

def clamp_deltas(path_x, path_y, max_delta=50.0):
    for i in range(1, len(path_x)):
        dx = path_x[i] - path_x[i-1]
        dy = path_y[i] - path_y[i-1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > max_delta:
            ratio = max_delta / dist
            path_x[i] = path_x[i-1] + dx * ratio
            path_y[i] = path_y[i-1] + dy * ratio
    return path_x, path_y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect')
def collect():
    return render_template('collect.html')

@app.route('/api/save_trace', methods=['POST'])
def save_trace():
    data = request.json
    outfile = Path("data/human_traces.jsonl")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    with outfile.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(data))
        fp.write("\n")
        
    count = 0
    if outfile.exists():
        with outfile.open("r", encoding="utf-8") as fp:
            count = sum(1 for _ in fp)
            
    return jsonify({"status": "success", "count": count})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    start_x = float(data.get('start_x', 0))
    start_y = float(data.get('start_y', 0))
    end_x = float(data.get('end_x', 100))
    end_y = float(data.get('end_y', 100))
    seed = data.get('seed')
    steps = int(data.get('steps', 50))
    smooth = data.get('smooth', True)
    
    if seed is not None:
        torch.manual_seed(seed)
        
    start_time = time.time()
    
    dx = end_x - start_x
    dy = end_y - start_y
    scale = max(abs(dx), abs(dy), 1.0)
    
    target = torch.tensor([[dx/scale, dy/scale]], dtype=torch.float32).to(device)
    shape = (1, 2, 32)
    
    samples = diffusion.sample_ddim(target, shape, steps=steps, clamp_boundaries=smooth)
    
    path = samples[0].cpu().numpy()
    path_x = path[0] * scale + start_x
    path_y = path[1] * scale + start_y
    
    if smooth:
        path_x, path_y = smooth_path(path_x, path_y)
        path_x, path_y = clamp_deltas(path_x, path_y, max_delta=scale*0.2)

    result_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x, path_y)]
    
    duration = time.time() - start_time
    
    return jsonify({
        "path": result_path,
        "stats": {
            "duration": f"{duration:.2f}s"
        }
    })

@app.route('/stream_generate')
def stream_generate():
    start_x = float(request.args.get('start_x', 0))
    start_y = float(request.args.get('start_y', 0))
    end_x = float(request.args.get('end_x', 100))
    end_y = float(request.args.get('end_y', 100))
    seed = request.args.get('seed')
    steps = int(request.args.get('steps', 50))
    smooth_arg = request.args.get('smooth', 'true')
    smooth = smooth_arg.lower() == 'true'
    
    if seed:
        torch.manual_seed(int(seed))
        
    def generate_stream():
        start_time = time.time()
        
        dx = end_x - start_x
        dy = end_y - start_y
        scale = max(abs(dx), abs(dy), 1.0)
        
        target = torch.tensor([[dx/scale, dy/scale]], dtype=torch.float32).to(device)
        shape = (1, 2, 32)
        
        b = shape[0]
        L = shape[-1]
        t_vals = torch.linspace(0, 1, L, device=device).view(1, 1, L)
        img = target.unsqueeze(-1) * t_vals
        
        times = torch.linspace(0, diffusion.timesteps - 1, steps, dtype=torch.long).flip(0).to(device)
        
        for i, t_step in enumerate(times):
            t = torch.full((b,), t_step, device=device, dtype=torch.long)
            noise_pred = diffusion.model(img, t, target)
            
            alpha_bar = diffusion.alphas_cumprod[t_step]
            if i == len(times) - 1:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            else:
                prev_step = times[i + 1]
                alpha_bar_prev = diffusion.alphas_cumprod[prev_step]
            
            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            if smooth:
                img[:, :, 0] = 0.0
                img[:, :, -1] = target
            
            path = img[0].cpu().detach().numpy()
            path_x = path[0] * scale + start_x
            path_y = path[1] * scale + start_y
            
            current_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x, path_y)]
            
            yield f"data: {json.dumps({'path': current_path, 'step': steps - i, 'total_steps': steps, 'duration': f'{time.time()-start_time:.2f}s', 'done': False})}\n\n"
            
        path = img[0].cpu().detach().numpy()
        path_x = path[0] * scale + start_x
        path_y = path[1] * scale + start_y
        
        if smooth:
            path_x, path_y = smooth_path(path_x, path_y)
            path_x, path_y = clamp_deltas(path_x, path_y, max_delta=scale*0.2)
            
        final_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x, path_y)]
        
        total_duration = time.time() - start_time
        yield f"data: {json.dumps({'done': True, 'duration': f'{total_duration:.2f}s', 'path': final_path})}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=7860, debug=True)
