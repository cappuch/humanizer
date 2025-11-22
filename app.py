import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import json
import time
from pathlib import Path
import threading
import subprocess
import sys

HF_MODE = "--hf" in sys.argv

from modules import UNet1D
from diffusion import Diffusion

app = Flask(__name__)

model = None
diffusion = None
device = None
training_in_progress = False
model_lock = threading.Lock()

elo_ratings = {
    'base': 1500.0,
    'human': 1500.0,
    'rlhf': 1500.0,
    'rl_classifier': 1500.0
}
elo_lock = threading.Lock()

def calculate_elo_update(winner_elo, loser_elo, k=32):
    """Calculate ELO rating changes for winner and loser"""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    
    winner_new = winner_elo + k * (1 - expected_winner)
    loser_new = loser_elo + k * (0 - expected_loser)
    
    return winner_new, loser_new

def load_elo_ratings():
    """Load ELO ratings from file"""
    global elo_ratings
    elo_file = Path("data/elo_ratings.json")
    if elo_file.exists():
        with elo_file.open("r") as f:
            elo_ratings = json.load(f)
            print(f"Loaded ELO ratings: {elo_ratings}")
    else:
        print("No ELO ratings file found, starting with default ratings")

def save_elo_ratings():
    """Save ELO ratings to file"""
    elo_file = Path("data/elo_ratings.json")
    elo_file.parent.mkdir(parents=True, exist_ok=True)
    with elo_file.open("w") as f:
        json.dump(elo_ratings, f, indent=2)

def save_elo_snapshot(reason="comparison"):
    """Save a historical snapshot of ELO ratings"""
    import datetime
    history_file = Path("data/elo_history.jsonl")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    snapshot = {
        'timestamp': datetime.datetime.now().isoformat(),
        'reason': reason,
        'ratings': dict(elo_ratings)
    }
    
    with history_file.open("a") as f:
        f.write(json.dumps(snapshot) + "\n")

model_base = None
model_rlhf = None
model_rl_classifier = None
diffusion_base = None
diffusion_rlhf = None
diffusion_rl_classifier = None

def load_model():
    global model, diffusion, device
    global model_base, model_rlhf, model_rl_classifier
    global diffusion_base, diffusion_rlhf, diffusion_rl_classifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    print(f"Loading models on {device}...")
    
    model = UNet1D(seq_len=32, in_channels=2, cond_channels=2, base_channels=64)
    model_path = Path("models/diffusion_rlhf_best.pt")
    if not model_path.exists():
        model_path = Path("models/diffusion_best.pt")
    if not model_path.exists():
        model_path = Path("models/diffusion_final.pt")
        
    if not model_path.exists():
        print("Warning: Model not found. Please train first.")
    else:
        print(f"Loading primary model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    
    model_base = UNet1D(seq_len=32, in_channels=2, cond_channels=2, base_channels=64)
    base_path = Path("models/diffusion_best.pt")
    if base_path.exists():
        print(f"Loading base model from {base_path}")
        model_base.load_state_dict(torch.load(base_path, map_location=device))
        model_base.eval()
        diffusion_base = Diffusion(model_base, timesteps=50, beta_end=0.2, device=device)
    else:
        print("Warning: Base model not found, using primary model")
        model_base = model
        diffusion_base = diffusion
    
    # Model 2: RLHF
    model_rlhf = UNet1D(seq_len=32, in_channels=2, cond_channels=2, base_channels=64)
    rlhf_path = Path("models/diffusion_rlhf_best.pt")
    if rlhf_path.exists():
        print(f"Loading RLHF model from {rlhf_path}")
        model_rlhf.load_state_dict(torch.load(rlhf_path, map_location=device))
        model_rlhf.eval()
        diffusion_rlhf = Diffusion(model_rlhf, timesteps=50, beta_end=0.2, device=device)
    else:
        print("Warning: RLHF model not found, using primary model")
        model_rlhf = model
        diffusion_rlhf = diffusion
    
    model_rl_classifier = UNet1D(seq_len=32, in_channels=2, cond_channels=2, base_channels=64)
    rl_classifier_path = Path("models/diffusion_rl_classifier_best.pt")
    if rl_classifier_path.exists():
        print(f"Loading RL-Classifier model from {rl_classifier_path}")
        model_rl_classifier.load_state_dict(torch.load(rl_classifier_path, map_location=device))
        model_rl_classifier.eval()
        diffusion_rl_classifier = Diffusion(model_rl_classifier, timesteps=50, beta_end=0.2, device=device)
    else:
        print("Warning: RL-Classifier model not found, using primary model")
        model_rl_classifier = model
        diffusion_rl_classifier = diffusion
    
    print("All models loaded.")

def reload_model():
    global model, diffusion, model_rlhf, diffusion_rlhf
    with model_lock:
        print("Reloading RLHF model with latest weights...")
        model_path = Path("models/diffusion_rlhf_best.pt")
        if model_path.exists():
            model_rlhf.load_state_dict(torch.load(model_path, map_location=device))
            model_rlhf.eval()
            diffusion_rlhf = Diffusion(model_rlhf, timesteps=50, beta_end=0.2, device=device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
            print("RLHF model reloaded successfully!")
        else:
            print("RLHF model not found, keeping current model.")

def trigger_rlhf_training():
    global training_in_progress
    try:
        result = subprocess.run(
            ["uv", "run", "train_multimodel_rlhf.py", 
             "--target_model", "rlhf",
             "--pretrained_model", "models/diffusion_rlhf_best.pt",
             "--epochs", "25", 
             "--lr", "5e-5"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            reload_model()
            with elo_lock:
                save_elo_snapshot(reason="after_rlhf_training")
        else:
            print("RLHF training failed:")
            print(result.stdout)
            print(result.stderr)            
    except Exception as e:
        print(f"Error during RLHF training: {e}")
    finally:
        training_in_progress = False

def check_and_train_rlhf(current_count):
    global training_in_progress
    
    if current_count > 0 and current_count % 50 == 0 and not training_in_progress:
        training_in_progress = True
        training_thread = threading.Thread(target=trigger_rlhf_training, daemon=True)
        training_thread.start()
        return True
    return False

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
    return render_template('index.html', hf_mode=HF_MODE)

@app.route('/collect')
def collect():
    if HF_MODE: return "Not Found", 404
    return render_template('collect.html')

@app.route('/rlhf')
def rlhf():
    if HF_MODE: return "Not Found", 404
    return render_template('rlhf.html')

@app.route('/leaderboard')
def leaderboard():
    if HF_MODE: return "Not Found", 404
    return render_template('leaderboard.html')

@app.route('/api/training_status', methods=['GET'])
def training_status():
    if HF_MODE: return "Not Found", 404
    return jsonify({
        "training_in_progress": training_in_progress
    })

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    if HF_MODE: return "Not Found", 404
    with elo_lock:
        comparisons_file = Path("data/model_comparisons.jsonl")
        total_comparisons = 0
        if comparisons_file.exists():
            with comparisons_file.open("r") as f:
                total_comparisons = sum(1 for _ in f)
        
        leaderboard = [
            {
                'name': 'Human',
                'model_id': 'human',
                'elo': round(elo_ratings['human'], 1)
            },
            {
                'name': 'Base',
                'model_id': 'base',
                'elo': round(elo_ratings['base'], 1)
            },
            {
                'name': 'RLHF',
                'model_id': 'rlhf',
                'elo': round(elo_ratings['rlhf'], 1)
            },
            {
                'name': 'RL Classifier',
                'model_id': 'rl_classifier',
                'elo': round(elo_ratings['rl_classifier'], 1)
            }
        ]
        
        leaderboard.sort(key=lambda x: x['elo'], reverse=True)
        
        return jsonify({
            'leaderboard': leaderboard,
            'total_comparisons': total_comparisons
        })

@app.route('/api/elo_history', methods=['GET'])
def get_elo_history():
    if HF_MODE: return "Not Found", 404
    history_file = Path("data/elo_history.jsonl")
    history = []
    
    if history_file.exists():
        with history_file.open("r") as f:
            for line in f:
                try:
                    snapshot = json.loads(line)
                    history.append(snapshot)
                except Exception:
                    pass
    
    return jsonify({'history': history})

@app.route('/api/get_comparison', methods=['GET'])
def get_comparison():
    if HF_MODE: return "Not Found", 404
    import random
    
    data_file = Path("data/human_traces.jsonl")
    if not data_file.exists():
        return jsonify({"error": "No data available"}), 404
    
    samples = []
    with data_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
    
    if not samples:
        return jsonify({"error": "No valid samples"}), 404
    
    sample = random.choice(samples)
    
    start_x = sample['start']['x']
    start_y = sample['start']['y']
    end_x = sample['end']['x']
    end_y = sample['end']['y']
    scale = sample['scale']
    
    include_human = random.random() < 0.5
    
    dx = end_x - start_x
    dy = end_y - start_y
    target = torch.tensor([[dx/scale, dy/scale]], dtype=torch.float32).to(device)
    shape = (1, 2, 32)
    
    with model_lock:
        if include_human:
            human_path_raw = sample['path']
            human_path = [
                {
                    'x': p['x'] * scale + start_x,
                    'y': p['y'] * scale + start_y
                }
                for p in human_path_raw
            ]
            base_path = human_path
            base_path_normalized = human_path_raw
        else:
            samples_base = diffusion_base.sample_ddim(target, shape, steps=50, clamp_boundaries=True)
            path_base = samples_base[0].cpu().numpy()
            path_x_base = path_base[0] * scale + start_x
            path_y_base = path_base[1] * scale + start_y
            path_x_base, path_y_base = smooth_path(path_x_base, path_y_base)
            path_x_base, path_y_base = clamp_deltas(path_x_base, path_y_base, max_delta=scale*0.2)
            base_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x_base, path_y_base)]
            base_path_normalized = [{'x': (x - start_x) / scale, 'y': (y - start_y) / scale} for x, y in zip(path_x_base, path_y_base)]
        
        samples_rlhf = diffusion_rlhf.sample_ddim(target, shape, steps=50, clamp_boundaries=True)
        path_rlhf = samples_rlhf[0].cpu().numpy()
        path_x_rlhf = path_rlhf[0] * scale + start_x
        path_y_rlhf = path_rlhf[1] * scale + start_y
        path_x_rlhf, path_y_rlhf = smooth_path(path_x_rlhf, path_y_rlhf)
        path_x_rlhf, path_y_rlhf = clamp_deltas(path_x_rlhf, path_y_rlhf, max_delta=scale*0.2)
        
        samples_rl_classifier = diffusion_rl_classifier.sample_ddim(target, shape, steps=50, clamp_boundaries=True)
        path_rl_classifier = samples_rl_classifier[0].cpu().numpy()
        path_x_rl_classifier = path_rl_classifier[0] * scale + start_x
        path_y_rl_classifier = path_rl_classifier[1] * scale + start_y
        path_x_rl_classifier, path_y_rl_classifier = smooth_path(path_x_rl_classifier, path_y_rl_classifier)
        path_x_rl_classifier, path_y_rl_classifier = clamp_deltas(path_x_rl_classifier, path_y_rl_classifier, max_delta=scale*0.2)
    
    rlhf_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x_rlhf, path_y_rlhf)]
    rl_classifier_path = [{"x": float(x), "y": float(y)} for x, y in zip(path_x_rl_classifier, path_y_rl_classifier)]
    
    models = ['human' if include_human else 'base', 'rlhf', 'rl_classifier']
    paths = {
        'human' if include_human else 'base': base_path,
        'rlhf': rlhf_path,
        'rl_classifier': rl_classifier_path
    }
    
    normalized_paths = {
        'human' if include_human else 'base': base_path_normalized,
        'rlhf': [{'x': (x - start_x) / scale, 'y': (y - start_y) / scale} for x, y in zip(path_x_rlhf, path_y_rlhf)],
        'rl_classifier': [{'x': (x - start_x) / scale, 'y': (y - start_y) / scale} for x, y in zip(path_x_rl_classifier, path_y_rl_classifier)]
    }
    
    random.shuffle(models)
    model_mapping = {
        'A': models[0],
        'B': models[1],
        'C': models[2]
    }
    
    return jsonify({
        'sample_id': hash(json.dumps(sample)),
        'start': {'x': start_x, 'y': start_y},
        'end': {'x': end_x, 'y': end_y},
        'pathA': paths[models[0]],
        'pathB': paths[models[1]],
        'pathC': paths[models[2]],
        'model_mapping': model_mapping,
        'model_paths': normalized_paths
    })

@app.route('/api/save_comparison', methods=['POST'])
def save_comparison():
    if HF_MODE: return "Not Found", 404
    data = request.json
    outfile = Path("data/model_comparisons.jsonl")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    with outfile.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(data))
        fp.write("\n")
    
    with elo_lock:
        chosen_option = data['chosen_option']
        model_mapping = data['model_mapping']
        
        winner = model_mapping[chosen_option]
        losers = [model_mapping[opt] for opt in ['A', 'B', 'C'] if opt != chosen_option]
        
        for loser in losers:
            winner_elo = elo_ratings[winner]
            loser_elo = elo_ratings[loser]
            
            new_winner_elo, new_loser_elo = calculate_elo_update(winner_elo, loser_elo)
            
            elo_ratings[winner] = new_winner_elo
            elo_ratings[loser] = new_loser_elo
        
        save_elo_ratings()
        
    count = 0
    if outfile.exists():
        with outfile.open("r", encoding="utf-8") as fp:
            count = sum(1 for _ in fp)
    
    with elo_lock:
        if count > 0 and count % 10 == 0:
            save_elo_snapshot(reason=f"after_{count}_comparisons")
    
    training_triggered = check_and_train_rlhf(count)
    
    response_data = {
        "status": "success",
        "count": count
    }
    
    if training_triggered:
        response_data["training_started"] = True
        response_data["message"] = f"🎉 Milestone reached! Background training started with {count} comparisons."
    
    return jsonify(response_data)

@app.route('/api/save_trace', methods=['POST'])
def save_trace():
    if HF_MODE: return "Not Found", 404
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
    
    with model_lock:
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
    load_elo_ratings()
    app.run(host='0.0.0.0', port=7860, debug=True)
