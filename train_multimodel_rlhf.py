import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import numpy as np
import json

from modules import UNet1D
from diffusion import Diffusion

def custom_collate(batch):
    result = {
        'x': torch.stack([item['x'] for item in batch]),
        'cond': torch.stack([item['cond'] for item in batch]),
        'is_chosen': torch.stack([item['is_chosen'] for item in batch])
    }
    
    has_better = any('better_x' in item for item in batch)
    
    if has_better:
        better_x_list = []
        for item in batch:
            if 'better_x' in item:
                better_x_list.append(item['better_x'])
            else:
                better_x_list.append(item['x'])
        result['better_x'] = torch.stack(better_x_list)
    
    return result

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiModelComparisonDataset(Dataset):
    """Dataset for 3-way model comparison training"""
    def __init__(self, data_path, seq_len=32, target_model='rlhf'):
        self.seq_len = seq_len
        self.target_model = target_model  # which model we're training
        self.samples = []
        
        path = Path(data_path)
        if not path.exists():
            print(f"Warning: {data_path} not found. Dataset will be empty.")
            return

        with open(path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed = self._process_sample(obj)
                    if processed:
                        self.samples.append(processed)
                except Exception as e:
                    print(f"Error parsing line: {e}")
        
        print(f"Loaded {len(self.samples)} comparison samples for {target_model} training")

    def _process_sample(self, obj):
        """Process 3-way comparison sample"""
        chosen_option = obj['chosen_option']  # A, B, or C
        model_mapping = obj['model_mapping']  # {'A': 'base', 'B': 'rlhf', 'C': 'rl_classifier'} or 'human'
        model_paths = obj['model_paths']  # {'base': [...], 'rlhf': [...], 'rl_classifier': [...]} or 'human'
        
        chosen_model = model_mapping[chosen_option]
        
        # Skip samples where human was chosen (we can't train models to be "more human" directly)
        # But we can use them as positive examples for all models
        if chosen_model == 'human':
            # When human is chosen, all AI models "lost" - this is valuable training signal
            # We'll create samples for all AI models showing they need to improve
            if self.target_model == 'human':
                return None  # Can't train a "human" model
                
            target_path = model_paths.get(self.target_model)
            if not target_path:
                return None
        else:
            # Only create training samples relevant to our target model
            # Positive examples: when our model was chosen
            # Negative examples: when our model wasn't chosen
            
            target_path = model_paths.get(self.target_model)
            if not target_path:
                return None
        
        # Extract target from start/end points
        start = obj['start']
        end = obj['end']
        dx = end['x'] - start['x']
        dy = end['y'] - start['y']
        
        # Normalize target
        scale = max(abs(dx), abs(dy), 1.0)
        target = np.array([dx / scale, dy / scale], dtype=np.float32)
        
        # Process target model's path
        target_points = np.array([[p['x'], p['y']] for p in target_path], dtype=np.float32)
        target_resampled = self._resample(target_points, self.seq_len)
        target_resampled[0] = np.array([0.0, 0.0], dtype=np.float32)
        target_resampled[-1] = target
        target_resampled = target_resampled.transpose(1, 0).astype(np.float32)
        
        # Determine if this was a win or loss
        is_chosen = (chosen_model == self.target_model)
        
        # For losses, get the winner's path as the "better" example
        # Special case: if human won, use human path as "better"
        if not is_chosen:
            if chosen_model == 'human':
                # Human won - use human path as the gold standard
                better_path = model_paths.get('human')
                if not better_path:
                    # Fallback: if human path not in model_paths, skip
                    return None
            else:
                # Another model won
                better_path = model_paths[chosen_model]
                
            better_points = np.array([[p['x'], p['y']] for p in better_path], dtype=np.float32)
            better_resampled = self._resample(better_points, self.seq_len)
            better_resampled[0] = np.array([0.0, 0.0], dtype=np.float32)
            better_resampled[-1] = target
            better_resampled = better_resampled.transpose(1, 0).astype(np.float32)
        else:
            better_resampled = None
        
        return {
            'target_path': target_resampled,  # Our model's path
            'better_path': better_resampled,  # Winner's path (if we lost)
            'target': target,  # [2]
            'is_chosen': is_chosen
        }

    def _resample(self, points, target_len):
        """Same resampling logic as other datasets"""
        if len(points) < 2:
            return np.tile(points[0], (target_len, 1))
            
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_dist = cum_dist[-1]
        
        if total_dist == 0:
            return np.tile(points[0], (target_len, 1))
            
        new_dists = np.linspace(0, total_dist, target_len)
        new_x = np.interp(new_dists, cum_dist, points[:, 0])
        new_y = np.interp(new_dists, cum_dist, points[:, 1])
        
        return np.stack([new_x, new_y], axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {
            'x': torch.from_numpy(sample['target_path']),
            'cond': torch.from_numpy(sample['target']),
            'is_chosen': torch.tensor(sample['is_chosen'], dtype=torch.float32)
        }
        if sample['better_path'] is not None:
            result['better_x'] = torch.from_numpy(sample['better_path'])
        return result


def train_multi_model_rlhf(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    full_dataset = MultiModelComparisonDataset(args.data_path, seq_len=args.seq_len, target_model=args.target_model)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = UNet1D(seq_len=args.seq_len, in_channels=2, cond_channels=2, base_channels=64)
    
    pretrained_path = Path(args.pretrained_model)
    if pretrained_path.exists():
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}. Training from scratch.")
    
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Multi-Model RLHF]")
        train_loss = 0
        train_margin_loss = 0
        train_wins = 0
        train_total = 0
        
        for batch in pbar:
            x = batch['x'].to(device)  # [B, 2, L] - our model's paths
            cond = batch['cond'].to(device)  # [B, 2]
            is_chosen = batch['is_chosen'].to(device)  # [B]
            
            optimizer.zero_grad()
            
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            
            loss_ours = diffusion.p_losses(x, t, cond)
            
            if 'better_x' in batch:
                better_x = batch['better_x'].to(device)
                loss_better = diffusion.p_losses(better_x, t, cond)
                
                margin_loss = torch.clamp(args.margin - (loss_ours - loss_better), min=0.0)
                margin_loss = margin_loss * (1.0 - is_chosen)
                margin_loss = margin_loss.mean()
            else:
                margin_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = loss_ours + args.margin_weight * margin_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss_ours.item()
            train_margin_loss += margin_loss.item()
            train_wins += is_chosen.sum().item()
            train_total += is_chosen.size(0)
            
            pbar.set_postfix(
                loss=loss_ours.item(),
                margin=margin_loss.item(),
                win_rate=train_wins/train_total if train_total > 0 else 0,
                lr=optimizer.param_groups[0]['lr']
            )
            
        avg_train_loss = train_loss / len(train_loader)
        avg_margin_loss = train_margin_loss / len(train_loader)
        win_rate = train_wins / train_total if train_total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_margin_loss = 0
        val_wins = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                cond = batch['cond'].to(device)
                is_chosen = batch['is_chosen'].to(device)
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
                
                loss_ours = diffusion.p_losses(x, t, cond)
                
                if 'better_x' in batch:
                    better_x = batch['better_x'].to(device)
                    loss_better = diffusion.p_losses(better_x, t, cond)
                    margin_loss = torch.clamp(args.margin - (loss_ours - loss_better), min=0.0)
                    margin_loss = margin_loss * (1.0 - is_chosen)
                    margin_loss = margin_loss.mean()
                else:
                    margin_loss = torch.tensor(0.0, device=device)
                
                val_loss += loss_ours.item()
                val_margin_loss += margin_loss.item()
                val_wins += is_chosen.sum().item()
                val_total += is_chosen.size(0)
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_margin = val_margin_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_win_rate = val_wins / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.6f} | Train Margin: {avg_margin_loss:.6f} | Train WR: {win_rate:.3f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val Margin: {avg_val_margin:.6f} | Val WR: {val_win_rate:.3f}")
        
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = save_dir / f"diffusion_{args.target_model}_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with val loss {best_val_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"diffusion_{args.target_model}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    save_path = save_dir / f"diffusion_{args.target_model}_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete for {args.target_model}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/model_comparisons.jsonl")
    parser.add_argument("--pretrained_model", type=str, default="models/diffusion_rlhf_best.pt")
    parser.add_argument("--target_model", type=str, default="rlhf", 
                       choices=['base', 'rlhf', 'rl_classifier'],
                       help="Which model to train from the 3-way comparison")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.1, help="Target margin for ranking loss")
    parser.add_argument("--margin_weight", type=float, default=1.0, help="Weight for margin loss")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_multi_model_rlhf(args)
