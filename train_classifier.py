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

from classifier import PathClassifier
from modules import UNet1D
from diffusion import Diffusion

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ClassifierDataset(Dataset):
    """Dataset that pairs human traces with AI-generated traces"""
    def __init__(self, human_data_path, diffusion_model_path, seq_len=32, device='cpu', num_samples=None):
        self.seq_len = seq_len
        self.device = device
        self.samples = []
        
        human_path = Path(human_data_path)
        if not human_path.exists():
            print(f"Warning: {human_data_path} not found.")
            return
            
        human_traces = []
        with open(human_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    human_traces.append(obj)
                except Exception as e:
                    print(f"Error parsing line: {e}")
        
        if num_samples:
            human_traces = human_traces[:num_samples]
        
        print(f"Loaded {len(human_traces)} human traces")
        
        print("Loading diffusion model for AI trace generation...")
        model = UNet1D(seq_len=seq_len, in_channels=2, cond_channels=2, base_channels=64)
        model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
        model.eval()
        diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
        
        print("Generating AI traces for paired dataset...")
        for human_trace in tqdm(human_traces, desc="Creating paired dataset"):
            human_path = self._process_human_trace(human_trace)
            
            target = torch.from_numpy(human_path[:, -1]).unsqueeze(0).to(device)  # [1, 2]
            with torch.no_grad():
                ai_path = diffusion.sample_ddim(target, (1, 2, seq_len), steps=50)
                ai_path = ai_path[0].cpu().numpy()  # [2, L]
            
            self.samples.append({
                'path': human_path,
                'label': 1  # human
            })
            self.samples.append({
                'path': ai_path,
                'label': 0  # AI
            })
        
        print(f"Created dataset with {len(self.samples)} samples ({len(self.samples)//2} human, {len(self.samples)//2} AI)")
    
    def _process_human_trace(self, obj):
        """Convert human trace to normalized tensor"""
        raw_path = obj['path']
        path_points = np.array([[p['x'], p['y']] for p in raw_path], dtype=np.float32)
        
        start_point = path_points[0]
        end_point = path_points[-1]
        target = end_point - start_point
        
        path_points = path_points - start_point
        
        resampled_path = self._resample(path_points, self.seq_len)
        resampled_path[0] = np.array([0.0, 0.0], dtype=np.float32)
        resampled_path[-1] = target.astype(np.float32)
        
        return resampled_path.transpose(1, 0).astype(np.float32)  # [2, N]
    
    def _resample(self, points, target_len):
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
        return {
            'path': torch.from_numpy(sample['path']),
            'label': torch.tensor(sample['label'], dtype=torch.float32)
        }

def train_classifier(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    full_dataset = ClassifierDataset(
        args.human_data_path,
        args.diffusion_model_path,
        seq_len=args.seq_len,
        device=device,
        num_samples=args.num_samples
    )
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    classifier = PathClassifier(seq_len=args.seq_len, base_channels=64).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            paths = batch['path'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = classifier(paths)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            # Calculate accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()
            
            pbar.set_postfix(
                loss=loss.item(),
                acc=train_correct/train_total,
                lr=optimizer.param_groups[0]['lr']
            )
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                paths = batch['path'].to(device)
                labels = batch['label'].to(device)
                
                logits = classifier(paths)
                loss = criterion(logits, labels)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), save_dir / "classifier_best.pt")
            print(f"New best classifier saved with val accuracy {best_val_acc:.4f}")
        
        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"classifier_epoch_{epoch+1}.pt"
            torch.save(classifier.state_dict(), save_path)
    
    torch.save(classifier.state_dict(), save_dir / "classifier_final.pt")
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_data_path", type=str, default="data/human_traces.jsonl")
    parser.add_argument("--diffusion_model_path", type=str, default="models/diffusion_best.pt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    train_classifier(args)
