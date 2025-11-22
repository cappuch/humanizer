import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import numpy as np

from dataset import HumanTraceDataset
from modules import UNet1D
from diffusion import Diffusion

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    full_dataset = HumanTraceDataset(args.data_path, seq_len=args.seq_len)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = UNet1D(seq_len=args.seq_len, in_channels=2, cond_channels=2, base_channels=64)
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        train_loss = 0
        
        for batch in pbar:
            x = batch['x'].to(device) # [B, 2, L]
            cond = batch['cond'].to(device) # [B, 2]
            
            optimizer.zero_grad()
            
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            
            loss = diffusion.p_losses(x, t, cond)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                cond = batch['cond'].to(device)
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
                loss = diffusion.p_losses(x, t, cond)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "diffusion_best.pt")
            print(f"New best model saved with val loss {best_val_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"diffusion_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), save_dir / "diffusion_final.pt")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/human_traces.jsonl")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
