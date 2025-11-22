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
from classifier import PathClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_rl_classifier(args):
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
    
    pretrained_path = Path(args.pretrained_model)
    if pretrained_path.exists():
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}. Training from scratch.")
    
    diffusion = Diffusion(model, timesteps=50, beta_end=0.2, device=device)
    
    classifier = PathClassifier(seq_len=args.seq_len, base_channels=64).to(device)
    classifier_path = Path(args.classifier_path)
    if classifier_path.exists():
        print(f"Loading classifier from {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
    else:
        raise ValueError(f"Classifier not found at {classifier_path}. Please train classifier first.")
    
    for param in classifier.parameters():
        param.requires_grad = False
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_reward = -float('inf')

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [RL-Classifier]")
        train_loss = 0
        train_reward = 0
        train_human_prob = 0
        
        for batch in pbar:
            x = batch['x'].to(device)  # [B, 2, L] - real human traces
            cond = batch['cond'].to(device)  # [B, 2]
            
            optimizer.zero_grad()
            
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            diffusion_loss = diffusion.p_losses(x, t, cond)
            
            with torch.no_grad():
                generated = diffusion.sample_ddim(cond, (x.shape[0], 2, args.seq_len), steps=args.sample_steps)
            
            with torch.no_grad():
                human_prob = classifier.predict_prob(generated)
                reward = human_prob.mean()
            
            with torch.no_grad():
                real_human_prob = classifier.predict_prob(x).mean()

            reward_weight = args.reward_scale * (1.0 - reward.detach())
            
            loss = diffusion_loss * (1.0 + reward_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += diffusion_loss.item()
            train_reward += reward.item()
            train_human_prob += real_human_prob.item()
            
            pbar.set_postfix(
                diff_loss=diffusion_loss.item(),
                reward=reward.item(),
                real_prob=real_human_prob.item(),
                lr=optimizer.param_groups[0]['lr']
            )
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_reward = train_reward / len(train_loader)
        avg_human_prob = train_human_prob / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_reward = 0
        val_human_prob = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                cond = batch['cond'].to(device)
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
                
                diffusion_loss = diffusion.p_losses(x, t, cond)
                
                generated = diffusion.sample_ddim(cond, (x.shape[0], 2, args.seq_len), steps=args.sample_steps)
                
                human_prob = classifier.predict_prob(generated)
                reward = human_prob.mean()
                
                real_human_prob = classifier.predict_prob(x).mean()
                
                val_loss += diffusion_loss.item()
                val_reward += reward.item()
                val_human_prob += real_human_prob.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_reward = val_reward / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_human_prob = val_human_prob / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.6f} | Train Reward: {avg_train_reward:.4f} | Real Human Prob: {avg_human_prob:.4f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val Reward: {avg_val_reward:.4f} | Val Real Prob: {avg_val_human_prob:.4f}")
        
        scheduler.step()

        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            torch.save(model.state_dict(), save_dir / "diffusion_rl_classifier_best.pt")
            print(f"New best RL model saved with val reward {best_val_reward:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"diffusion_rl_classifier_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), save_dir / "diffusion_rl_classifier_final.pt")
    print(f"RL training complete. Best val reward: {best_val_reward:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/human_traces.jsonl")
    parser.add_argument("--pretrained_model", type=str, default="models/diffusion_best.pt")
    parser.add_argument("--classifier_path", type=str, default="models/classifier_best.pt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Scale factor for reward weighting")
    parser.add_argument("--sample_steps", type=int, default=25, help="Number of steps for sampling during training")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_rl_classifier(args)
