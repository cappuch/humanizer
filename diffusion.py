import torch
import torch.nn.functional as F
from tqdm import tqdm

class Diffusion:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, cond)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, cond, t_index):
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_ddim(self, cond, shape, steps=50, clamp_boundaries=True):
        b = shape[0]
        L = shape[-1]
        t_vals = torch.linspace(0, 1, L, device=self.device).view(1, 1, L)
        img = cond.unsqueeze(-1) * t_vals
        
        times = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long).flip(0).to(self.device)
        
        for i, t_step in enumerate(tqdm(times, desc='DDIM Sampling')):
            t = torch.full((b,), t_step, device=self.device, dtype=torch.long)
            
            noise_pred = self.model(img, t, cond)
            
            alpha_bar = self.alphas_cumprod[t_step]
            
            if i == len(times) - 1:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)
            else:
                prev_step = times[i + 1]
                alpha_bar_prev = self.alphas_cumprod[prev_step]
            
            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            if clamp_boundaries:
                img[:, :, 0] = 0.0
                img[:, :, -1] = cond
            
        return img
