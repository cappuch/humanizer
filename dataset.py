import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class HumanTraceDataset(Dataset):
    def __init__(self, data_path, seq_len=32):
        self.seq_len = seq_len
        self.samples = []
        
        path = Path(data_path)
        if not path.exists():
            print(f"Warning: {data_path} not found. Dataset will be empty.")
            return

        with open(path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    self.samples.append(self._process_sample(obj))
                except Exception as e:
                    print(f"Error parsing line: {e}")

    def _process_sample(self, obj):
        target = np.array([obj['target']['dx'], obj['target']['dy']], dtype=np.float32)
        
        raw_path = obj['path']

        path_points = np.array([[p['x'], p['y']] for p in raw_path], dtype=np.float32)
        
        resampled_path = self._resample(path_points, self.seq_len)
        
        resampled_path = resampled_path.transpose(1, 0).astype(np.float32)
        
        return {
            'path': resampled_path, # [2, N]
            'target': target.astype(np.float32)        # [2]
        }

    def _resample(self, points, target_len):
        # points: [T, 2]
        if len(points) < 2:
            return np.tile(points[0], (target_len, 1)) # handle edge case of a single point repeating
            
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
            'x': torch.from_numpy(sample['path']),
            'cond': torch.from_numpy(sample['target'])
        }
