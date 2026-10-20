"""
WikiText-2 Byte-Level Dataset and DataLoader
Prepares contiguous sequences of length block_size=128 from train.txt, valid.txt, test.txt.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WikiText2ByteDataset(Dataset):
    def __init__(self, file_path, block_size=128, max_samples=None):
        self.block_size = block_size
        text = Path(file_path).read_text(encoding='utf-8')
        raw_bytes = text.encode('utf-8')
        self.data = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # Total number of full non-overlapping sequences
        self.num_sequences = (len(self.data) - 1) // block_size
        if max_samples is not None:
            self.num_sequences = min(self.num_sequences, max_samples)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def get_dataloaders(data_dir='data/wikitext-2', block_size=128, batch_size=32, train_samples=10000, eval_samples=1500):
    data_dir = Path(data_dir)
    train_ds = WikiText2ByteDataset(data_dir / 'train.txt', block_size=block_size, max_samples=train_samples)
    val_ds = WikiText2ByteDataset(data_dir / 'valid.txt', block_size=block_size, max_samples=eval_samples)
    test_ds = WikiText2ByteDataset(data_dir / 'test.txt', block_size=block_size, max_samples=eval_samples)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
