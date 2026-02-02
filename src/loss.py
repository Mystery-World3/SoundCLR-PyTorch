import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        Menghitung NT-Xent loss
        z_i, z_j: Output proyeksi dari dua view augmentasi
        """
        batch_size = z_i.size(0)
        
        # 1. menggabungkan kedua view
        z = torch.cat((z_i, z_j), dim=0)
        
        # 2. menghitung Similarity Matrix antar semua pasangan
        z = nn.functional.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / self.temperature
        
        # 3. Masking dan Hitung Loss
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach() # Trik kestabilan numerik
        
        mask = ~torch.eye(2 * batch_size, device=sim.device).bool()
        # Positif: diagonal offset batch_size dan -batch_size
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        exp_sim = torch.exp(sim)
        denominator = exp_sim.masked_select(mask).view(2 * batch_size, -1).sum(dim=-1)
        
        loss = -positives + torch.log(denominator)
        
        return loss.mean()