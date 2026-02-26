import torch
import torch.nn.functional as F

def analyze_temperature(dim=768, num_samples=100, temperatures=[0.05, 0.07, 0.1, 0.15, 0.2, 0.3]):
    print(f"Analyzing Temperature Effect for Feature Dimension: {dim}, Samples: {num_samples}")
    print("Testing three scenarios of 'True Neighbor' similarity:\n1. High Sim (~0.9)\n2. Medium Sim (~0.7)\n3. Low Sim (~0.5)")
    
    # 3 sets of target similarities
    target_sims = [0.9, 0.7, 0.5]
    
    # Header
    print("-" * 120)
    print(f"{'True Sim' :<12} | {'Temp (T)':<10} | {'Self Attn':<12} | {'Neighbor':<12} | {'Noise (All)':<12} | {'Ratio (S/N)':<12} | {'Eff. Neigh?':<12}")
    print("-" * 120)
    
    for target in target_sims:
        # Generate data
        half = num_samples // 2
        base = F.normalize(torch.randn(half, dim), dim=1)
        noise = F.normalize(torch.randn(half, dim), dim=1)
        
        # Mix to get desired similarity
        # If we just do mix * base + noise_scale * noise, need to ensure norm is 1
        # Let's simplify: v = a*u + b*w. <v, u> = a. ||v||^2 = a^2 + b^2 = 1 => b = sqrt(1-a^2)
        a = target
        b = (1 - a**2)**0.5
        neighbor = a * base + b * noise
        neighbor = F.normalize(neighbor, dim=1)
        
        features = torch.cat([base, neighbor], dim=0)
        sim = features @ features.T
        
        # True neighbor indices
        idx = torch.arange(half)
        # Pair (i, i+half)
        
        # Background noise mask
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask.fill_diagonal_(False)
        mask[idx, idx+half] = False
        mask[idx+half, idx] = False
        
        max_noise = sim[mask].max().item()
        
        # Loop Temps
        for T in temperatures:
            logits = sim / T
            probs = F.softmax(logits, dim=-1)
            
            # Metrics
            self_attn = probs.diagonal().mean().item()
            
            # Neighbor attn
            p1 = probs[idx, idx+half]
            p2 = probs[idx+half, idx]
            neigh_attn = torch.cat([p1, p2]).mean().item()
            
            # Noise
            bg_noise = 1.0 - self_attn - neigh_attn
            
            ratio = self_attn / (neigh_attn + 1e-9)
            
            effective = "YES" if (neigh_attn > 0.1 and bg_noise < 0.1) else "NO"
            if bg_noise > 0.5: effective = "NOISE!"
            if self_attn > 0.95: effective = "IDENTITY"

            print(f"{target:<12.2f} | {T:<10} | {self_attn:<12.4f} | {neigh_attn:<12.4f} | {bg_noise:<12.4f} | {ratio:<12.2f} | {effective}")
        
        print("-" * 120)

if __name__ == "__main__":
    torch.manual_seed(42)
    analyze_temperature()
