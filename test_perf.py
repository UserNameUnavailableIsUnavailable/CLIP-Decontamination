
import torch
import torch.nn.functional as F
import time

def test_large_matrix_mult():
    print("Testing large matrix multiplication feasibility...")
    B = 1
    N = 224 * 224 # 50176
    D = 128 # Reduced dim for test
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("Skipping large matrix test on CPU")
        return

    try:
        feats = torch.randn(B, N, D, device=device, dtype=torch.float16)
        print(f"Allocated feats: {feats.shape}")
        
        t0 = time.time()
        # Affinity: [B, N, N] -> 50176^2 * 2 bytes = 5GB
        affinity = torch.bmm(feats, feats.transpose(1, 2))
        torch.cuda.synchronize()
        print(f"Affinity computed in {time.time() - t0:.4f}s")
        
        t0 = time.time()
        S = F.softmax(affinity, dim=-1)
        torch.cuda.synchronize()
        print(f"Softmax computed in {time.time() - t0:.4f}s")
        
    except RuntimeError as e:
        print(f"OOM or Error: {e}")

if __name__ == "__main__":
    test_large_matrix_mult()
