
import torch
import torch.nn.functional as F
from similarity_refiner import SimilarityRefiner

def test_similarity_refiner():
    print("--- Testing SimilarityRefiner ---")
    
    # 1. Instantiate SimilarityRefiner
    alpha = 0.5
    temperature = 0.1
    refiner = SimilarityRefiner(alpha=alpha, temperature=temperature)
    print(f"Refiner initialized with alpha={alpha}, temperature={temperature}")

    # 2. Create random logits and features
    B = 1
    C = 5  # Number of classes
    D = 16 # Feature dimension
    H = 32
    W = 32
    N = H * W
    
    # Random logits [B, C, H, W]
    logits_spatial = torch.randn(B, C, H, W)
    # Random features [B, D, H, W]
    features_spatial = torch.randn(B, D, H, W)
    
    # 3. Run the refiner (spatial)
    print("\n[Spatial Input Check]")
    refined_spatial = refiner(logits_spatial, features_spatial)
    
    # 4. Check if input == output
    diff_spatial = (refined_spatial - logits_spatial).abs().max().item()
    print(f"Max difference (Spatial): {diff_spatial}")
    if diff_spatial == 0:
        print("FAIL: Refiner output is identical to input (Spatial)!")
    else:
        print("PASS: Refiner output changed the values (Spatial).")
        
    # Check manual calculation to verify logic
    # Flatten
    logits_flat = logits_spatial.view(B, C, N).permute(0, 2, 1) # [B, N, C]
    features_flat = features_spatial.view(B, D, N).permute(0, 2, 1) # [B, N, D]
    
    feats_norm = F.normalize(features_flat, dim=-1)
    affinity = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
    S = F.softmax(affinity / temperature, dim=-1)
    smoothed = torch.bmm(S, logits_flat)
    expected = logits_flat + alpha * smoothed
    expected = expected.permute(0, 2, 1).view(B, C, H, W)
    
    calc_diff = (refined_spatial - expected).abs().max().item()
    print(f"Calculation verification diff: {calc_diff}")
    if calc_diff > 1e-5:
         print("FAIL: Verification calculation does not match!")

    # 5. Test with flat input (as used in segearth_segmentor.py around line 211)
    # segearth_segmentor.py uses flat features [B, N, D] and implicit flat logits
    # Actually segmentor computes logits = features @ classifier.T -> [B, N, C]
    
    print("\n[Flat Input Check - mimicking segearth_segmentor]")
    logits_flat_input = logits_flat
    features_flat_input = features_flat
    
    refined_flat = refiner(logits_flat_input, features_flat_input)
    
    diff_flat = (refined_flat - logits_flat_input).abs().max().item()
    print(f"Max difference (Flat): {diff_flat}")
    if diff_flat == 0:
        print("FAIL: Refiner output is identical to input (Flat)!")
    else:
        print("PASS: Refiner output changed the values (Flat).")

if __name__ == "__main__":
    test_similarity_refiner()
