
import torch
import torch.nn as nn
from similarity_refiner import SimilarityRefiner

def debug_refiner_effect():
    print("--- Debugging Refiner Effect ---")
    
    # 1. Setup
    alpha = 0.5 # As hardcoded in segmentor
    temperature = 0.1 # As hardcoded in segmentor
    refiner = SimilarityRefiner(alpha=alpha, temperature=temperature)
    
    # 2. Test Case: 2 pixels, 2 classes, 2 feature dims
    # Features are orthogonal -> S should be eye (if temp is small enough relative to distance) in ideal case?
    # No, S = softmax(F @ F.T / temp).
    # If F = [[1, 0], [0, 1]], F@F.T = [[1, 0], [0, 1]].
    # S = softmax([[10, 0], [0, 10]]) approx [[1, 0], [0, 1]].
    # Then smoothed_logits = S @ logits = logits.
    # So refined_logits = logits + alpha * logits = (1+alpha) * logits.
    # This IS a change (scaling), but relative order (argmax) might not change.
    
    # Let's try features that are similar.
    # F = [[1, 0], [0.7, 0.7]] (normalized)
    f1 = torch.tensor([1.0, 0.0])
    f2 = torch.tensor([0.7071, 0.7071])
    features = torch.stack([f1, f2]).unsqueeze(0) # [1, 2, 2]
    
    # Logits: diverse
    # L = [[10, -10], [-10, 10]]
    # Class 0 strong at pix 0, Class 1 strong at pix 1
    l1 = torch.tensor([10.0, -10.0])
    l2 = torch.tensor([-10.0, 10.0])
    logits = torch.stack([l1, l2]).unsqueeze(0) # [1, 2, 2]
    
    print(f"Features: \n{features}")
    print(f"Logits: \n{logits}")
    
    refined = refiner(logits, features)
    
    print(f"Refined Logits: \n{refined}")
    
    # Check if argmax changed?
    pred_orig = logits.argmax(dim=-1)
    pred_ref = refined.argmax(dim=-1)
    print(f"Original Prediction: {pred_orig}")
    print(f"Refined Prediction: {pred_ref}")
    
    diff = (refined - logits).abs().max()
    print(f"Max difference: {diff}")
    
    if diff == 0:
        print("FAIL: No change.")
    else:
        print("PASS: Values changed.")


if __name__ == "__main__":
    debug_refiner_effect()
