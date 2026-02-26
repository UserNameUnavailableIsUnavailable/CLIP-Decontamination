# Similarity Enhancement Update

The **Similarity Enhancement Module** has been upgraded from an internal attention-modification mechanism to a **Post-Processing Logit Refinement** module.

## Why this change?
- **Stability**: Modifying internal transformer attention weights can destabilize the pre-trained CLIP features.
- **Performance**: Post-processing refinement (using `alpha * S @ Logits`) is a proven technique in Open-Vocabulary Segmentation (e.g., SC-CLIP, S-CLIP Mask Refinement).
- **Efficiency**: No need to cache intermediate layers or hook into the backward pass.

## New Implementation
- **File**: `similarity_refiner.py`
- **Class**: `SimilarityRefiner`
- **Algorithm**:
  1. Compute affinity $A = F \cdot F^T$ from normalized final visual features.
  2. Compute transition matrix $S = \text{Softmax}(A / \tau)$.
  3. Refine logits: $L_{new} = L_{old} + \alpha (S \cdot L_{old})$.
- **Parameters**:
  - `alpha`: 0.5 (Controls strength of smoothing)
  - `temperature`: 0.1 (Controls sharpness of neighbors)

## Usage in Config
Set `apply_similarity_enhancement=True` in your config. The `SegEarth` segmentor will automatically apply the refinement step after computing the initial logits.
