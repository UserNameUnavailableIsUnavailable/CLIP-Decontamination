# Outlier Suppression Module

## Overview

The **Outlier Suppression Module** is a specialized component designed to mitigate the impact of "outlier tokens" in Vision Transformers (ViT), specifically within the CLIP vision encoder. Recent studies (e.g., *Vision Transformers Need Registers*) have shown that ViTs tend to generate high-norm artifact tokens in background areas that hoard attention but contain little semantic information. These outliers can distort segmentation maps and degrade feature quality.

This module dynamically detects these artifacts based on their attention signatures and suppresses them using a dual-action correction mechanism: **Replacement** and **Decontamination**.

## Methodology

### 1. Outlier Detection

The module identifies outliers by analyzing the attention matrix $A$ from the transformer layers. An outlier is characterized as a token that receives significant attention from the `[CLS]` token (indicating global relevance aggregation) but exhibits very low self-attention (indicating a lack of local distinctiveness or "sink" behavior).

For each patch token $i$, we compute an **Outlier Score** $S_i$:

$$ S_i = \frac{A_{\text{cls}, i}}{A_{i, i} + \epsilon} $$

The top-$k$ tokens with the highest scores are flagged as outliers set $\mathcal{O}$.

### 2. Dual-Stage Suppression

Once identified, outlier tokens are processed to neutralize their negative impact:

#### A. Weighted Spatial Replacement
The feature vector of the outlier token $X_{\text{outlier}}$ is replaced by a weighted average of its spatial neighbors. To minimize the influence of neighbors that might already be similar to the outlier (potentially contaminated), we assign higher weights to neighbors that are *dissimilar* to the outlier.

$$ w_{ij} = 1 - \text{sim}(X_i, X_j) $$
$$ \alpha_{ij} = \frac{\exp(w_{ij})}{\sum_{k \in N(i)} \exp(w_{ik})} $$
$$ X'_i = \sum_{j \in N(i)} \alpha_{ij} X_j $$

Where $\text{sim}(\cdot)$ is Cosine Similarity.

#### B. Neighbor Decontamination
Since outliers in self-attention layers propagate their information to neighbors, surrounding tokens may already be "contaminated" by the outlier's signal. The module performs a subtraction update to clean the neighbors:

$$ \sigma_{ij} = \lambda \cdot \text{sim}(X_j, X_i) $$
$$ X'_j = X_j - \sigma_{ij} X_i $$

Where:
*   $X_i$ is the original outlier feature.
*   $\lambda$ is the `contamination_temp`, controlling the suppression strength.

## Implementation

### Configuration

The module is integrated directly into the forward pass of the CLIP Vision Transformer. Key hyperparameters include:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `top_k` | 10 | The number of outlier tokens to suppress per image. |
| `consistency_temp` | 0.1 | Controls the aggressiveness of neighbor decontamination. |

### Integration Point

The module operates online during inference. It is injected into the transformer blocks (or specifically after the final encoder layer, depending on configuration) to clean features before they are projected for segmentation.

```python
# Pseudo-code usage
suppressor = OutlierSuppressionModule(top_k=10)
# Inside ViT forward pass:
x = self.transformer(x)
x = suppressor(x, attn_weights) 
```

## Benefits

1.  **Reduced Artifacts**: Eliminates high-norm "hotspots" in feature maps that can be mistaken for salient objects.
2.  **Smoother Boundaries**: By replacing singular artifacts with neighborhood averages, local feature continuity is improved.
3.  **Global Robustness**: Preventing the `[CLS]` token from attending to artifacts improves the global image representation used for classification and retrieval.
