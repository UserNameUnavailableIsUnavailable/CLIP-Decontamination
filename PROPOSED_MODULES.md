# Proposed Modules for Training-Free Open-Vocabulary Semantic Segmentation

This document provides a formal description of three proposed modules designed to enhance vision-language models for dense prediction tasks. All modules are training-free and operate on frozen CLIP features.

---

## Overview

Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ denote the patch token features from a Vision Transformer (ViT), where $N$ is the number of patches and $D$ is the feature dimension. Let $\mathbf{x}_\text{cls} \in \mathbb{R}^{D}$ denote the global CLS token. We propose three complementary modules:

1. **Outlier Suppression Module** — Detects and suppresses anomalous tokens
2. **Similarity-Enhanced Attention Module** — Enhances feature saliency via customized attention
3. **Global Debiasing Module** — Removes global bias through similarity-weighted CLS subtraction

---

## Theoretical Foundation: Understanding Attention Outliers and Global Bias

### Why Outliers Occur in Attention Maps

In vision transformers, the self-attention mechanism computes attention weights $A_{ij}$ between patch $i$ and patch $j$ as:

$$A_{ij} = \frac{\exp(Q_i \cdot K_j / \sqrt{d})}{\sum_{k=1}^{N} \exp(Q_i \cdot K_k / \sqrt{d})}$$

Outliers in attention maps emerge due to several factors:

1. **Semantic Inconsistency**: Patches containing noise, artifacts, or semantically irrelevant content produce query vectors $Q_i$ that are poorly aligned with the majority of key vectors $K_j$, resulting in abnormally low self-attention weights.

2. **Attention Collapse**: Certain dominant patches (often containing salient objects or high-contrast regions) attract disproportionately high attention from most other patches, creating attention outliers with weights $A_{ij} \gg \bar{A}$.

3. **Boundary Confusion**: Patches at object boundaries often exhibit unstable attention patterns due to mixed semantic content, leading to inconsistent attention distributions.

#### Why CLS Over-Attends to Outliers

A critical mechanism underlying outlier formation is that the CLS token specifically attends to patches that are **semantically inconsistent** with the global image content. This occurs due to a fundamental imbalance in query strengths:

Consider the attention computation for patch $i$:
- CLS-to-patch attention: $A_{\text{CLS}, i} \propto \exp(Q_{\text{CLS}} \cdot K_i / \sqrt{d})$
- Patch self-attention: $A_{i, i} \propto \exp(Q_i \cdot K_i / \sqrt{d})$

**Query Strength Imbalance**: When $Q_{\text{CLS}}$ is globally strong (well-optimized during CLIP pretraining) and $Q_i$ is locally weak (due to poor semantic content, noise, or boundary artifacts), we observe:

$$Q_{\text{CLS}} \cdot K_i \gg Q_i \cdot K_i$$

This imbalance causes patch $i$ to be **more attended by CLS than by itself**, creating the outlier signature:

$$r_i = \frac{A_{\text{CLS}, i}}{A_{i,i}} = \frac{\exp(Q_{\text{CLS}} \cdot K_i / \sqrt{d})}{\exp(Q_i \cdot K_i / \sqrt{d})} = \exp\left(\frac{(Q_{\text{CLS}} - Q_i) \cdot K_i}{\sqrt{d}}\right) \gg 1$$

This occurs because:

1. **Global Query Strength**: $Q_{\text{CLS}}$ is robustly trained to capture diverse global patterns, making it consistently "strong" across different image contexts.

2. **Local Query Weakness**: Patches containing noise, artifacts, or boundary confusion produce weak query vectors $Q_i$ that poorly represent their own semantic content.

3. **Distinctiveness Attraction**: Outlier patches with inconsistent semantics have key vectors that are geometrically distant from the cluster of consistent patches, making them appear "salient" to the global attention mechanism.

4. **Contrastive Learning Bias**: CLIP's contrastive pretraining encourages the model to identify distinctive visual elements that might be relevant for image-text matching, inadvertently causing CLS to focus on semantically irrelevant but visually distinctive patches.

This creates a fundamental mismatch: **CLS attends most to patches that are least representative of the global semantics**, resulting in a corrupted global representation that propagates noise throughout the network.

### Local Feature Contamination Mechanism

The self-attention output for patch $i$ is computed as:
$$Z_i = \sum_{j=1}^{N} A_{ij} V_j$$

#### CLS Over-Attention and Patch Contamination

A particularly problematic scenario occurs when the CLS token over-attends to certain patches that contain **weak global semantics**. When $A_{\text{CLS}, j} \gg \bar{A}_{\text{CLS}}$ for patches $j$ that are not semantically consistent with the global image content, these outlier patches become "privileged" despite their poor semantic quality:

$$Z_{\text{CLS}} = \sum_{j=1}^{N} A_{\text{CLS}, j} V_j \approx \sum_{j \in \mathcal{O}} A_{\text{CLS}, j} V_j^{\text{weak}}$$

where $\mathcal{O}$ is the set of outlier patches with weak global semantics, and $V_j^{\text{weak}}$ represents their low-quality semantic content. This creates contamination through several mechanisms:

1. **Global Noise Injection**: The CLS token, now heavily influenced by outlier patches with weak semantics, carries their noisy signatures as a corrupted "global template." This global noise propagates back to clean patches in subsequent layers.

2. **Attention Reciprocity**: Due to the symmetric nature of attention computation, outlier patches that receive high CLS attention often reciprocally attend strongly to the CLS token, creating a reinforcement loop where $A_{j, \text{CLS}} \propto A_{\text{CLS}, j}$. This amplifies the influence of weak semantic content.

3. **Transitive Contamination**: Clean patches that attend to the noise-contaminated CLS token inherit its corrupted global information:
   $$Z_i^{\text{next}} = A_{i, \text{CLS}} \cdot Z_{\text{CLS}}^{\text{noisy}} + \sum_{j \neq \text{CLS}} A_{ij} V_j$$

4. **Global Noise Leakage**: The contaminated CLS token effectively "leaks" global noise into local patch representations, causing patches to carry irrelevant semantic information that should be purely global (or absent entirely).

When outlier patches exist with abnormal attention weights $A_{ij}^{\text{outlier}}$, they contaminate local features in additional ways:

5. **Direct Contamination**: Outlier patches with extreme attention weights contribute disproportionately to $Z_i$, injecting noise into clean patch representations.

6. **Propagated Contamination**: Since attention is computed globally, contaminated features $Z_i^{\text{contaminated}}$ influence subsequent layers, amplifying the contamination effect through the network depth.

### Global Bias Formation

The contaminated local features aggregate to form a biased global representation, typically manifested in the CLS token:
$$Z_{\text{CLS}} = \sum_{j=1}^{N} A_{\text{CLS},j} V_j^{\text{contaminated}}$$

This global bias leads to a phenomenon where the model tends to recognize objects as whatever the dominant contaminated semantic represents, rather than the true local semantics. The bias becomes self-reinforcing because:

1. **Feedback Loop**: The biased global representation influences subsequent attention computations, further strengthening the dominant but incorrect semantic patterns.

2. **Feature Homogenization**: Local patch features become increasingly similar to the biased global representation, reducing the model's ability to distinguish fine-grained local semantics.

3. **Classification Drift**: The final classification relies heavily on the biased global features, causing systematic misclassification toward the dominant contaminated class.

### Module Design Rationale

The three proposed modules directly address these issues:
- **Outlier Suppression** removes the source of contamination by detecting and suppressing outlier patches
- **Similarity Enhancement** strengthens clean attention patterns while preserving semantic consistency
- **Global Debiasing** explicitly removes the accumulated global bias from local features

---

## Module 1: Outlier Suppression

### Motivation

Vision Transformers pretrained with image-level contrastive objectives (e.g., CLIP) exhibit a well-documented phenomenon where certain patch tokens become "outliers"—tokens that absorb disproportionate attention from the CLS token while exhibiting low self-attention. These outlier tokens often correspond to low-frequency background regions or boundary artifacts, and their presence contaminates neighboring features through the attention mechanism, degrading dense prediction performance.

### Outlier Detection

We propose an attention-based detection criterion that identifies outliers by examining the ratio between CLS-to-patch attention and self-attention. Given the attention weight matrix $\mathbf{A} \in \mathbb{R}^{(N+1) \times (N+1)}$ from the final transformer block (where index 0 corresponds to the CLS token), we define the outlier score for patch $i$ as:

$$
r_i = \frac{A_{\text{cls}, i}}{A_{i,i} + \epsilon}
$$

where $A_{\text{cls}, i}$ denotes the attention weight from the CLS token to patch $i$, $A_{i,i}$ denotes the self-attention of patch $i$, and $\epsilon$ is a small constant for numerical stability.

**Intuition:** Outlier tokens tend to have high $A_{\text{cls}, i}$ (strongly attended by CLS) but low $A_{i,i}$ (weak self-representation), resulting in elevated ratios. We select the top-$k$ patches with the highest $r_i$ values as outliers:

$$
\mathcal{O} = \text{TopK}\left(\{r_i\}_{i=1}^{N}, k\right)
$$

### Outlier Suppression via Neighbor Interpolation

For each detected outlier $i \in \mathcal{O}$, we replace its feature with a weighted average of its 8 spatial neighbors. Let $\mathcal{N}(i)$ denote the set of valid neighbors of patch $i$ in the 2D spatial grid. The replacement feature is computed as:

$$
\tilde{\mathbf{x}}_i = \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \mathbf{x}_j
$$

where the weights are derived from inverse cosine similarity to encourage contributions from dissimilar (less contaminated) neighbors:

$$
w_{ij} = \frac{\exp\left(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(1 - \cos(\mathbf{x}_i, \mathbf{x}_k)\right)}
$$

### Bidirectional Decontamination

To address feature contamination that has already propagated to neighbors, we additionally apply a mild decontamination step. For each neighbor $j \in \mathcal{N}(i)$ of outlier $i$:

$$
\tilde{\mathbf{x}}_j = \mathbf{x}_j - \tau \cdot \cos(\mathbf{x}_i, \mathbf{x}_j) \cdot \mathbf{x}_i
$$

where $\tau \in [0, 1]$ is a contamination temperature parameter controlling the removal strength.

---

## Module 2: Similarity-Enhanced Attention

### Motivation

Standard CLIP attention mechanisms compute Query-Key interactions that are optimized for image-level alignment, not dense spatial reasoning. To enhance feature saliency and local coherence for segmentation, we propose a customized attention mechanism that incorporates self-similarity priors from intermediate features.

### Gated Self-Similarity Attention

We introduce a novel attention formulation that gates Key-Key self-attention with Query-Query interactions:

$$
\mathbf{A}_\text{gated} = \sigma\left(\mathbf{Q}\mathbf{Q}^\top\right) \odot \left(\mathbf{K}\mathbf{K}^\top\right)
$$

where $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{N \times d}$ are the query and key projections (with $d = D / h$ for $h$ attention heads), $\sigma(\cdot)$ denotes the sigmoid function, and $\odot$ represents element-wise multiplication.

**Intuition:** The sigmoid-gated Query self-attention acts as a learned saliency mask that modulates the Key-based structural similarity. This allows the model to focus on semantically salient regions while preserving local spatial coherence.

### Self-Similarity Enhancement

To further enhance locality, we augment the attention weights with a self-similarity map $\mathbf{S}$ computed from mid-layer features. Let $\mathbf{X}^\text{mid} \in \mathbb{R}^{N \times D}$ denote the patch features from an intermediate transformer layer. We compute:

$$
S_{ij} = \frac{\cos\left(\mathbf{x}_i^\text{mid}, \mathbf{x}_j^\text{mid}\right)}{\tau_s}
$$

where $\tau_s$ is a temperature parameter. The enhanced attention (before softmax) becomes:

$$
\tilde{\mathbf{A}} = \mathbf{A}_\text{gated} + \lambda_s \cdot \mathbf{S}
$$

where $\lambda_s$ is the similarity weight. The final attention weights are obtained via:

$$
\mathbf{A}_\text{final} = \text{softmax}\left(\tilde{\mathbf{A}}\right)
$$

**Intuition:** Adding the self-similarity map reinforces connections between semantically similar patches, encouraging coherent attention patterns that respect semantic boundaries.

---

## Module 3: Global Debiasing

### Motivation

The CLS token in CLIP aggregates global image-level information, which introduces a bias toward dominant visual concepts. For dense prediction, this global bias can overwhelm local discriminative features, particularly for less prominent classes. We propose a mild, similarity-weighted debiasing strategy that adaptively removes CLS contamination.

### Similarity-Weighted CLS Subtraction

For each patch feature $\mathbf{x}_i$, we compute its cosine similarity with the CLS token:

$$
s_i = \cos(\mathbf{x}_i, \mathbf{x}_\text{cls}) = \frac{\mathbf{x}_i \cdot \mathbf{x}_\text{cls}}{\|\mathbf{x}_i\| \|\mathbf{x}_\text{cls}\|}
$$

The debiased feature is then computed as:

$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i - \alpha \cdot s_i \cdot \mathbf{x}_\text{cls}
$$

where $\alpha \in \mathbb{R}^+$ is the global debiasing factor (default $\alpha = 0.2$).

**Intuition:** Patches that are highly similar to the CLS token are more likely to be contaminated by global bias and thus receive stronger debiasing. Patches that are already distinctive (low similarity) are preserved. This adaptive scheme prevents over-correction while effectively reducing global bias.

### Design Rationale

Unlike uniform CLS subtraction, our similarity-weighted approach has several advantages:

1. **Adaptive strength:** Contaminated patches receive proportionally stronger correction
2. **Preservation of distinctive features:** Low-similarity patches retain their discriminative information
3. **Mild regularization:** The small factor $\alpha$ ensures stability and prevents feature collapse

---

## Configuration

All three modules can be independently enabled via configuration flags:

```python
model = dict(
    # Module 1: Outlier Suppression
    apply_outlier_suppression=True,
    outlier_suppression_cfg=dict(
        top_k=30,  # Number of outliers to suppress
    ),
    
    # Module 2: Similarity-Enhanced Attention
    model_type='Experimental',  # Enables gated QQ·KK attention
    apply_similarity_enhancement=True,
    similarity_enhancement_cfg=dict(
        similarity_weight=1.0,   # λ_s: weight for self-similarity
        temperature=1.0,         # τ_s: temperature for similarity
    ),
    
    # Module 3: Global Debiasing
    global_debias_factor=0.2,    # α: debiasing strength
)
```

---

## Summary

| Module | Configuration | Key Operation | Purpose |
|--------|---------------|---------------|---------|
| Outlier Suppression | `apply_outlier_suppression` | $r_i = A_{\text{cls},i} / A_{i,i}$ | Remove anomalous tokens |
| Similarity-Enhanced Attention | `model_type='Experimental'` + `apply_similarity_enhancement` | $\sigma(\mathbf{QQ}^\top) \odot \mathbf{KK}^\top + \lambda_s \mathbf{S}$ | Enhance saliency & locality |
| Global Debiasing | `global_debias_factor` | $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \alpha s_i \mathbf{x}_\text{cls}$ | Remove global bias |

All modules are **training-free** and designed to be **plug-and-play** for open-vocabulary semantic segmentation with frozen CLIP models.
