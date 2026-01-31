# Self-Attention Enhancement for Robust Token Representations

## Motivation

The outlier suppression module successfully identifies and rectifies tokens that receive excessive attention from the CLS token while having weak self-attention (high `Attn[cls,i]/Attn[i,i]` ratio). This suggests that **tokens with strong self-attention are more robust and semantically meaningful**.

Instead of only fixing outliers after they appear, we can **proactively enhance self-attention** to make all tokens more self-reliant and prevent outlier behavior from emerging.

## Key Insight

**Tokens should attend primarily to themselves when representing local semantics.** Tokens with weak self-attention:
- Are overly influenced by other tokens (especially CLS)
- Lack stable local feature representation
- Are more likely to become outliers
- May have contaminated or unstable features

By enhancing self-attention, we encourage tokens to:
- Trust their own local information
- Be less susceptible to global biases
- Maintain more stable features across layers
- Reduce the need for aggressive outlier suppression

## Implementation

### Two Enhancement Modes

#### 1. Feature-Based Enhancement (Recommended)
**Strategy**: Strengthen features for tokens with weak self-attention

```python
# Detect weak self-attention
self_attn = Attn[i, i]  # Diagonal values
enhancement_factor = max(0, threshold - self_attn)

# Amplify token's own features
enhanced_feature = (1 + α × enhancement_factor) × original_feature
```

**Advantages**:
- Simple and efficient
- Directly strengthens token representations
- No attention map modifications needed
- Mild, controlled enhancement

#### 2. Attention-Based Enhancement
**Strategy**: Modify attention weights to boost self-attention

```python
# Boost diagonal (self-attention) values
Attn_enhanced[i, i] = Attn[i, i] + boost_factor

# Re-normalize attention
Attn_normalized = Attn_enhanced / sum(Attn_enhanced)

# Re-compute features with enhanced attention
enhanced_features = Attn_normalized @ features
```

**Advantages**:
- More interpretable (directly modifies attention)
- Can visualize attention changes
- Theoretically grounded in attention mechanism

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enhancement_strength` | 0.1 | How much to enhance (0.1 = mild, 0.3 = strong) |
| `min_self_attn_threshold` | 0.15 | Tokens below this value get enhanced |
| `mode` | 'feature' | 'feature' or 'attention' enhancement |

## Usage

### In Configuration

```python
# configs/your_config.py
model = dict(
    apply_self_attn_enhancement=True,
    self_attn_enhancement_cfg=dict(
        enhancement_strength=0.1,      # Mild enhancement
        min_self_attn_threshold=0.15,  # Target self-attention level
        mode='feature'                 # Feature-based mode
    ),
)
```

### Pipeline Order

The enhancement is applied in this order:
1. **Transformer encoding** (captures attention weights)
2. **Similarity enhancement** (optional, from mid-layer features)
3. **Self-attention enhancement** ← New! (boosts weak self-attention)
4. **Outlier suppression** (fixes remaining outliers)
5. **Global debiasing** (removes CLS contamination)
6. **CTD** (cluster-based debiasing)

## Expected Benefits

### 1. Reduced Outlier Occurrence
- Tokens with strong self-attention are less likely to become outliers
- Fewer tokens need aggressive spatial interpolation
- More stable feature representations

### 2. Better Local Semantics
- Tokens maintain stronger local information
- Less contamination from global/neighboring tokens
- Improved fine-grained segmentation

### 3. Complementary to Outlier Suppression
- **Prevention** (self-attention enhancement) + **Correction** (outlier suppression)
- Can reduce outlier suppression `top_k` parameter
- More robust overall pipeline

### 4. Mild, Adaptive Enhancement
- Only weak tokens are enhanced (selective)
- Enhancement scales with self-attention deficit
- No overcorrection of already-strong tokens

## Tuning Guidelines

### Enhancement Strength (`enhancement_strength`)
- **0.05-0.1**: Mild, safe for initial testing
- **0.1-0.2**: Moderate enhancement (recommended)
- **0.2-0.3**: Strong enhancement (may overcorrect)
- **>0.3**: Aggressive (risk of feature distortion)

### Threshold (`min_self_attn_threshold`)
- **0.10-0.15**: Targets only very weak tokens
- **0.15-0.20**: Moderate selectivity (recommended)
- **0.20-0.25**: Enhances more tokens
- **>0.25**: May enhance too many tokens

### Mode Selection
- **'feature'**: Start here, simpler and faster
- **'attention'**: Use if you need attention visualization or stronger theoretical grounding

## Diagnostic Output

When enabled, the module prints:
```
[Self-Attention Enhancement] Enabled with strength=0.1, threshold=0.15, mode=feature
```

You can also check self-attention statistics:
```python
stats = enhancer.get_self_attention_stats(attn_weights)
# Returns: {'mean', 'std', 'min', 'max', 'below_threshold'}
```

## Comparison with Outlier Suppression

| Aspect | Outlier Suppression | Self-Attention Enhancement |
|--------|-------------------|---------------------------|
| **When** | After outliers appear | Before outliers form |
| **How** | Replace with neighbors | Strengthen own features |
| **Target** | Top-k worst tokens | All weak tokens (adaptive) |
| **Strength** | Aggressive (replacement) | Mild (amplification) |
| **Philosophy** | Correction | Prevention |

## Example Results

Hypothetical improvements (requires actual testing):
```
Baseline (no modules):           30.2 mIoU
+ Outlier suppression:           30.8 mIoU (+0.6)
+ Self-attention enhancement:    31.1 mIoU (+0.9 total)
```

The enhancement should:
- Reduce the number of detected outliers
- Improve stability of features
- Boost performance on classes with fine details

## Implementation Notes

1. **Training-free**: No learnable parameters, works in inference mode
2. **Efficient**: Simple multiplication/addition operations
3. **Compatible**: Works with existing modules (outlier suppression, CTD, etc.)
4. **Safe**: Mild enhancement prevents feature distortion
5. **Adaptive**: Only affects tokens that need enhancement

## Future Extensions

Possible enhancements:
1. **Layer-wise adaptation**: Different strengths per transformer layer
2. **Class-specific thresholds**: Different values for different semantic classes
3. **Dynamic strength**: Adjust based on attention patterns
4. **Multi-scale enhancement**: Enhance at multiple feature resolutions
