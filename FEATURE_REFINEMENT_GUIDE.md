# Feature Refinement Modules

Two complementary approaches for improving semantic segmentation:

## 1. Outlier Suppression Module
**Purpose**: Prevent outliers from polluting global features

**How it works**:
- Applied **between transformer blocks** (like SFP/SC-CLIP papers)
- Detects outlier tokens via attention analysis (cls_attn vs self_attn ratio)
- Suppresses via:
  - **Dampen mode**: Reduces outlier magnitude by factor (fast, lightweight)
  - **Replace mode**: Replaces with 4-neighbor mean (more aggressive)

**Configuration**:
```python
outlier_suppression_cfg=dict(
    suppression_mode='dampen',      # 'dampen' or 'replace'
    detection_threshold=1.5,        # Lower = more aggressive
    dampen_factor=0.5,              # Suppression strength (0-1)
    top_k=10,                       # Max outliers per layer
    suppression_layers=[-2]         # Apply at 2nd-to-last layer (SFP default)
)
```

**Status**: ✅ Fully integrated between transformer blocks

## 2. Feature Enhancement Module
**Purpose**: Enhance semantic coherence to dilute outlier influence

**How it works**:
- **Aggregate mode**: Blends features with spatial neighbors for consistency
- **Sharpen mode**: Enhances discriminative features via magnitude scaling

**Configuration**:
```python
feature_enhancement_cfg=dict(
    enhancement_mode='aggregate',  # 'aggregate' or 'sharpen'
    neighbor_weight=0.3,           # Spatial blending weight (0-1)
    sharpen_temperature=2.0        # Sharpening strength (>1)
)
```

**Status**: ✅ Ready to use

## Usage in Config

```python
# configs/Vaihingen.py
model = dict(
    name_path='./configs/cls_vaihingen.txt',
    prob_thd=0.1,
    bg_idx=5,
    cls_token_lambda=0.0,
    apply_ctd=True,
    
    # Enable feature refinement
    apply_outlier_suppression=True,  # ✅ Works between transformer blocks
    outlier_suppression_cfg=dict(
        suppression_mode='dampen',
        detection_threshold=1.5,
        dampen_factor=0.5,
        top_k=10,
        suppression_layers=[-2]  # Apply at 2nd-to-last layer (like SFP)
    ),
    
    apply_feature_enhancement=True,  # ✅ Works after feature extraction
    feature_enhancement_cfg=dict(
        enhancement_mode='aggregate',
        neighbor_weight=0.3,
    ),
)
```

## Testing Strategy

### Step 1: Test Outlier Suppression Only
Start with dampen mode at second-to-last layer:

```python
apply_outlier_suppression=True
outlier_suppression_cfg=dict(
    suppression_mode='dampen',
    suppression_layers=[-2]  # SFP default
)
apply_feature_enhancement=False
```

**Expected**: Reduced outlier influence on global features

### Step 2: Test Feature Enhancement Only
Start with spatial aggregation to improve consistency:

```python
apply_outlier_suppression=False
apply_feature_enhancement=True
feature_enhancement_cfg=dict(
    enhancement_mode='aggregate',
    neighbor_weight=0.3,  # Try 0.2, 0.3, 0.5
)
```

**Expected**: Smoother predictions, potentially higher mIoU if outliers were causing noise

### Step 3: Try Different Enhancement Modes
Test sharpening for more discriminative features:

```python
feature_enhancement_cfg=dict(
    enhancement_mode='sharpen',
    sharpen_temperature=2.0,  # Try 1.5, 2.0, 2.5
)
```

**Expected**: Sharper boundaries, potentially better class separation

### Step 4: Combine Both Modules
Test synergy between suppression and enhancement:

```python
apply_outlier_suppression=True
apply_feature_enhancement=True
```

**Expected**: Suppression prevents pollution, enhancement dilutes remaining influence

### Step 5: Add CTD for Full Pipeline
Test all three approaches together:

```python
apply_ctd=True
apply_outlier_suppression=True
apply_feature_enhancement=True
```

**Expected**: CTD handles global debiasing, suppression prevents outliers, enhancement ensures consistency

## Implementation Details

### Feature Processing Flow
```
Transformer blocks
  ├─ Block 0
  ├─ Block 1
  ├─ ...
  ├─ Block N-2  ← Outlier suppression applied here (default)
  ├─ Block N-1
  └─ Output
       ↓
extract_cls_token()
       ↓
apply_feature_enhancement()  ← Enhancement after extraction
       ↓
apply_ctd() (if enabled)
       ↓
featup()
       ↓
compute_logits()
```

### Module Architecture
- **Transformer with Suppression**: Outlier suppression applied between specified layers
- **OutlierSuppressionModule**: Attention-based outlier detection & suppression
- **FeatureEnhancementModule**: Spatial or magnitude-based enhancement after extraction

## Future Work

### Potential Improvements
1. **Learnable Enhancement**: Replace fixed kernels with learned conv layers
2. **Adaptive Thresholds**: Learn detection threshold per layer
3. **Multi-scale**: Apply refinement at multiple spatial scales
4. **Attention-guided**: Use attention maps to weight neighbor contributions

## Debugging

Check if modules are active:
```bash
./eval.ps1
# Look for output:
# Feature Refinement Pipeline:
#   → Feature Enhancement: mode=aggregate, neighbor_weight=0.3
```

Compare performance:
- Baseline (CTD only): `apply_feature_enhancement=False`
- With Enhancement: `apply_feature_enhancement=True`
- Measure mIoU difference
