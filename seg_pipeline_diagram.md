# SegEarth-OV segmentation pipeline (SegEarthSegmentation)

This diagram describes the end-to-end inference path implemented in `SegEarthSegmentation`.

> Notation
> - Input image tensor: `x ∈ R^{B×3×H×W}`
> - Patch size: `(P_h, P_w)`
> - Patch grid: `H_p = H / P_h`, `W_p = W / P_w` (after any padding/cropping used for inference)
> - Embedding dim: `C`
> - Number of text queries (including synonyms): `Q`
> - Number of classes (after merging synonyms): `K`

## High-level flow

```mermaid
flowchart TD
    A[Input image x\nB×3×H×W] -->|predict()| B{slide_crop > 0?}

    B -->|Yes| C[forward_slide()\nSliding window crops]
    C --> C1[for each crop:\n- optional pad to patch multiple\n- forward_feature(crop)]
    C1 --> C2[stitch logits by overlap-average\nB×Q×H×W]
    C2 --> D[resize to original image size\nB×Q×H0×W0]

    B -->|No| E[forward_feature(x)]
    E --> D

    D --> F[postprocess_result()]
    F --> F1[scale logits by logit_scale]
    F1 --> F2[softmax over queries Q\nprob: Q×H0×W0]
    F2 --> G{Q equals K?\n(are there synonyms?)}
    G -->|No| H[merge synonyms to classes\nmax over queries per class\nK×H0×W0]
    G -->|Yes| I[already class-aligned\nK×H0×W0]

    H --> J[argmax over classes\nmask: 1×H0×W0]
    I --> J
    J --> K[thresholding:\nif max prob < prob_thd -> bg_idx]
    K --> L[final semantic mask]
```

## forward_feature(): from image to per-pixel open-vocabulary logits

```mermaid
flowchart TD
    A[Crop image x\nB×3×Hc×Wc] --> B[CLIP-like image encoder\nencode_image / visual encoder]
    B --> C[Patch/token features\nB×N×C]

    C --> D{output CLS token?\n(cls_token_lambda != 0)}
    D -->|Yes| D1[Compute CLS logits\ncls_logits = cls_token · text\nB×Q]
    D -->|No| E
    D1 --> E

    E{feature_up?} -->|Yes| F[Reshape to patch grid\nB×C×Hp×Wp]
    F --> G[SimFeatUp upsampler\nB×C×Hc×Wc]
    G --> H[Flatten to pixels\nB×(Hc·Wc)×C]

    E -->|No| I[Keep patch tokens\nB×N×C]\n
    H --> J[L2-normalize image feats]
    I --> J

    J --> K[Cosine similarity\nlogits = image · text^T]
    K --> L[Per-location logits\nB×(Hloc·Wloc)×Q]

    L --> M[Reshape to map\nB×Q×Hloc×Wloc]
    M --> N[Interpolate to requested size\nB×Q×Hout×Wout]

    D1 --> O[Broadcast-add global CLS bias\nlogits += cls_lambda * cls_logits]
    O --> M
```

### Key details

- Text queries are built once in `__init__`:
  - Each class line can contain comma-separated synonyms.
  - Each query is expanded by ImageNet prompt templates, encoded by the text encoder, averaged, and normalized.
  - This creates `query_features ∈ R^{Q×C}`.

- Per-location classification uses cosine similarity:
  - After normalizing, each pixel/patch feature is compared with every query feature.

- If `feature_up=True`, the pipeline becomes per-pixel by upsampling the feature map before similarity.
  - Otherwise, logits are produced on the patch grid and then interpolated to the image size.

- If `cls_token_lambda != 0`, a global term derived from the CLS token is added to every location’s logits.
  - This is an explicit “global bias” injection mechanism in the current implementation.

## Shape cheat-sheet (typical ViT)

- Without feature_up:
  - `encode_image(x)` → `B×N×C` (N = number of patch tokens)
  - similarity → `B×N×Q`
  - reshape to `B×Q×Hp×Wp`
  - interpolate to `B×Q×H×W`

- With feature_up:
  - reshape to `B×C×Hp×Wp`
  - upsample to `B×C×H×W`
  - flatten to `B×(H·W)×C`
  - similarity → `B×(H·W)×Q`
  - reshape to `B×Q×H×W`

## How to render

- GitHub renders Mermaid diagrams directly in Markdown.
- In VS Code, install a Mermaid preview extension or use the built-in Markdown preview (Mermaid support depends on your setup).
