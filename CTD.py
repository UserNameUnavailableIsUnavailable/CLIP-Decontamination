from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from sklearn.cluster import DBSCAN

import os
import torch


_DBSCAN_SKIP_WARNED: set[str] = set()


def _warn_dbscan_skip_once(key: str, message: str) -> None:
    # Enabled by default (warn once). Set DBSCAN_SKIP_WARN=0 to silence.
    if os.environ.get('DBSCAN_SKIP_WARN', '1') in ('0', 'false', 'False', 'no', 'NO'):
        return
    if key in _DBSCAN_SKIP_WARNED:
        return
    _DBSCAN_SKIP_WARNED.add(key)
    print(message)


@dataclass
class DBSCANCfg:
    eps: float = 1.1
    min_samples: int = 8
    metric: str = 'cosine'  # 'cosine' | 'euclidean'

    # If metric='euclidean' and use_spatial=True, we concatenate (feat, xy)
    # after normalization.
    use_spatial: bool = False
    spatial_weight: float = 0.25
    feat_weight: float = 1.0

    # Runtime controls
    run_on: str = 'cpu'  # 'cpu' | 'cuda'
    max_points: int = 4096

    # Optional refinement
    refine_tokens: bool = False

    # Optional CLS-based subtraction ("relocalize")
    # For each cluster k, compute prototype p_k = mean(tokens in cluster k).
    # Then compute s_k = cosine_similarity(p_k, cls_token).
    # For each token x in cluster k: x <- x - (cls_subtract_scale * s_k) * cls_token_unit
    # Noise points (label=-1) are left unchanged.
    cls_subtract: bool = False
    cls_subtract_scale: float = 1.0
    cls_subtract_use_unit_cls: bool = True


def _cfg_from_dict(cfg: Optional[Dict]) -> DBSCANCfg:
    if not cfg:
        return DBSCANCfg()
    base = DBSCANCfg()
    for k, v in cfg.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1.1) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _pairwise_distance(points: torch.Tensor, metric: str) -> torch.Tensor:
    """Returns NxN distance matrix."""
    if metric == 'cosine':
        p = _normalize(points.float(), dim=-1)
        sim = p @ p.t()
        dist = 1.0 - sim
        return dist
    if metric == 'euclidean':
        return torch.cdist(points.float(), points.float())
    elif metric == "cosine":
        p = _normalize(points.float(), dim=-1)
        sim = p @ p.t()
        dist = 1.0 - sim
        return dist
    raise ValueError(f"Unsupported metric: {metric}")


def dbscan(points: torch.Tensor, eps: float, min_samples: int, metric: str = 'cosine') -> torch.Tensor:
    """DBSCAN clustering using GPU-accelerated cuML if available, else sklearn.

    Args:
        points: (N, D)
        eps: neighborhood radius
        min_samples: minimum number of points (including itself) to be a core point
        metric: 'cosine' or 'euclidean'

    Returns:
        labels: (N,) with -1 for noise, otherwise 0..(n_clusters-1)
    """
    if points.dim() != 2:
        raise ValueError(f"points must be 2D (N,D), got {tuple(points.shape)}")
    n = points.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)

    # L2 normalize before clustering: w/||w||
    points_norm = points / (points.norm(dim=-1, keepdim=True) + 1e-8)

    # Try GPU-accelerated cuML first
    try:
        from cuml.cluster import DBSCAN as cuDBSCAN
        import cupy as cp
        
        # Convert to CuPy array (stays on GPU) with float64 for larger integer indexing support
        if points.is_cuda:
            points_cp = cp.asarray(points_norm, dtype=cp.float64)
        else:
            points_cp = cp.asarray(points_norm.cuda(), dtype=cp.float64)
        
        # cuML DBSCAN (GPU-accelerated)
        # max_mbytes_per_batch=None allows cuML to use more memory for larger batches
        # verbose=0 to suppress warnings about batch size
        db = cuDBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric),
                      max_mbytes_per_batch=None, verbose=0)
        labels_cp = db.fit_predict(points_cp)
        
        # Convert back to torch
        labels = torch.as_tensor(labels_cp, device=points.device, dtype=torch.long)
        
    except (ImportError, Exception):
        # Fallback to sklearn (CPU)
        print("[DBSCAN] cuML not available or failed, falling back to sklearn DBSCAN on CPU.")
        from sklearn.cluster import DBSCAN
        
        # Convert to numpy for sklearn
        points_np = points_norm.cpu().numpy()
        
        # sklearn DBSCAN with optimized parameters
        # Use 'auto' algorithm to automatically select the best algorithm for the metric
        # n_jobs=-1 for parallel processing
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric), 
                    algorithm='auto', n_jobs=-1)
        labels_np = db.fit_predict(points_np)
        
        # Convert back to torch
        labels = torch.from_numpy(labels_np).to(dtype=torch.long, device=points.device)
    
    return labels


def cluster_patch_tokens_dbscan(
    patch_tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    cfg_dict: Optional[Dict] = None,
    cls_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cluster patch tokens and optionally refine tokens by cluster mean.

    Args:
        patch_tokens: (B, N, C) patch embeddings
        grid_hw: (H_p, W_p) such that N == H_p*W_p
        cfg_dict: DBSCAN settings (see DBSCANCfg)

    Returns:
        refined_tokens: (B, N, C) (may be same as input if disabled)
        labels: (B, N) cluster labels, or None if clustering is skipped
    """
    cfg = _cfg_from_dict(cfg_dict)

    if patch_tokens.dim() != 3:
        _warn_dbscan_skip_once(
            'shape_not_3d',
            f"[DBSCAN] skip: patch_tokens must be (B,N,C), got {tuple(patch_tokens.shape)}",
        )
        return patch_tokens, None

    b, n, c = patch_tokens.shape
    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    if hp * wp != n:
        # shape mismatch: skip clustering safely
        _warn_dbscan_skip_once(
            'grid_mismatch',
            f"[DBSCAN] skip: grid_hw=({hp},{wp}) implies {hp * wp} tokens, but N={n}",
        )
        return patch_tokens, None

    if n > int(cfg.max_points):
        _warn_dbscan_skip_once(
            'too_many_points',
            f"[DBSCAN] skip: N={n} exceeds max_points={int(cfg.max_points)}",
        )
        return patch_tokens, None

    run_device = patch_tokens.device
    if cfg.run_on == 'cpu':
        run_device = torch.device('cpu')
    elif cfg.run_on == 'cuda':
        run_device = patch_tokens.device if patch_tokens.is_cuda else torch.device('cuda')

    # Precompute normalized xy coords in [0,1]
    if cfg.metric == 'euclidean' and cfg.use_spatial:
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, hp, device=run_device),
            torch.linspace(0.0, 1.0, wp, device=run_device),
            indexing='ij',
        )
        xy = torch.stack([xx, yy], dim=-1).reshape(n, 2)

    all_labels = []
    refined = patch_tokens

    if cfg.refine_tokens:
        refined = patch_tokens.clone()

    for bi in range(b):
        feats = patch_tokens[bi]
        feats_run = feats.detach().to(run_device)

        if cfg.metric == 'cosine':
            points = feats_run
        else:
            # euclidean: normalize features and optionally concatenate xy
            f = _normalize(feats_run, dim=-1)
            if cfg.use_spatial:
                points = torch.cat([float(cfg.feat_weight) * f, float(cfg.spatial_weight) * xy], dim=-1)
            else:
                points = float(cfg.feat_weight) * f

        labels = dbscan(points, eps=float(cfg.eps), min_samples=int(cfg.min_samples), metric=str(cfg.metric))
        all_labels.append(labels.to(patch_tokens.device))

        # Prepare a working copy to modify
        refined_b = feats_run
        if cfg.refine_tokens or cfg.cls_subtract:
            refined_b = feats_run.clone()

        if cfg.refine_tokens:
            # Replace token with its cluster mean (noise stays unchanged)
            labels_cpu = labels
            valid = labels_cpu >= 0
            if valid.any():
                cluster_ids = labels_cpu[valid]
                num_clusters = int(cluster_ids.max().item()) + 1

                # compute means via scatter add
                feat_valid = feats_run[valid].float()
                sums = torch.zeros((num_clusters, c), device=run_device, dtype=torch.float32)
                counts = torch.zeros((num_clusters,), device=run_device, dtype=torch.float32)
                sums.index_add_(0, cluster_ids, feat_valid)
                ones = torch.ones((cluster_ids.shape[0],), device=run_device, dtype=torch.float32)
                counts.index_add_(0, cluster_ids, ones)
                means = sums / counts.clamp_min(1.0).unsqueeze(1)

                # assign back
                refined_b[valid] = means[cluster_ids].to(refined_b.dtype)

        if cfg.cls_subtract and cls_token is not None:
            # CLS can be (C,) or (B,C)
            cls_b = cls_token
            if cls_b.dim() == 2:
                cls_b = cls_b[bi]
            cls_b_run = cls_b.detach().to(run_device)
            cls_b_f = cls_b_run.float()
            if cfg.cls_subtract_use_unit_cls:
                cls_vec = _normalize(cls_b_f, dim=-1)
            else:
                cls_vec = cls_b_f

            labels_cpu = labels
            valid = labels_cpu >= 0
            if valid.any():
                cluster_ids = labels_cpu[valid]
                num_clusters = int(cluster_ids.max().item()) + 1

                # prototypes from ORIGINAL feats (not refined) for stability
                feat_valid = feats_run[valid].float()
                sums = torch.zeros((num_clusters, c), device=run_device, dtype=torch.float32)
                counts = torch.zeros((num_clusters,), device=run_device, dtype=torch.float32)
                sums.index_add_(0, cluster_ids, feat_valid)
                ones = torch.ones((cluster_ids.shape[0],), device=run_device, dtype=torch.float32)
                counts.index_add_(0, cluster_ids, ones)
                protos = sums / counts.clamp_min(1.0).unsqueeze(1)

                proto_u = _normalize(protos, dim=-1)
                cls_u = _normalize(cls_b_f, dim=-1)
                sims = (proto_u * cls_u.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)  # (K,)

                # subtract per-token: x <- x - scale * sim_k * cls_vec
                scale = float(cfg.cls_subtract_scale)
                sub = (sims[cluster_ids].unsqueeze(1) * cls_vec.unsqueeze(0)) * scale
                refined_b[valid] = (refined_b[valid].float() - sub).to(refined_b.dtype)

        # write back if needed
        if cfg.refine_tokens or cfg.cls_subtract:
            refined[bi] = refined_b.to(patch_tokens.device)

    labels_b = torch.stack(all_labels, dim=0) if all_labels else None
    return refined, labels_b


def adaptive_debiasing(
    items: torch.Tensor,
    labels: Optional[torch.Tensor],
    bias: torch.Tensor,
    *,
    factor: float,
    eps: float = 1.1,
) -> torch.Tensor:
    """Apply clustered CLS-logit addition.

    Given a patch logit L in cluster C with prototype M (mean over logits in C),
    then
      L' = L + CosSim(M, CLS) * factor * bias

    Noise points (label=-1) remain unchanged.

    Args:
        patch_logits: (B, N, Q) per-patch logits before CLS addition
        labels: (B, N) cluster ids, -1 for noise; if None, returns input
        bias: (B, Q) CLS logits
        factor: scaling factor used in the addition
    """
    if labels is None:
        return items
    if items.dim() != 3:
        return items
    if labels.dim() != 2:
        return items
    if bias.dim() != 2:
        return items

    b, n, q = items.shape
    if labels.shape != (b, n):
        return items
    if bias.shape != (b, q):
        return items

    out = items
    lam = float(factor)
    if lam == 0.0:
        return out

    for bi in range(b):
        lab = labels[bi]
        valid = lab >= 0
        if not bool(valid.any()):
            continue

        cluster_ids = lab[valid]
        num_clusters = int(cluster_ids.max().item()) + 1

        # Prototype M_k = mean(logits) in the cluster (computed from original logits)
        pl = items[bi, valid].float()  # (M,Q)
        sums = torch.zeros((num_clusters, q), device=pl.device, dtype=torch.float32)
        counts = torch.zeros((num_clusters,), device=pl.device, dtype=torch.float32)
        sums.index_add_(0, cluster_ids, pl)
        ones = torch.ones((cluster_ids.shape[0],), device=pl.device, dtype=torch.float32)
        counts.index_add_(0, cluster_ids, ones)
        protos = sums / counts.clamp_min(1.0).unsqueeze(1)  # (K,Q)

        # CosSim(M_k, CLS)
        proto_u = protos / (protos.norm(dim=-1, keepdim=True) + eps)
        cls_vec = bias[bi].float()
        cls_u = cls_vec / (cls_vec.norm(dim=-1, keepdim=True) + eps)
        sims = (proto_u * cls_u.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)  # (K,)

        add = (sims[cluster_ids].unsqueeze(1) * (lam * cls_vec).unsqueeze(0))
        out[bi, valid] = (out[bi, valid].float() + add).to(out.dtype)

    return out
