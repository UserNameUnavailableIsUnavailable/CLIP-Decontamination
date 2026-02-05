import os
import torch
import torch.nn as nn
import sys
import numpy as np
import cv2

sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler
from CTD import cluster_patch_tokens_dbscan, adaptive_debiasing

@MODELS.register_module()
class SegmentorEx(BaseSegmentor):
    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0.0,
                 global_debias_factor=0.0,
                 bg_idx=0,
                 apply_sim_feat_up=False,
                 sim_feat_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path'),
                 apply_ctd=False,
                 apply_outlier_suppression=False,
                 outlier_suppression_cfg=None,
                 apply_self_attn_enhancement=False,
                 self_attn_enhancement_cfg=None,
                 apply_layer_fusion=False,
                 layer_fusion_lambda=0.5,
                 layer_fusion_threshold=0.7,
                 apply_similarity_enhancement=False,
                 similarity_enhancement_cfg=None,
                 result_dir=None,
                 heatmap_dir=None,
                 ):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.apply_sim_feat_up = apply_sim_feat_up
        self.cls_token_lambda = cls_token_lambda
        if cls_token_lambda != 0.0:
            print(f"[SegEarth-OV] CLS subtraction from logits enabled (λ = {cls_token_lambda})")
        
        self.global_debias_factor = global_debias_factor
        if global_debias_factor != 0.0:
            print(f"[SegEarth-OV] Similarity-weighted global debiasing enabled (factor = {global_debias_factor})")
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        # DBSCAN clustering/refinement on patch tokens (B, N, C) before similarity.
        self.apply_ctd = apply_ctd
        if apply_ctd:
            print(f"[CTD] Cluster-Then-Debias enabled")
        
        # Layer-wise feature fusion via exponential moving average
        self.apply_layer_fusion = apply_layer_fusion
        self.layer_fusion_lambda = layer_fusion_lambda
        self.layer_fusion_threshold = layer_fusion_threshold
        if apply_layer_fusion:
            print(f"[Layer Fusion] Enabled with lambda={layer_fusion_lambda}, threshold={layer_fusion_threshold}")
        
        # Similarity-based attention enhancement: adds self-similarity map to attention weights
        self.apply_similarity_enhancement = apply_similarity_enhancement
        if apply_similarity_enhancement:
            from similarity_enhancement import SimilarityEnhancementModule
            
            default_sim_cfg = dict(
                similarity_weight=1.0,  # Weight for similarity map when adding to attention
                temperature=1.0,  # Temperature for similarity computation
                add_self_similarity=True,  # Whether to include diagonal in similarity map
            )
            if similarity_enhancement_cfg:
                default_sim_cfg.update(similarity_enhancement_cfg)
            
            enhancer = SimilarityEnhancementModule(
                similarity_weight=default_sim_cfg['similarity_weight'],
                temperature=default_sim_cfg['temperature'],
                add_self_similarity=default_sim_cfg['add_self_similarity'],
            ).to(device)
            
            # Set enhancer on vision transformer
            if self.clip_type != 'BLIP':
                self.net.visual.similarity_enhancer = enhancer
            else:
                self.net.visual_encoder.similarity_enhancer = enhancer
            
            print(f"[Similarity Enhancement] Enabled: adds self-similarity to attention (weight={default_sim_cfg['similarity_weight']}, temp={default_sim_cfg['temperature']})")
        
        # Self-Attention Enhancement: Boosts self-attention for tokens with weak self-attention
        self.apply_self_attn_enhancement = apply_self_attn_enhancement
        
        if self.apply_self_attn_enhancement:
            from self_attention_enhancement import SelfAttentionEnhancementModule
            
            # Default configuration
            default_self_attn_cfg = dict(
                enhancement_strength=0.1,  # How much to enhance (0.1 = mild, 0.3 = strong)
                min_self_attn_threshold=0.15,  # Tokens below this get enhanced
                mode='feature'  # 'feature' or 'attention'
            )
            if self_attn_enhancement_cfg:
                default_self_attn_cfg.update(self_attn_enhancement_cfg)
            
            self_attn_enhancer = SelfAttentionEnhancementModule(
                enhancement_strength=default_self_attn_cfg['enhancement_strength'],
                min_self_attn_threshold=default_self_attn_cfg['min_self_attn_threshold'],
                mode=default_self_attn_cfg['mode']
            ).to(device)
            
            # Set enhancer on vision transformer
            if self.clip_type != 'BLIP':
                self.net.visual.self_attn_enhancer = self_attn_enhancer
            else:
                self.net.visual_encoder.self_attn_enhancer = self_attn_enhancer
            
            print(f"[Self-Attention Enhancement] Enabled with strength={default_self_attn_cfg['enhancement_strength']}, threshold={default_self_attn_cfg['min_self_attn_threshold']}, mode={default_self_attn_cfg['mode']}")
        
        # Outlier Suppression: Detects and replaces outliers based on attention weights
        self.apply_outlier_suppression = apply_outlier_suppression
        
        if self.apply_outlier_suppression:
            from outlier_suppression import OutlierSuppressionModule
            
            # Default configuration
            default_outlier_cfg = dict(
                top_k=10,  # Number of outlier tokens to suppress per image
            )
            if outlier_suppression_cfg:
                default_outlier_cfg.update(outlier_suppression_cfg)
            
            suppressor = OutlierSuppressionModule(
                top_k=default_outlier_cfg['top_k']
            ).to(device)
            
            # Set suppressor on vision transformer
            if self.clip_type != 'BLIP':
                self.net.visual.outlier_suppressor = suppressor
            else:
                self.net.visual_encoder.outlier_suppressor = suppressor
            
            print(f"[Outlier Suppression] Enabled with top_k={default_outlier_cfg['top_k']}")
        self.result_dir = result_dir
        self.heatmap_dir = heatmap_dir
        
        if self.apply_sim_feat_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(sim_feat_up_cfg['model_name'], self.feat_dim).cuda().half()
            ckpt = torch.load(sim_feat_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)
            print(f"[SimFeatUp] Feature upsampling enabled (model={sim_feat_up_cfg['model_name']})")

    def forward_feature(self, img, logit_size=None, tile_h_idx=None, tile_w_idx=None):
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            # Standard CLIP encoding
            image_features = self.net.encode_image(
                img, 
                self.model_type, 
                self.ignore_residual, 
                output_cls_token=True,
                apply_layer_fusion=self.apply_layer_fusion,
                layer_fusion_lambda=self.layer_fusion_lambda,
                layer_fusion_threshold=self.layer_fusion_threshold,
                apply_similarity_enhancement=self.apply_similarity_enhancement,
            )
            
        # Always extract CLS token (needed for CTD and optional for cls_logits)
        image_cls_token, image_features = image_features
        image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
        cls_logits = image_cls_token @ self.query_features.T

        # Grid dimensions at patch resolution
        feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        image_w, image_h = img[0].shape[-2], img[0].shape[-1]
        
        # Pipeline: Outlier suppression (between layers) → Global debiasing → CTD → SimFeatUp → CLS subtraction
        # Note: Outlier suppression is already applied between transformer blocks (see __init__)
        
        # Global debiasing: Similarity-weighted CLS subtraction from features BEFORE upsampling
        # This removes global bias early in the pipeline (controlled by global_debias_factor)
        if self.global_debias_factor != 0:
            # Normalize features for cosine similarity computation
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            cls_token_norm = image_cls_token / image_cls_token.norm(dim=-1, keepdim=True)
            
            # Compute similarity between each patch and CLS token: [B, N]
            similarity = (image_features_norm * cls_token_norm.unsqueeze(1)).sum(dim=-1)
            
            # Weighted factor: higher similarity = more contaminated = more aggressive removal
            # similarity ranges from -1 to 1, multiply by factor for adaptive debiasing
            weighted_factor = similarity.unsqueeze(-1) * self.global_debias_factor  # [B, N, 1]
            
            # Apply similarity-weighted CLS subtraction
            # Patches similar to CLS get stronger debiasing
            image_features = image_features - image_cls_token.unsqueeze(1) * weighted_factor
        
        # Apply CTD (Cluster-Then-Debias) on image_features
        if self.apply_ctd:
            # Cluster at patch resolution on image features
            grid_h, grid_w = feature_w, feature_h
            # Cluster on patch-level image features
            _, cluster_labels = cluster_patch_tokens_dbscan(
                image_features,
                grid_hw=(grid_h, grid_w),
                cfg_dict={
                    'max_points': 8192,
                    'metric': 'euclidean',
                    'eps': 1.1,
                    'min_samples': 11,
                }
            )

            # Diagnostic output
            if cluster_labels is not None:
                num_clusters = (cluster_labels >= 0).long().unique().numel() if cluster_labels.numel() > 0 else 0
                num_noise = (cluster_labels == -1).sum().item()
                # print(f"[DBSCAN] Found {num_clusters} clusters, {num_noise} noise points out of {cluster_labels.numel()} total")
            
            image_features = adaptive_debiasing(
                items=image_features,
                labels=cluster_labels,
                bias=image_cls_token,
                factor=-1.5
            )

        # featup
        if self.apply_sim_feat_up:
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        # SegEarth-OV debiasing from logits (optional, controlled by cls_token_lambda)
        if self.cls_token_lambda != 0:
            logits = logits + cls_logits * self.cls_token_lambda

        if self.apply_sim_feat_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        # Standard sliding window inference
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img, tile_h_idx=h_idx, tile_w_idx=w_idx)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })

                if self.result_dir or self.heatmap_dir:
                    meta = data_samples[i].metainfo if hasattr(data_samples[i], 'metainfo') else {}
                    # derive a filename stem from available meta
                    stem = None
                    for key in ('img_path', 'ori_path', 'filename', 'ori_filename'):
                        if key in meta and meta[key]:
                            stem = os.path.splitext(os.path.basename(meta[key]))[0]
                            break
                    if stem is None:
                        stem = f'sample_{i}'

                    if self.result_dir:
                        os.makedirs(self.result_dir, exist_ok=True)
                        color_mask = self._colorize_mask(seg_pred.squeeze(0).to('cpu').numpy())
                        if cv2 is not None:
                            cv2.imwrite(os.path.join(self.result_dir, f"{stem}.png"), color_mask[:, :, ::-1])
                        else:
                            # Fallback: save via numpy if cv2 unavailable
                            from PIL import Image
                            Image.fromarray(color_mask).save(os.path.join(self.result_dir, f"{stem}.png"))

                    if self.heatmap_dir:
                        os.makedirs(self.heatmap_dir, exist_ok=True)
                        # Max probability across classes as confidence heatmap
                        conf = seg_logits.max(dim=0, keepdim=False)[0].to('cpu').numpy()
                        heat = self._to_colormap(conf)
                        if cv2 is not None:
                            cv2.imwrite(os.path.join(self.heatmap_dir, f"{stem}.png"), heat[:, :, ::-1])
                        else:
                            from PIL import Image
                            Image.fromarray(heat).save(os.path.join(self.heatmap_dir, f"{stem}.png"))
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

    def _generate_palette(self, n):
        # Deterministic HSV-based palette
        import colorsys
        palette = []
        for idx in range(n):
            h = (idx / max(1, n)) % 1.0
            s = 0.75
            v = 1.0 if idx != self.bg_idx else 0.2
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette.append([int(r * 255), int(g * 255), int(b * 255)])
        return np.array(palette, dtype=np.uint8)

    def _colorize_mask(self, mask2d: np.ndarray) -> np.ndarray:
        # mask2d: HxW integer labels
        n = int(self.num_classes) if hasattr(self, 'num_classes') else int(mask2d.max() + 1)
        palette = getattr(self, '_palette_cache', None)
        if palette is None or len(palette) < n:
            palette = self._generate_palette(n)
            self._palette_cache = palette
        mask2d = mask2d.astype(np.int32)
        h, w = mask2d.shape
        color = palette[np.clip(mask2d, 0, len(palette) - 1)]  # HxWx3 (RGB)
        return color  # RGB uint8

    def _to_colormap(self, conf2d: np.ndarray) -> np.ndarray:
        # conf2d: HxW float in [0,1]
        conf = conf2d.astype(np.float32)
        if conf.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        conf = np.nan_to_num(conf, nan=0.0)
        conf = np.clip(conf, 0.0, 1.0)
        if cv2 is not None:
            gray = (conf * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # BGR uint8
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            return heat
        else:
            # simple RGB gradient fallback (blue->red)
            gray = (conf * 255.0).astype(np.uint8)
            heat = np.stack([gray, np.zeros_like(gray), 255 - gray], axis=-1)
            return heat


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices