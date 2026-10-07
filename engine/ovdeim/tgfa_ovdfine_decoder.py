"""Text-guided competitive query selection and multispectral decoding.

This implementation keeps the text-guidance path while removing the
post-CQS and post-TextMHA learned gates. Text context is fused directly
through residual addition followed by LayerNorm. The objectness residual and
all non-gated text attention operations are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register
from ..deim.dfine_utils import distance2bbox, weighting_function
from ._shared import inverse_sigmoid
from .cls_embed import ClassEmbed
from .ovdfine_decoder import OVDFINETransformer, OVDFINETransformerDecoder

__all__ = ["TGFAOVDFINETransformer"]


class TextGuidedMultispectralDecoder(OVDFINETransformerDecoder):
    """Refine every object query with class text before visual cross-attention."""

    def __init__(self, base_decoder: nn.Module, text_nhead: int = 4) -> None:
        super().__init__(base_decoder)
        if self.hidden_dim % text_nhead:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by text_nhead={text_nhead}")
        self.text_attn = nn.ModuleList(
            [nn.MultiheadAttention(self.hidden_dim, text_nhead, batch_first=True) for _ in range(self.num_layers)]
        )
        self.text_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        text_feats,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs = [], [], [], []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, "project") else self.project
        ref_points_detach = torch.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            text_delta, _ = self.text_attn[i](output, text_feats, text_feats, need_weights=False)
            output = self.text_norm[i](output + text_delta)

            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = torch.nn.functional.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = torch.nn.functional.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
            if i == 0:
                pre_bboxes = torch.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = None if not self.training and isinstance(score_head[0], nn.Identity) else self._run_score_head(
                    score_head[0], output, text_feats
                )
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)
            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](self._run_score_head(score_head[i], output, text_feats), pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register()
class TGFAOVDFINETransformer(OVDFINETransformer):
    """DAMSDet-inspired text-competitive query initialization for DEIM-MMT."""

    def __init__(
        self,
        *args,
        text_nhead: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_text_norm = nn.LayerNorm(self.hidden_dim)
        self.query_objectness = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.query_objectness.weight)
        nn.init.zeros_(self.query_objectness.bias)
        self.decoder = TextGuidedMultispectralDecoder(self.decoder, text_nhead)

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        text_feats: torch.Tensor,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        initial_logits = self.enc_score_head(output_memory, text_feats)
        text_weights = initial_logits.softmax(dim=-1)
        text_context = text_weights @ text_feats
        semantic_memory = self.query_text_norm(output_memory + text_context)

        enc_outputs_logits = self.enc_score_head(semantic_memory, text_feats)
        competitive_logits = enc_outputs_logits + self.query_objectness(semantic_memory)
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            semantic_memory, competitive_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes_list.append(torch.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)
        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content else enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits
