# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 FP8 MoE method using TurboMind s884 GEMM kernels."""

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_DEFAULT_PERSISTENT_MAX_TOKENS = 32


class Fp8SM70MoEMethod(FusedMoEMethodBase):
    """FP8 MoE path for V100 using TurboMind W8A16 GEMM.

    Supports serialized FP8 checkpoints with block size 128x128. The weights
    stay packed as FP8 and are consumed by TurboMind's SM70 s884 kernels.
    """

    def __init__(self, quant_config, moe):
        super().__init__(moe)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        if self.block_quant and tuple(self.weight_block_size) != (128, 128):
            raise ValueError(
                "Fp8SM70MoEMethod only supports FP8 block size [128, 128]."
            )
        if not self.block_quant:
            raise ValueError(
                "Fp8SM70MoEMethod currently requires block-wise FP8 weights."
            )
        self.group_size = 128

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        params_dtype = torch.float8_e4m3fn
        block_n, block_k = self.weight_block_size

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_scale = Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale_inv
        w2_scale = layer.w2_weight_scale_inv

        if self.quant_config.activation_scheme == "static":
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                layer.w13_input_scale, layer.w2_input_scale
            )
            layer.w13_input_scale = w13_input_scale
            layer.w2_input_scale = w2_input_scale

        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        num_experts = int(w13.shape[0])
        w13_tm_weights, w13_tm_scales, w13_meta = [], [], []
        w2_tm_weights, w2_tm_scales, w2_meta = [], [], []
        for expert_id in range(num_experts):
            r13 = ops.fp8_sm70_prepare(
                w13[expert_id], w13_scale[expert_id], self.group_size
            )
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])

            r2 = ops.fp8_sm70_prepare(
                w2[expert_id], w2_scale[expert_id], self.group_size
            )
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld, w13_q_ld = int(w13_meta[0][0].item()), int(w13_meta[0][1].item())
        w2_k_ld, w2_q_ld = int(w2_meta[0][0].item()), int(w2_meta[0][1].item())
        w13_ptrs = ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight, layer.w13_tm_scales, w13_k_ld, w13_q_ld, num_experts
        )
        w2_ptrs = ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight, layer.w2_tm_scales, w2_k_ld, w2_q_ld, num_experts
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_logical_size = int(w2.shape[1])
        layer.sm70_w13_k_dim = int(layer.w13_tm_weight.shape[1])
        layer.sm70_w13_n_dim = int(layer.w13_tm_weight.shape[2])
        layer.sm70_w2_k_dim = int(layer.w2_tm_weight.shape[1])
        layer.sm70_w2_n_dim = int(layer.w2_tm_weight.shape[2])
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim

        self._allocate_buffers(layer)
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        logger.info_once(
            "SM70 FP8 MoE TurboMind path enabled (%d experts).", num_experts
        )

    def _allocate_buffers(self, layer: torch.nn.Module) -> None:
        device = layer.w13_tm_weight.device
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        num_experts = layer.sm70_num_experts
        layer._fp8_buf_max_tokens = persistent_tokens
        layer._fp8_buf_max_slots = max_slots
        layer._fp8_buf_top_k = top_k
        layer._fp8_buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        layer._fp8_buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._fp8_buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        layer._fp8_buf_expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        )
        layer._fp8_buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._fp8_buf_topk_ids_i32 = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._fp8_buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        layer._fp8_buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._fp8_buf_m_indices = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )

    def _get_buffers(
        self, layer: torch.nn.Module, total_slots: int, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        if (
            total_slots <= layer._fp8_buf_max_slots
            and num_tokens <= layer._fp8_buf_max_tokens
        ):
            return {
                "output": layer._fp8_buf_output[:num_tokens],
                "permuted_input": layer._fp8_buf_permuted_input[:total_slots],
                "intermediate": layer._fp8_buf_intermediate[:total_slots],
                "gate_up": layer._fp8_buf_gate_up[:total_slots],
                "sorted_output": layer._fp8_buf_sorted_output[:total_slots],
                "expert_offsets": layer._fp8_buf_expert_offsets,
                "expert_offsets64": layer._fp8_buf_expert_offsets64,
                "inv_permuted_idx": layer._fp8_buf_inv_permuted_idx[:num_tokens],
                "topk_ids_i32": layer._fp8_buf_topk_ids_i32[:num_tokens],
                "token_expert_indices": layer._fp8_buf_token_expert_indices[
                    :num_tokens
                ],
                "permuted_idx": layer._fp8_buf_permuted_idx[:total_slots],
                "m_indices": layer._fp8_buf_m_indices[:total_slots],
            }
        device = layer._fp8_buf_output.device
        top_k = layer._fp8_buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        return {
            "output": torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                total_slots,
                layer.sm70_intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "gate_up": torch.empty(
                total_slots,
                layer.sm70_w13_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "sorted_output": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids_i32": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "m_indices": torch.empty(total_slots, dtype=torch.int32, device=device),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        topk_ids_i32 = buffers["topk_ids_i32"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        torch.ops._moe_C.moe_permute(
            x,
            topk_ids_i32,
            buffers["token_expert_indices"],
            None,
            layer.sm70_num_experts,
            layer.sm70_num_experts,
            top_k,
            None,
            buffers["permuted_input"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["permuted_idx"],
            buffers["m_indices"],
        )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        ops.fp8_moe_gemm_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            buffers["expert_offsets"],
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim,
            self.group_size,
            False,
        )
        torch.ops._C.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        ops.fp8_moe_gemm_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            buffers["expert_offsets"],
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            layer.sm70_num_experts,
            layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim,
            self.group_size,
            False,
        )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            top_k,
            output,
        )
        return output

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None
