/*
 * Copyright (c) 2025 by SpargeAttn team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#include "attn_rocm.h"

// External template declarations
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t HEAD_DIM,
         uint32_t qk_quant_gran, bool use_inst_buffer, uint32_t pv_threshold_mode,
         typename DTypeOut, bool is_causal, bool fuse_v_scale, bool return_pv_count>
void SpargeAttentionROCmFP8Dispatched(
    int8_t* Q, int8_t* K, __hip_fp8_e4m3* V, DTypeOut* O,
    int32_t* PV_Count, int32_t* Lut, int32_t* Valid_Block_Num, float* PV_Threshold,
    float* Q_scale, float* K_scale, float* V_scale,
    const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale);

inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) TORCH_CHECK(x.stride(-1) == 1, #x " must have contiguous last dim")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK(x.scalar_type() == dtype, #x " must have dtype " #dtype)
#define CHECK_DIMS(x, dims) TORCH_CHECK(x.dim() == dims, #x " must have " #dims " dimensions")

// Dispatch macro for FP8 kernel with head_dim=64
#define DISPATCH_FP8_KERNEL_64(QUANT_GRAN, CAUSAL, DTYPE) \
    SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, QUANT_GRAN, true, 0, DTYPE, CAUSAL, true, false>( \
        reinterpret_cast<int8_t*>(query.data_ptr()), \
        reinterpret_cast<int8_t*>(key.data_ptr()), \
        reinterpret_cast<__hip_fp8_e4m3*>(value.data_ptr()), \
        reinterpret_cast<DTYPE*>(output.data_ptr()), \
        nullptr, \
        reinterpret_cast<int32_t*>(lut.data_ptr()), \
        reinterpret_cast<int32_t*>(valid_block_num.data_ptr()), \
        nullptr, \
        reinterpret_cast<float*>(query_scale.data_ptr()), \
        reinterpret_cast<float*>(key_scale.data_ptr()), \
        reinterpret_cast<float*>(value_scale.data_ptr()), \
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, \
        stride_bz_q, stride_seq_q, stride_h_q, \
        stride_bz_k, stride_seq_k, stride_h_k, \
        stride_bz_v, stride_h_v, stride_d_v, \
        stride_bz_o, stride_seq_o, stride_h_o, \
        sm_scale)

// Dispatch macro for FP8 kernel with head_dim=128
#define DISPATCH_FP8_KERNEL_128(QUANT_GRAN, CAUSAL, DTYPE) \
    SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, QUANT_GRAN, true, 0, DTYPE, CAUSAL, true, false>( \
        reinterpret_cast<int8_t*>(query.data_ptr()), \
        reinterpret_cast<int8_t*>(key.data_ptr()), \
        reinterpret_cast<__hip_fp8_e4m3*>(value.data_ptr()), \
        reinterpret_cast<DTYPE*>(output.data_ptr()), \
        nullptr, \
        reinterpret_cast<int32_t*>(lut.data_ptr()), \
        reinterpret_cast<int32_t*>(valid_block_num.data_ptr()), \
        nullptr, \
        reinterpret_cast<float*>(query_scale.data_ptr()), \
        reinterpret_cast<float*>(key_scale.data_ptr()), \
        reinterpret_cast<float*>(value_scale.data_ptr()), \
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, \
        stride_bz_q, stride_seq_q, stride_h_q, \
        stride_bz_k, stride_seq_k, stride_h_k, \
        stride_bz_v, stride_h_v, stride_d_v, \
        stride_bz_o, stride_seq_o, stride_h_o, \
        sm_scale)

// Main entry point for FP8 V matrix attention
void qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lut,
    torch::Tensor valid_block_num,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale)
{
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    CHECK_CUDA(output);
    CHECK_CUDA(lut);
    CHECK_CUDA(valid_block_num);
    CHECK_CUDA(query_scale);
    CHECK_CUDA(key_scale);
    CHECK_CUDA(value_scale);

    CHECK_LASTDIM_CONTIGUOUS(query);
    CHECK_LASTDIM_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(value);
    CHECK_LASTDIM_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(lut);
    CHECK_CONTIGUOUS(valid_block_num);
    CHECK_CONTIGUOUS(query_scale);
    CHECK_CONTIGUOUS(key_scale);
    CHECK_CONTIGUOUS(value_scale);

    CHECK_DTYPE(query, torch::kInt8);
    CHECK_DTYPE(key, torch::kInt8);
    CHECK_DTYPE(value, torch::kFloat8_e4m3fn);
    CHECK_DTYPE(query_scale, torch::kFloat32);
    CHECK_DTYPE(key_scale, torch::kFloat32);
    CHECK_DTYPE(value_scale, torch::kFloat32);

    CHECK_DIMS(query, 4);
    CHECK_DIMS(key, 4);
    CHECK_DIMS(value, 4);
    CHECK_DIMS(output, 4);

    const int batch_size = query.size(0);
    const int head_dim = query.size(3);

    int stride_bz_q = query.stride(0);
    int stride_bz_k = key.stride(0);
    int stride_bz_v = value.stride(0);
    int stride_bz_o = output.stride(0);

    int qo_len, kv_len, num_qo_heads, num_kv_heads;
    int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k, stride_h_v, stride_d_v, stride_seq_o, stride_h_o;

    if (tensor_layout == 0) {
        qo_len = query.size(1);
        kv_len = key.size(1);
        num_qo_heads = query.size(2);
        num_kv_heads = key.size(2);

        stride_seq_q = query.stride(1);
        stride_h_q = query.stride(2);
        stride_seq_k = key.stride(1);
        stride_h_k = key.stride(2);
        stride_h_v = value.stride(2);
        stride_d_v = value.stride(1);
        stride_seq_o = output.stride(1);
        stride_h_o = output.stride(2);
    } else {
        qo_len = query.size(2);
        kv_len = key.size(2);
        num_qo_heads = query.size(1);
        num_kv_heads = key.size(1);

        stride_seq_q = query.stride(2);
        stride_h_q = query.stride(1);
        stride_seq_k = key.stride(2);
        stride_h_k = key.stride(1);
        stride_h_v = value.stride(1);
        stride_d_v = value.stride(2);
        stride_seq_o = output.stride(2);
        stride_h_o = output.stride(1);
    }

    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads must be divisible by num_kv_heads");
    TORCH_CHECK(qk_quant_gran >= 1 && qk_quant_gran <= 2, "qk_quant_gran must be 1 (per-block) or 2 (per-warp)");

    const bool is_bf16_out = output.scalar_type() == torch::kBFloat16;

    // Dispatch based on head_dim, causal, quantization granularity, and dtype
    if (head_dim == 64) {
        if (qk_quant_gran == 1) {
            if (is_causal) {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_64(1, true, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_64(1, true, half);
                }
            } else {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_64(1, false, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_64(1, false, half);
                }
            }
        } else {
            if (is_causal) {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_64(2, true, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_64(2, true, half);
                }
            } else {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_64(2, false, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_64(2, false, half);
                }
            }
        }
    } else {
        // head_dim == 128
        if (qk_quant_gran == 1) {
            if (is_causal) {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_128(1, true, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_128(1, true, half);
                }
            } else {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_128(1, false, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_128(1, false, half);
                }
            }
        } else {
            if (is_causal) {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_128(2, true, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_128(2, true, half);
                }
            } else {
                if (is_bf16_out) {
                    DISPATCH_FP8_KERNEL_128(2, false, hip_bfloat16);
                } else {
                    DISPATCH_FP8_KERNEL_128(2, false, half);
                }
            }
        }
    }
}

// Entry point with PV threshold
torch::Tensor qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lut,
    torch::Tensor valid_block_num,
    torch::Tensor pv_threshold,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_pv_count)
{
    // For now, call the non-threshold version
    qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale(
        query, key, value, output, lut, valid_block_num,
        query_scale, key_scale, value_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale);

    return torch::empty({0});
}
