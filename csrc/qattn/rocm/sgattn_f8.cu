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

/*
 * FP8 attention kernel for AMD RDNA4 (gfx12) using rocWMMA.
 *
 * This kernel uses:
 * - INT8 WMMA for Q@K^T computation
 * - FP8 WMMA for S@V computation (V stored as FP8)
 *
 * RDNA4 supports OCP FP8 types (hip_fp8_e4m3, hip_fp8_e5m2) via rocWMMA.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

namespace gfx12Params {
    constexpr uint32_t WAVE_SIZE = 32u;
    constexpr uint32_t WMMA_M = 16u;
    constexpr uint32_t WMMA_N = 16u;
    constexpr uint32_t WMMA_K_INT8 = 16u;    // WMMA K for INT8
    constexpr uint32_t WMMA_K_FP8 = 32u;     // WMMA K for FP8 (gfx12 uses 32)
    constexpr uint32_t WMMA_K_FP16 = 16u;    // WMMA K for FP16 (fallback)
}

constexpr float LOG2E = 1.44269504088896340736f;
#define div_ceil_hip(M, N) (((M) + (N) - 1) / (N))

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

// Type traits for output types
template<typename T> struct OutputTypeTraits;

template<>
struct OutputTypeTraits<half> {
    __device__ static float to_float(half val) { return __half2float(val); }
    __device__ static half from_float(float val) { return __float2half(val); }
};

template<>
struct OutputTypeTraits<hip_bfloat16> {
    __device__ static float to_float(hip_bfloat16 val) { return static_cast<float>(val); }
    __device__ static hip_bfloat16 from_float(float val) { return hip_bfloat16(val); }
};

// Fragment types for QK phase: INT8 M16N16K16
using FragA_QK = fragment<matrix_a, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_INT8, int8_t, col_major>;
using FragB_QK = fragment<matrix_b, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_INT8, int8_t, row_major>;
using FragAcc_QK = fragment<accumulator, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_INT8, int32_t>;

// Fragment types for SV phase with FP8 V: FP16 S @ FP8 V
// Note: rocWMMA on gfx12 supports FP8 matrix B (V matrix)
// S is stored as FP16/BF16 after softmax, V is FP8
// For the MMA, we use FP16 accumulator
template<typename DTypeOut> struct SVFragmentTypesFP8;

template<>
struct SVFragmentTypesFP8<half> {
    // S is half, V is FP8, accumulator is float
    using FragA = fragment<matrix_a, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, half, row_major>;
    using FragB = fragment<matrix_b, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, half, row_major>;
    using FragAcc = fragment<accumulator, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, float>;
};

template<>
struct SVFragmentTypesFP8<hip_bfloat16> {
    using FragA = fragment<matrix_a, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, bfloat16_t, row_major>;
    using FragAcc = fragment<accumulator, gfx12Params::WMMA_M, gfx12Params::WMMA_N, gfx12Params::WMMA_K_FP16, float>;
};

/*
 * WMMA element-to-matrix mapping helpers for gfx12.
 *
 * gfx12 accumulator layout (16x16 matrix, 32 lanes, 8 registers per lane):
 * - Lanes 0-15 own rows 0-7, lanes 16-31 own rows 8-15
 * - lane_id % 16 gives the column (0-15)
 * - register index gives the row within the 8-row block
 *
 * Example: lane 5, reg 3 -> row 3, col 5
 *          lane 21, reg 3 -> row 11, col 5
 */
__device__ __forceinline__ uint32_t wmma_elem_row(uint32_t reg, uint32_t lane_id) {
    return reg + (lane_id >> 4) * 8;  // reg + (lane >= 16 ? 8 : 0)
}

__device__ __forceinline__ uint32_t wmma_elem_col(uint32_t lane_id) {
    return lane_id & 15;  // lane_id % 16
}

/*
 * FP8 attention kernel for RDNA4 (gfx12).
 * Uses FP8 for V matrix storage and FP16/BF16 for intermediate S matrix.
 */
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t HEAD_DIM,
         QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
         bool use_inst_buffer, uint32_t pv_threshold_mode,
         typename DTypeOut, MaskMode mask_mode, bool return_pv_count, bool fuse_v_scale,
         uint32_t NUM_THREADS = (CTA_Q / WARP_Q) * gfx12Params::WAVE_SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
qk_int_sv_f8_block_sparse_attn_kernel_rocm(
    int8_t* __restrict__ Q,
    int8_t* __restrict__ K,
    __hip_fp8_e4m3* __restrict__ V,
    DTypeOut* __restrict__ O,
    int32_t* __restrict__ PV_Count,
    int32_t* __restrict__ Lut,
    int32_t* __restrict__ Valid_Block_Num,
    float* __restrict__ PV_Threshold,
    float* __restrict__ Q_scale,
    float* __restrict__ K_scale,
    float* __restrict__ V_scale,
    const uint32_t qo_len,
    const uint32_t kv_len,
    const uint32_t num_kv_groups,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale)
{
    using namespace gfx12Params;
    using Traits = OutputTypeTraits<DTypeOut>;
    using SVFrags = SVFragmentTypesFP8<DTypeOut>;
    using FragA_SV_T = typename SVFrags::FragA;
    using FragB_SV_T = typename SVFrags::FragB;
    using FragAcc_SV_T = typename SVFrags::FragAcc;

    constexpr uint32_t NUM_WARPS_Q = CTA_Q / WARP_Q;
    constexpr uint32_t NUM_WARPS_K = 1;
    constexpr uint32_t NUM_TILES_K = CTA_K / WMMA_N;
    constexpr uint32_t NUM_TILES_V = HEAD_DIM / WMMA_N;
    constexpr uint32_t NUM_K_ITERS = HEAD_DIM / WMMA_K_INT8;
    constexpr uint32_t NUM_SV_ITERS = CTA_K / WMMA_K_FP16;

    const uint32_t batch_id = blockIdx.z;
    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;

    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    const uint32_t warp_id = tid / WAVE_SIZE;
    const uint32_t lane_id = tid % WAVE_SIZE;
    const uint32_t warp_idx_q = warp_id / NUM_WARPS_K;
    const uint32_t warp_idx_k = warp_id % NUM_WARPS_K;

    sm_scale *= LOG2E;

    const uint32_t num_block_q = gridDim.x;
    const uint32_t num_block_k = div_ceil_hip(kv_len, CTA_K);
    const uint32_t num_iterations = Valid_Block_Num[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];

    if (num_iterations == 0) return;

    const int32_t* lut_ptr = Lut + batch_id * num_qo_heads * num_block_q * num_block_k +
                             head_id * num_block_q * num_block_k + bx * num_block_k;

    // Shared memory layout
    extern __shared__ char smem[];

    int8_t* smem_Q = reinterpret_cast<int8_t*>(smem);
    int8_t* smem_K = smem_Q + HEAD_DIM * CTA_Q;
    DTypeOut* smem_V = reinterpret_cast<DTypeOut*>(smem_K + HEAD_DIM * CTA_K);
    DTypeOut* smem_S = smem_V + CTA_K * HEAD_DIM;

    const uint32_t q_start = bx * CTA_Q;
    const uint32_t q_tile_row = warp_idx_q * WMMA_M;

    // Register-based state
    float RO[NUM_TILES_V][8];
    float m[8];
    float d[8];

    #pragma unroll
    for (uint32_t fv = 0; fv < NUM_TILES_V; fv++) {
        #pragma unroll
        for (uint32_t i = 0; i < 8; i++) {
            RO[fv][i] = 0.0f;
        }
    }
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        m[i] = -5000000.0f;
        d[i] = 0.0f;
    }

    // Load Q to shared memory
    for (uint32_t i = tid; i < CTA_Q * HEAD_DIM; i += NUM_THREADS) {
        uint32_t q_row = i % CTA_Q;
        uint32_t q_col = i / CTA_Q;
        uint32_t q_idx = q_start + q_row;
        int8_t val = 0;
        if (q_idx < qo_len) {
            val = Q[batch_id * stride_bz_q + q_idx * stride_seq_q + head_id * stride_h_q + q_col];
        }
        smem_Q[i] = val;
    }
    __syncthreads();

    // Get Q scale
    float q_scale_val;
    if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
        q_scale_val = Q_scale[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];
    } else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_block_q = num_block_q * NUM_WARPS_Q;
        q_scale_val = Q_scale[batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * NUM_WARPS_Q + warp_idx_q];
    }

    // Main loop over K blocks
    uint32_t k_block_idx = 0;
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        k_block_idx += lut_ptr[iter];
        uint32_t k_start = k_block_idx * CTA_K;

        // Load K to shared memory
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
            uint32_t n = i % CTA_K;
            uint32_t k = i / CTA_K;
            uint32_t k_idx = k_start + n;
            int8_t val = 0;
            if (k_idx < kv_len) {
                val = K[batch_id * stride_bz_k + k_idx * stride_seq_k + (head_id / num_kv_groups) * stride_h_k + k];
            }
            smem_K[k * CTA_K + n] = val;
        }

        // Load V from FP8, dequantize to FP16/BF16 for S@V
        // V is stored as [batch, head, headdim, padded_seq] in FP8
        // We need to transpose and dequantize to [seq, headdim]
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
            uint32_t v_row = i / HEAD_DIM;  // seq position within CTA_K
            uint32_t v_col = i % HEAD_DIM;  // head dimension
            uint32_t v_idx = k_start + v_row;

            DTypeOut val;
            if (v_idx < kv_len) {
                // V is stored as [batch, head, headdim, padded_seq]
                __hip_fp8_e4m3 v_fp8 = V[batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + v_col * stride_d_v + v_idx];
                float v_float = static_cast<float>(v_fp8);

                // Apply per-head-dim V scale if fusing
                if constexpr (fuse_v_scale) {
                    float scale = V_scale[batch_id * (num_qo_heads / num_kv_groups) * HEAD_DIM + (head_id / num_kv_groups) * HEAD_DIM + v_col];
                    v_float *= scale;
                }

                val = Traits::from_float(v_float);
            } else {
                val = Traits::from_float(0.0f);
            }
            smem_V[v_row * HEAD_DIM + v_col] = val;
        }
        __syncthreads();

        // Get K scale
        float k_scale_val;
        if constexpr (K_GRAN == QuantGranularity::kPerBlock) {
            const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
            k_scale_val = K_scale[batch_id * num_kv_heads * num_block_k + (head_id / num_kv_groups) * num_block_k + k_block_idx];
        } else if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
            const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
            const uint32_t num_warp_block_k = num_block_k * NUM_WARPS_K;
            k_scale_val = K_scale[batch_id * num_kv_heads * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + k_block_idx * NUM_WARPS_K + warp_idx_k];
        }

        float dequant_scale = q_scale_val * k_scale_val * sm_scale;

        // Phase 1: Compute QK^T using rocWMMA INT8
        float RS[NUM_TILES_K][8];

        #pragma unroll
        for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
            FragAcc_QK acc_qk;
            fill_fragment(acc_qk, 0);

            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < NUM_K_ITERS; k_iter++) {
                FragA_QK frag_q;
                load_matrix_sync(frag_q, smem_Q + k_iter * WMMA_K_INT8 * CTA_Q + q_tile_row, CTA_Q);

                FragB_QK frag_k;
                load_matrix_sync(frag_k, smem_K + k_iter * WMMA_K_INT8 * CTA_K + tile_k * WMMA_N, CTA_K);
                mma_sync(acc_qk, frag_q, frag_k, acc_qk);
            }

            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                float val = static_cast<float>(acc_qk.x[reg]) * dequant_scale;

                uint32_t row = wmma_elem_row(reg, lane_id);
                uint32_t col = wmma_elem_col(lane_id);
                uint32_t q_idx = q_start + q_tile_row + row;
                uint32_t k_idx = k_start + tile_k * WMMA_N + col;

                if (k_idx >= kv_len) val = -5000000.0f;
                if constexpr (mask_mode == MaskMode::kCausal) {
                    if (k_idx > q_idx) val = -5000000.0f;
                }

                RS[tile_k][reg] = val;
            }
        }

        // Phase 2: Online softmax update
        #pragma unroll
        for (uint32_t reg = 0; reg < 8; reg++) {
            float m_prev = m[reg];

            float m_local = RS[0][reg];
            #pragma unroll
            for (uint32_t tile_k = 1; tile_k < NUM_TILES_K; tile_k++) {
                m_local = fmaxf(m_local, RS[tile_k][reg]);
            }

            #pragma unroll
            for (uint32_t offset = 8; offset > 0; offset /= 2) {
                m_local = fmaxf(m_local, __shfl_xor(m_local, offset, WAVE_SIZE));
            }

            m[reg] = fmaxf(m_prev, m_local);
            float o_scale = exp2f(m_prev - m[reg]);

            d[reg] *= o_scale;
            #pragma unroll
            for (uint32_t fv = 0; fv < NUM_TILES_V; fv++) {
                RO[fv][reg] *= o_scale;
            }

            float local_sum = 0.0f;
            #pragma unroll
            for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
                RS[tile_k][reg] = exp2f(RS[tile_k][reg] - m[reg]);
                local_sum += RS[tile_k][reg];
            }

            #pragma unroll
            for (uint32_t offset = 8; offset > 0; offset /= 2) {
                local_sum += __shfl_xor(local_sum, offset, WAVE_SIZE);
            }

            d[reg] += local_sum;
        }

        // Phase 3: Store S to shared memory
        #pragma unroll
        for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                uint32_t row = wmma_elem_row(reg, lane_id);
                uint32_t col = wmma_elem_col(lane_id);
                uint32_t global_row = q_tile_row + row;
                uint32_t global_col = tile_k * WMMA_N + col;

                smem_S[global_row * CTA_K + global_col] = Traits::from_float(RS[tile_k][reg]);
            }
        }
        __syncthreads();

        // Phase 4: Compute S @ V using FP16 WMMA (V already dequantized)
        #pragma unroll
        for (uint32_t tile_v = 0; tile_v < NUM_TILES_V; tile_v++) {
            FragAcc_SV_T acc_sv;
            fill_fragment(acc_sv, 0.0f);

            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < NUM_SV_ITERS; k_iter++) {
                FragA_SV_T frag_s;
                load_matrix_sync(frag_s, smem_S + q_tile_row * CTA_K + k_iter * WMMA_K_FP16, CTA_K);

                FragB_SV_T frag_v;
                load_matrix_sync(frag_v, smem_V + k_iter * WMMA_K_FP16 * HEAD_DIM + tile_v * WMMA_N, HEAD_DIM);

                mma_sync(acc_sv, frag_s, frag_v, acc_sv);
            }

            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                RO[tile_v][reg] += acc_sv.x[reg];
            }
        }

        __syncthreads();
    }

    // Final: Normalize by d and write to output
    DTypeOut* smem_out = smem_S;

    #pragma unroll
    for (uint32_t tile_v = 0; tile_v < NUM_TILES_V; tile_v++) {
        #pragma unroll
        for (uint32_t reg = 0; reg < 8; reg++) {
            uint32_t row = wmma_elem_row(reg, lane_id);
            uint32_t col = wmma_elem_col(lane_id);
            uint32_t global_row = q_tile_row + row;
            uint32_t global_col = tile_v * WMMA_N + col;

            float inv_d = 1.0f / d[reg];
            float out_val = RO[tile_v][reg] * inv_d;

            smem_out[global_row * HEAD_DIM + global_col] = Traits::from_float(out_val);
        }
    }
    __syncthreads();

    // Copy from smem to global memory
    for (uint32_t i = tid; i < CTA_Q * HEAD_DIM; i += NUM_THREADS) {
        uint32_t row = i / HEAD_DIM;
        uint32_t col = i % HEAD_DIM;
        uint32_t o_idx = q_start + row;

        if (o_idx < qo_len) {
            O[batch_id * stride_bz_o + o_idx * stride_seq_o + head_id * stride_h_o + col] = smem_out[row * HEAD_DIM + col];
        }
    }
}

// Kernel launcher
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
    float sm_scale)
{
    constexpr QuantGranularity Q_GRAN = (qk_quant_gran == 1) ? QuantGranularity::kPerBlock : QuantGranularity::kPerWarp;
    constexpr QuantGranularity K_GRAN = Q_GRAN;
    constexpr MaskMode mask_mode = is_causal ? MaskMode::kCausal : MaskMode::kNone;

    const uint32_t num_kv_groups = num_qo_heads / num_kv_heads;
    const uint32_t num_block_q = div_ceil_hip(qo_len, CTA_Q);

    size_t smem_size = HEAD_DIM * CTA_Q * sizeof(int8_t) +
                       HEAD_DIM * CTA_K * sizeof(int8_t) +
                       CTA_K * HEAD_DIM * sizeof(DTypeOut) +
                       max(CTA_Q * CTA_K, CTA_Q * HEAD_DIM) * sizeof(DTypeOut);

    constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
    constexpr uint32_t NUM_THREADS = NUM_WARPS * gfx12Params::WAVE_SIZE;

    dim3 grid(num_block_q, num_qo_heads, batch_size);
    dim3 block(NUM_THREADS, 1, 1);

    hipLaunchKernelGGL((qk_int_sv_f8_block_sparse_attn_kernel_rocm<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
                        Q_GRAN, K_GRAN, use_inst_buffer, pv_threshold_mode, DTypeOut, mask_mode, return_pv_count, fuse_v_scale, NUM_THREADS>),
                       grid, block, smem_size, 0,
                       Q, K, V, O, PV_Count, Lut, Valid_Block_Num, PV_Threshold, Q_scale, K_scale, V_scale,
                       qo_len, kv_len, num_kv_groups,
                       stride_bz_q, stride_seq_q, stride_h_q,
                       stride_bz_k, stride_seq_k, stride_h_k,
                       stride_bz_v, stride_h_v, stride_d_v,
                       stride_bz_o, stride_seq_o, stride_h_o,
                       sm_scale);
}

// Explicit template instantiations for RDNA4
// CTA_Q=64, CTA_K=64, HEAD_DIM=64, half, fuse_v_scale=true
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 1, true, 0, half, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 1, true, 0, half, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=64, CTA_K=64, HEAD_DIM=64, hip_bfloat16, fuse_v_scale=true
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 1, true, 0, hip_bfloat16, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 1, true, 0, hip_bfloat16, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=64, CTA_K=64, HEAD_DIM=128, half, fuse_v_scale=true
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 1, true, 0, half, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 1, true, 0, half, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=64, CTA_K=64, HEAD_DIM=128, hip_bfloat16, fuse_v_scale=true
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 1, true, 0, hip_bfloat16, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 1, true, 0, hip_bfloat16, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 variants
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 2, true, 0, half, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 2, true, 0, half, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 2, true, 0, half, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 2, true, 0, half, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 with hip_bfloat16
template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 2, true, 0, hip_bfloat16, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 64, 2, true, 0, hip_bfloat16, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 2, true, 0, hip_bfloat16, true, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmFP8Dispatched<64, 64, 16, 64, 128, 2, true, 0, hip_bfloat16, false, true, false>(
    int8_t*, int8_t*, __hip_fp8_e4m3*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);
