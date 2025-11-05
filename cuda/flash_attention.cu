#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int WARP_SIZE = 32;

template <typename scalar_t, int BLOCK_D>
__global__ void flash_attention_forward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    float* __restrict__ L,
    int N,
    int d,
    int stride_qn,
    int stride_kn,
    int stride_vn,
    int stride_on,
    bool is_causal
) {
    int batch_head = blockIdx.y;
    int tile_q = blockIdx.x;
    int tid = threadIdx.x;

    int q_start = tile_q * BQ;

    const scalar_t* Q_bh = Q + batch_head * stride_qn;
    const scalar_t* K_bh = K + batch_head * stride_kn;
    const scalar_t* V_bh = V + batch_head * stride_vn;
    scalar_t* O_bh = O + batch_head * stride_on;
    float* L_bh = L + batch_head * N;

    extern __shared__ char smem_raw[];
    float* s_Q = reinterpret_cast<float*>(smem_raw);
    float* s_K = s_Q + BQ * BLOCK_D;
    float* s_V = s_K + BK * BLOCK_D;
    float* s_S = s_V + BK * BLOCK_D;

    float o_acc[BQ];
    float m_i[BQ];
    float l_i[BQ];

    int rows_per_thread = BQ / blockDim.x;
    if (rows_per_thread < 1) rows_per_thread = 1;

    for (int i = tid; i < BQ * BLOCK_D; i += blockDim.x) {
        int row = i / BLOCK_D;
        int col = i % BLOCK_D;
        int global_row = q_start + row;
        if (global_row < N) {
            s_Q[row * BLOCK_D + col] = static_cast<float>(Q_bh[global_row * d + col]);
        } else {
            s_Q[row * BLOCK_D + col] = 0.0f;
        }
    }

    for (int i = tid; i < BQ; i += blockDim.x) {
        m_i[0] = -FLT_MAX;
        l_i[0] = 0.0f;
    }

    float o_local[8];
    int d_per_thread = BLOCK_D / blockDim.x;
    if (d_per_thread < 1) d_per_thread = 1;

    float row_m[BQ];
    float row_l[BQ];
    float row_o[BQ * 4];

    for (int r = 0; r < BQ; r++) {
        row_m[r] = -FLT_MAX;
        row_l[r] = 0.0f;
    }

    __shared__ float s_O[BQ * 128];
    for (int i = tid; i < BQ * BLOCK_D; i += blockDim.x) {
        s_O[i] = 0.0f;
    }
    __syncthreads();

    int num_kv_tiles = (N + BK - 1) / BK;
    float scale = 1.0f / sqrtf(static_cast<float>(d));

    for (int tile_k = 0; tile_k < num_kv_tiles; tile_k++) {
        int k_start = tile_k * BK;

        if (is_causal && k_start > q_start + BQ - 1) break;

        for (int i = tid; i < BK * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D;
            int col = i % BLOCK_D;
            int global_row = k_start + row;
            if (global_row < N) {
                s_K[row * BLOCK_D + col] = static_cast<float>(K_bh[global_row * d + col]);
                s_V[row * BLOCK_D + col] = static_cast<float>(V_bh[global_row * d + col]);
            } else {
                s_K[row * BLOCK_D + col] = 0.0f;
                s_V[row * BLOCK_D + col] = 0.0f;
            }
        }
        __syncthreads();

        for (int i = tid; i < BQ * BK; i += blockDim.x) {
            int qr = i / BK;
            int kr = i % BK;
            int global_q = q_start + qr;
            int global_k = k_start + kr;

            float dot = 0.0f;
            for (int dd = 0; dd < BLOCK_D; dd++) {
                dot += s_Q[qr * BLOCK_D + dd] * s_K[kr * BLOCK_D + dd];
            }
            dot *= scale;

            if (global_q >= N || global_k >= N) dot = -1e6f;
            if (is_causal && global_k > global_q) dot = -1e6f;

            s_S[qr * BK + kr] = dot;
        }
        __syncthreads();

        for (int qr = tid; qr < BQ; qr += blockDim.x) {
            if (q_start + qr >= N) continue;

            float m_old = row_m[qr];
            float m_new = m_old;
            for (int kr = 0; kr < BK; kr++) {
                float s = s_S[qr * BK + kr];
                if (s > m_new) m_new = s;
            }

            float rescale = expf(m_old - m_new);
            float l_new = row_l[qr] * rescale;

            for (int kr = 0; kr < BK; kr++) {
                float p = expf(s_S[qr * BK + kr] - m_new);
                l_new += p;
                for (int dd = 0; dd < BLOCK_D; dd++) {
                    if (kr == 0) {
                        s_O[qr * BLOCK_D + dd] *= rescale;
                    }
                    s_O[qr * BLOCK_D + dd] += p * s_V[kr * BLOCK_D + dd];
                }
            }

            row_m[qr] = m_new;
            row_l[qr] = l_new;
        }
        __syncthreads();
    }

    for (int i = tid; i < BQ * BLOCK_D; i += blockDim.x) {
        int qr = i / BLOCK_D;
        int dd = i % BLOCK_D;
        int global_q = q_start + qr;
        if (global_q < N && row_l[qr] > 0.0f) {
            O_bh[global_q * d + dd] = static_cast<scalar_t>(s_O[qr * BLOCK_D + dd] / row_l[qr]);
        }
    }

    for (int qr = tid; qr < BQ; qr += blockDim.x) {
        int global_q = q_start + qr;
        if (global_q < N) {
            L_bh[global_q] = row_m[qr] + logf(row_l[qr] + 1e-12f);
        }
    }
}

template <typename scalar_t, int BLOCK_D>
__global__ void flash_attention_backward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ O,
    const scalar_t* __restrict__ dO,
    const float* __restrict__ L,
    scalar_t* __restrict__ dQ,
    scalar_t* __restrict__ dK,
    scalar_t* __restrict__ dV,
    int N,
    int d,
    int stride_n,
    bool is_causal
) {
    int batch_head = blockIdx.y;
    int tile_k = blockIdx.x;
    int tid = threadIdx.x;

    int k_start = tile_k * BK;

    const scalar_t* Q_bh = Q + batch_head * stride_n;
    const scalar_t* K_bh = K + batch_head * stride_n;
    const scalar_t* V_bh = V + batch_head * stride_n;
    const scalar_t* O_bh = O + batch_head * stride_n;
    const scalar_t* dO_bh = dO + batch_head * stride_n;
    const float* L_bh = L + batch_head * N;
    scalar_t* dQ_bh = dQ + batch_head * stride_n;
    scalar_t* dK_bh = dK + batch_head * stride_n;
    scalar_t* dV_bh = dV + batch_head * stride_n;

    extern __shared__ char smem_raw_bwd[];
    float* s_K = reinterpret_cast<float*>(smem_raw_bwd);
    float* s_V = s_K + BK * BLOCK_D;
    float* s_dK = s_V + BK * BLOCK_D;
    float* s_dV = s_dK + BK * BLOCK_D;

    for (int i = tid; i < BK * BLOCK_D; i += blockDim.x) {
        int row = i / BLOCK_D;
        int col = i % BLOCK_D;
        int gr = k_start + row;
        if (gr < N) {
            s_K[row * BLOCK_D + col] = static_cast<float>(K_bh[gr * d + col]);
            s_V[row * BLOCK_D + col] = static_cast<float>(V_bh[gr * d + col]);
        } else {
            s_K[row * BLOCK_D + col] = 0.0f;
            s_V[row * BLOCK_D + col] = 0.0f;
        }
        s_dK[i] = 0.0f;
        s_dV[i] = 0.0f;
    }
    __syncthreads();

    float scale = 1.0f / sqrtf(static_cast<float>(d));
    int num_q_tiles = (N + BQ - 1) / BQ;

    for (int tile_q = 0; tile_q < num_q_tiles; tile_q++) {
        int q_start = tile_q * BQ;
        if (is_causal && q_start > k_start + BK - 1) continue;
        if (is_causal && k_start > q_start + BQ - 1) continue;

        __shared__ float s_Q_bwd[BQ * 128];
        __shared__ float s_dO_bwd[BQ * 128];
        __shared__ float s_D[BQ];

        for (int i = tid; i < BQ * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D;
            int col = i % BLOCK_D;
            int gr = q_start + row;
            if (gr < N) {
                s_Q_bwd[row * BLOCK_D + col] = static_cast<float>(Q_bh[gr * d + col]);
                s_dO_bwd[row * BLOCK_D + col] = static_cast<float>(dO_bh[gr * d + col]);
            } else {
                s_Q_bwd[row * BLOCK_D + col] = 0.0f;
                s_dO_bwd[row * BLOCK_D + col] = 0.0f;
            }
        }

        for (int qr = tid; qr < BQ; qr += blockDim.x) {
            int gq = q_start + qr;
            float d_val = 0.0f;
            if (gq < N) {
                for (int dd = 0; dd < BLOCK_D; dd++) {
                    d_val += static_cast<float>(dO_bh[gq * d + dd]) * static_cast<float>(O_bh[gq * d + dd]);
                }
            }
            s_D[qr] = d_val;
        }
        __syncthreads();

        for (int idx = tid; idx < BQ * BK; idx += blockDim.x) {
            int qr = idx / BK;
            int kr = idx % BK;
            int gq = q_start + qr;
            int gk = k_start + kr;

            if (gq >= N || gk >= N) continue;
            if (is_causal && gk > gq) continue;

            float dot = 0.0f;
            for (int dd = 0; dd < BLOCK_D; dd++) {
                dot += s_Q_bwd[qr * BLOCK_D + dd] * s_K[kr * BLOCK_D + dd];
            }
            dot *= scale;

            float li = L_bh[gq];
            float p = expf(dot - li);

            float dp = 0.0f;
            for (int dd = 0; dd < BLOCK_D; dd++) {
                dp += s_dO_bwd[qr * BLOCK_D + dd] * s_V[kr * BLOCK_D + dd];
            }

            float ds = p * (dp - s_D[qr]) * scale;

            for (int dd = 0; dd < BLOCK_D; dd++) {
                atomicAdd(&s_dK[kr * BLOCK_D + dd], ds * s_Q_bwd[qr * BLOCK_D + dd]);
                atomicAdd(&s_dV[kr * BLOCK_D + dd], p * s_dO_bwd[qr * BLOCK_D + dd]);
                atomicAdd(&dQ_bh[gq * d + dd], static_cast<scalar_t>(ds * s_K[kr * BLOCK_D + dd]));
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < BK * BLOCK_D; i += blockDim.x) {
        int row = i / BLOCK_D;
        int gr = k_start + row;
        int col = i % BLOCK_D;
        if (gr < N) {
            atomicAdd(&dK_bh[gr * d + col], static_cast<scalar_t>(s_dK[i]));
            atomicAdd(&dV_bh[gr * d + col], static_cast<scalar_t>(s_dV[i]));
        }
    }
}

std::vector<torch::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);
    TORCH_CHECK(d <= 128, "Head dim must be <= 128");

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, H, N}, Q.options().dtype(torch::kFloat32));

    int BH = B * H;
    int num_q_tiles = (N + BQ - 1) / BQ;
    dim3 grid(num_q_tiles, BH);
    int threads = 128;

    int smem_size = (BQ * d + BK * d + BK * d + BQ * BK) * sizeof(float) + BQ * d * sizeof(float);

    int stride_n = N * d;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "flash_attn_fwd", ([&] {
        if (d <= 32) {
            flash_attention_forward_kernel<scalar_t, 32><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), L.data_ptr<float>(),
                N, d, stride_n, stride_n, stride_n, stride_n, is_causal);
        } else if (d <= 64) {
            flash_attention_forward_kernel<scalar_t, 64><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), L.data_ptr<float>(),
                N, d, stride_n, stride_n, stride_n, stride_n, is_causal);
        } else {
            flash_attention_forward_kernel<scalar_t, 128><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), L.data_ptr<float>(),
                N, d, stride_n, stride_n, stride_n, stride_n, is_causal);
        }
    }));

    return {O, L};
}

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    bool is_causal
) {
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    int BH = B * H;
    int num_k_tiles = (N + BK - 1) / BK;
    dim3 grid(num_k_tiles, BH);
    int threads = 128;
    int stride_n = N * d;

    int smem_size = (4 * BK * d + BQ * d * 2 + BQ) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "flash_attn_bwd", ([&] {
        if (d <= 32) {
            flash_attention_backward_kernel<scalar_t, 32><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(), L.data_ptr<float>(),
                dQ.data_ptr<scalar_t>(), dK.data_ptr<scalar_t>(), dV.data_ptr<scalar_t>(),
                N, d, stride_n, is_causal);
        } else if (d <= 64) {
            flash_attention_backward_kernel<scalar_t, 64><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(), L.data_ptr<float>(),
                dQ.data_ptr<scalar_t>(), dK.data_ptr<scalar_t>(), dV.data_ptr<scalar_t>(),
                N, d, stride_n, is_causal);
        } else {
            flash_attention_backward_kernel<scalar_t, 128><<<grid, threads, smem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(), L.data_ptr<float>(),
                dQ.data_ptr<scalar_t>(), dK.data_ptr<scalar_t>(), dV.data_ptr<scalar_t>(),
                N, d, stride_n, is_causal);
        }
    }));

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention 2 forward (CUDA)");
    m.def("backward", &flash_attention_backward, "Flash Attention 2 backward (CUDA)");
}
