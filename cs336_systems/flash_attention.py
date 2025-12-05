# Pure PyTorch FA2; tiled, online softmax.
import math
import torch

Q_TILE_SIZE = 32
K_TILE_SIZE = 32


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, n_queries, d = Q.shape
        _, n_keys, _ = K.shape
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)
        m = torch.full((batch, n_queries), float("-inf"), device=Q.device, dtype=Q.dtype)
        l = torch.zeros((batch, n_queries), device=Q.device, dtype=Q.dtype)

        n_q_tiles = math.ceil(n_queries / Q_TILE_SIZE)
        n_k_tiles = math.ceil(n_keys / K_TILE_SIZE)

        for i in range(n_q_tiles):
            q_start = i * Q_TILE_SIZE
            q_end = min(q_start + Q_TILE_SIZE, n_queries)

            Q_tile = Q[:, q_start:q_end, :]
            O_tile = O[:, q_start:q_end, :]
            m_tile = m[:, q_start:q_end]
            l_tile = l[:, q_start:q_end]

            for j in range(n_k_tiles):
                k_start = j * K_TILE_SIZE
                k_end = min(k_start + K_TILE_SIZE, n_keys)

                K_tile = K[:, k_start:k_end, :]
                V_tile = V[:, k_start:k_end, :]

                S_tile = (Q_tile @ K_tile.transpose(-2, -1)) * scale

                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                    causal_mask = q_idx < k_idx
                    S_tile = S_tile.masked_fill(causal_mask.unsqueeze(0), -1e6)

                m_tile_new = torch.maximum(m_tile, S_tile.max(dim=-1).values)
                exp_old = torch.exp(m_tile - m_tile_new)
                exp_s = torch.exp(S_tile - m_tile_new.unsqueeze(-1))

                l_tile_new = exp_old * l_tile + exp_s.sum(dim=-1)
                O_tile = O_tile * (exp_old * l_tile / l_tile_new).unsqueeze(-1) + \
                    (exp_s / l_tile_new.unsqueeze(-1)) @ V_tile

                m_tile = m_tile_new
                l_tile = l_tile_new

            O[:, q_start:q_end, :] = O_tile
            m[:, q_start:q_end] = m_tile
            l[:, q_start:q_end] = l_tile

        L = m + torch.log(l)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch, n_queries, d = Q.shape
        _, n_keys, _ = K.shape
        scale = 1.0 / math.sqrt(d)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        D = (O * dO).sum(dim=-1)

        n_q_tiles = math.ceil(n_queries / Q_TILE_SIZE)
        n_k_tiles = math.ceil(n_keys / K_TILE_SIZE)

        for j in range(n_k_tiles):
            k_start = j * K_TILE_SIZE
            k_end = min(k_start + K_TILE_SIZE, n_keys)

            K_tile = K[:, k_start:k_end, :]
            V_tile = V[:, k_start:k_end, :]
            dK_tile = dK[:, k_start:k_end, :]
            dV_tile = dV[:, k_start:k_end, :]

            for i in range(n_q_tiles):
                q_start = i * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, n_queries)

                Q_tile = Q[:, q_start:q_end, :]
                O_tile = O[:, q_start:q_end, :]
                dO_tile = dO[:, q_start:q_end, :]
                L_tile = L[:, q_start:q_end]
                D_tile = D[:, q_start:q_end]

                S_tile = (Q_tile @ K_tile.transpose(-2, -1)) * scale

                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                    causal_mask = q_idx < k_idx
                    S_tile = S_tile.masked_fill(causal_mask.unsqueeze(0), -1e6)

                P_tile = torch.exp(S_tile - L_tile.unsqueeze(-1))

                dP_tile = dO_tile @ V_tile.transpose(-2, -1)
                dS_tile = P_tile * (dP_tile - D_tile.unsqueeze(-1))

                dQ[:, q_start:q_end, :] += (dS_tile @ K_tile) * scale
                dK_tile += (dS_tile.transpose(-2, -1) @ Q_tile) * scale
                dV_tile += P_tile.transpose(-2, -1) @ dO_tile

            dK[:, k_start:k_end, :] = dK_tile
            dV[:, k_start:k_end, :] = dV_tile

        return dQ, dK, dV, None


def flash_attention_pytorch(Q, K, V, is_causal=False):
    return FlashAttentionPyTorch.apply(Q, K, V, is_causal)
