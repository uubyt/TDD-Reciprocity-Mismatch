import numpy as np


def compute_ber(bits, bits_hat):
    return float(np.mean(bits != bits_hat))


def compute_ser(bits, bits_hat):
    symbol_error = np.any(bits != bits_hat, axis=-1)
    return float(np.mean(symbol_error))


def compute_bler(bits, bits_hat, block_symbols=1):
    """
    bits:     [B, Ns, T, Q]
    bits_hat: [B, Ns, T, Q]

    每 block_symbols 个 symbol 作为一个小 block。
    对每个 sample、每个 block，只要任意 stream / symbol / bit 出错，
    就认为该 block error。
    """
    B, Ns, T, Q = bits.shape

    num_blocks = T // block_symbols
    if num_blocks == 0:
        raise ValueError("T must be >= block_symbols.")

    T_use = num_blocks * block_symbols

    bits_use = bits[:, :, :T_use, :]
    bits_hat_use = bits_hat[:, :, :T_use, :]

    bits_blk = bits_use.reshape(B, Ns, num_blocks, block_symbols, Q)
    bits_hat_blk = bits_hat_use.reshape(B, Ns, num_blocks, block_symbols, Q)

    # block_error: [B, num_blocks]
    # 对每个小 block，只要任意 stream、任意 symbol、任意 bit 错，就算 block error
    block_error = np.any(bits_blk != bits_hat_blk, axis=(1, 3, 4))

    return float(np.mean(block_error))


def compute_evm(s, s_hat):
    err_power = np.mean(np.abs(s_hat - s) ** 2)
    sig_power = np.mean(np.abs(s) ** 2) + 1e-12
    return float(np.sqrt(err_power / sig_power))


def compute_effective_se(bler, Ns, bits_per_symbol):
    return float(Ns * bits_per_symbol * (1.0 - bler))


def gaussian_capacity_with_precoder(H_DL_true, W, dl_snr_linear):
    """Average Gaussian-input link capacity under a given precoder W."""
    G = H_DL_true @ W
    B, Nr, _ = H_DL_true.shape

    cap_list = []
    eye = np.eye(Nr, dtype=np.complex64)
    for b in range(B):
        Gb = G[b]
        A = eye + dl_snr_linear * (Gb @ Gb.conj().T)
        sign, logdet = np.linalg.slogdet(A)
        cap_list.append(np.real(logdet / np.log(2.0)))

    return float(np.mean(cap_list))


def nmse(H_hat, H_true):
    err = np.sum(np.abs(H_hat - H_true) ** 2, axis=(1, 2))
    power = np.sum(np.abs(H_true) ** 2, axis=(1, 2)) + 1e-12
    return float(np.mean(err / power))


def compute_gram_leakage(H_DL_true, W, eps=1e-12):
    """
    Measure inter-stream coupling after precoding.

    Effective channel:
        G = H_DL_true @ W,  [B, Nr, Ns]

    Gram matrix:
        R = G^H G,          [B, Ns, Ns]

    If streams are perfectly orthogonal, R is diagonal.
    We define:
        leakage = ||R - diag(R)||_F^2 / ||diag(R)||_F^2

    Returns:
        Average leakage over the batch.
    """
    H_DL_true = np.asarray(H_DL_true, dtype=np.complex64)
    W = np.asarray(W, dtype=np.complex64)

    G = H_DL_true @ W  # [B, Nr, Ns]

    R = np.matmul(
        np.conjugate(np.transpose(G, (0, 2, 1))),
        G,
    )  # [B, Ns, Ns]

    B, Ns, _ = R.shape

    R_diag = np.zeros_like(R)
    idx = np.arange(Ns)
    R_diag[:, idx, idx] = np.diagonal(R, axis1=1, axis2=2)

    R_off = R - R_diag

    off_power = np.sum(np.abs(R_off) ** 2, axis=(1, 2))
    diag_power = np.sum(np.abs(R_diag) ** 2, axis=(1, 2)) + eps

    leakage = off_power / diag_power
    return float(np.mean(leakage))


def compute_post_mmse_sinr_stats(H_DL_true, W, noise_var, eps=1e-12):
    """
    Compute post-MMSE equivalent SINR statistics.

    Effective channel:
        G = H_DL_true @ W

    MMSE error covariance:
        E = (I + (1/sigma^2) G^H G)^(-1)

    Post-MMSE SINR of stream i:
        gamma_i = 1 / E_ii - 1

    Returns:
        mean_sinr_db:
            Mean post-MMSE SINR averaged over all streams and samples.
        min_sinr_db:
            For each sample, take the weakest stream SINR;
            then average over the batch.
    """
    H_DL_true = np.asarray(H_DL_true, dtype=np.complex64)
    W = np.asarray(W, dtype=np.complex64)

    G = H_DL_true @ W  # [B, Nr, Ns]

    R = np.matmul(
        np.conjugate(np.transpose(G, (0, 2, 1))),
        G,
    )  # [B, Ns, Ns]

    B, Ns, _ = R.shape
    I = np.eye(Ns, dtype=np.complex64)[None, :, :]

    noise_var = float(noise_var)
    E = np.linalg.inv(
        I + R / (noise_var + eps)
    )  # [B, Ns, Ns]

    E_diag = np.real(np.diagonal(E, axis1=1, axis2=2))
    E_diag = np.maximum(E_diag, eps)

    sinr_linear = 1.0 / E_diag - 1.0
    sinr_linear = np.maximum(sinr_linear, eps)

    sinr_db = 10.0 * np.log10(sinr_linear)

    mean_sinr_db = float(np.mean(sinr_db))
    min_sinr_db = float(np.mean(np.min(sinr_db, axis=1)))

    return mean_sinr_db, min_sinr_db