import numpy as np


def downlink_transmit(H_DL_true, W, s, dl_snr_db, tx_power=1.0):
    """
    H_DL_true: [B, Nr, Nt]
    W:         [B, Nt, Ns]
    s:         [B, Ns, T]
    return y: [B, Nr, T], noise_var: float
    """
    x = W @ s
    y_clean = H_DL_true @ x

    snr_linear = 10.0 ** (dl_snr_db / 10.0)
    noise_var = tx_power / (snr_linear + 1e-12)

    noise = (
        np.random.randn(*y_clean.shape)
        + 1j * np.random.randn(*y_clean.shape)
    ).astype(np.complex64) * np.sqrt(noise_var / 2.0)

    return (y_clean + noise).astype(np.complex64), float(noise_var)


def mmse_equalize(y, H_DL_true, W, noise_var):
    """
    y:         [B, Nr, T]
    H_DL_true: [B, Nr, Nt]
    W:         [B, Nt, Ns]
    return s_hat: [B, Ns, T]
    """
    G = H_DL_true @ W
    B = G.shape[0]
    Ns = G.shape[2]

    s_hat_list = []
    eye = np.eye(Ns, dtype=np.complex64)

    for b in range(B):
        Gb = G[b]
        yb = y[b]
        A = Gb.conj().T @ Gb + noise_var * eye
        rhs = Gb.conj().T @ yb
        s_hat_b = np.linalg.solve(A, rhs)
        s_hat_list.append(s_hat_b.astype(np.complex64))

    return np.stack(s_hat_list, axis=0).astype(np.complex64)
