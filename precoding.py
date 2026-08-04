import numpy as np


def normalize_precoder(W):
    W = W.astype(np.complex64)
    power = np.sum(np.abs(W) ** 2, axis=(1, 2), keepdims=True)
    return (W / np.sqrt(power + 1e-12)).astype(np.complex64)

def normalize_precoder_stream(W):
    """
    W: [B, Nt, Ns]
    Force each stream/column to have power 1/Ns.
    """
    W = W.astype(np.complex64)
    B, Nt, Ns = W.shape

    col_power = np.sum(np.abs(W) ** 2, axis=1, keepdims=True)  # [B,1,Ns]
    W = W / np.sqrt(col_power + 1e-12)
    W = W / np.sqrt(Ns)

    return W.astype(np.complex64)


def svd_power_precoder(H, Ns):
    """
    H: [B, Nr, Nt]
    return W: [B, Nt, Ns], total power normalized to 1.
    """
    H = H.astype(np.complex64)
    _, _, Vh = np.linalg.svd(H, full_matrices=False)
    V = np.conjugate(np.swapaxes(Vh, -1, -2))
    W = V[:, :, :Ns]
    return normalize_precoder(W)


def water_filling_power(singular_values, snr_linear, total_power=1.0):
    """Water-filling over singular modes."""
    gains = np.maximum(singular_values ** 2, 1e-12)
    inv_gains = 1.0 / (snr_linear * gains)

    order = np.argsort(inv_gains)
    inv_sorted = inv_gains[order]

    Ns = len(gains)
    mu = None
    for k in range(1, Ns + 1):
        mu_k = (total_power + np.sum(inv_sorted[:k])) / k
        if k == Ns or mu_k <= inv_sorted[k]:
            mu = mu_k
            break

    p_sorted = np.maximum(mu - inv_sorted, 0.0)
    p = np.zeros_like(p_sorted)
    p[order] = p_sorted
    p = p * (total_power / (np.sum(p) + 1e-12))
    return p


def svd_waterfilling_precoder(H, Ns, snr_linear):
    """
    H: [B, Nr, Nt]
    return W: [B, Nt, Ns].

    Better aligned with Gaussian capacity than fixed 4QAM BER/BLER.
    """
    H = H.astype(np.complex64)
    _, S, Vh = np.linalg.svd(H, full_matrices=False)
    V = np.conjugate(np.swapaxes(Vh, -1, -2))

    W_list = []
    for b in range(H.shape[0]):
        svals = S[b, :Ns]
        p = water_filling_power(svals, snr_linear, total_power=1.0)
        Wb = V[b, :, :Ns] * np.sqrt(p[None, :] + 1e-12)
        W_list.append(Wb.astype(np.complex64))

    W = np.stack(W_list, axis=0)
    return normalize_precoder(W)


def build_precoder(H_prec, Ns, precoder_type="svd_waterfilling", snr_linear=None):
    if precoder_type == "svd_waterfilling":
        return svd_power_precoder(H_prec, Ns)

    if precoder_type == "svd_waterfilling":
        if snr_linear is None:
            raise ValueError("snr_linear must be provided for svd_waterfilling.")
        return svd_waterfilling_precoder(H_prec, Ns, snr_linear)

    raise ValueError(f"Unknown precoder_type: {precoder_type}")
