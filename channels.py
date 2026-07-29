import os
import numpy as np


def load_and_split_complex_data(data_path, train_ratio, val_ratio, test_ratio, random_state=42):
    """
    Load uplink channel dataset.

    Supports either complex ndarray [N, Nt, Nr] or real-imag stacked ndarray
    [N, Nt, Nr, 2].
    """
    H = np.load(data_path)

    if H.ndim == 4 and H.shape[-1] == 2:
        H = H[..., 0] + 1j * H[..., 1]

    H = H.astype(np.complex64)

    num_samples = H.shape[0]
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(num_samples)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = H[train_idx]
    val_data = H[val_idx]
    test_data = H[test_idx]

    info = {
        "num_samples": num_samples,
        "Nt": H.shape[1],
        "Nr": H.shape[2],
    }

    return train_data, val_data, test_data, info


def generate_fixed_mismatch(
    Nt,
    Nr,
    std_amp_db,
    std_phase_deg,
    power_neutral=True,
    seed=42,
    save_path=None,
    force_regenerate=False,
):
    """Generate or load fixed diagonal reciprocity mismatch matrices."""
    if save_path is not None and os.path.exists(save_path) and not force_regenerate:
        data = np.load(save_path)
        print(f"Loaded fixed mismatch from: {save_path}")
        return data["C_BS"].astype(np.complex64), data["C_UE"].astype(np.complex64)

    rng = np.random.default_rng(seed)

    amp_bs_db = rng.normal(0.0, std_amp_db, Nt)
    amp_ue_db = rng.normal(0.0, std_amp_db, Nr)

    amp_bs = 10.0 ** (amp_bs_db / 20.0)
    amp_ue = 10.0 ** (amp_ue_db / 20.0)

    if power_neutral:
        amp_bs = amp_bs / np.sqrt(np.mean(amp_bs ** 2) + 1e-12)
        amp_ue = amp_ue / np.sqrt(np.mean(amp_ue ** 2) + 1e-12)

    phase_bs = rng.normal(0.0, np.deg2rad(std_phase_deg), Nt)
    phase_ue = rng.normal(0.0, np.deg2rad(std_phase_deg), Nr)

    c_bs = amp_bs * np.exp(1j * phase_bs)
    c_ue = amp_ue * np.exp(1j * phase_ue)

    C_BS = np.diag(c_bs).astype(np.complex64)
    C_UE = np.diag(c_ue).astype(np.complex64)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            C_BS=C_BS,
            C_UE=C_UE,
            std_amp_db=std_amp_db,
            std_phase_deg=std_phase_deg,
            power_neutral=power_neutral,
            seed=seed,
        )
        print(f"Saved fixed mismatch to: {save_path}")

    return C_BS, C_UE


def build_downlink_from_uplink(H_UL, C_BS, C_UE):
    """
    H_UL: [B, Nt, Nr]
    C_BS: [Nt, Nt]
    C_UE: [Nr, Nr]

    return:
        H_DL: [B, Nr, Nt]
    """
    H_UL = H_UL.astype(np.complex64)
    H_UL_T = np.transpose(H_UL, (0, 2, 1))
    H_DL = C_UE[None, :, :] @ H_UL_T @ C_BS[None, :, :]
    return H_DL.astype(np.complex64)


def add_awgn_to_uplink(H_UL_clean, csi_snr_db):
    """
    H_UL_clean: [B, Nt, Nr]
    csi_snr_db: float or [B]

    return:
        H_UL_est: [B, Nt, Nr]
    """
    H_UL_clean = H_UL_clean.astype(np.complex64)
    B = H_UL_clean.shape[0]

    if np.isscalar(csi_snr_db):
        csi_snr_db = np.ones(B, dtype=np.float32) * float(csi_snr_db)
    else:
        csi_snr_db = np.asarray(csi_snr_db, dtype=np.float32)

    snr_linear = 10.0 ** (csi_snr_db / 10.0)
    signal_power = np.mean(np.abs(H_UL_clean) ** 2, axis=(1, 2), keepdims=True)
    noise_var = signal_power / snr_linear[:, None, None]

    noise = (
        np.random.randn(*H_UL_clean.shape)
        + 1j * np.random.randn(*H_UL_clean.shape)
    ).astype(np.complex64) * np.sqrt(noise_var / 2.0)

    return (H_UL_clean + noise).astype(np.complex64)
