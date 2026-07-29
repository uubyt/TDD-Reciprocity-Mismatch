# 首先运行  计算误差场景的s_xi表

import os
import json
import numpy as np

from config_system import (
    DATA_CONFIG,
    PATH_CONFIG,
    SNR_CONFIG,
)

from channels import (
    load_and_split_complex_data,
    generate_fixed_mismatch,
    build_downlink_from_uplink,
    add_awgn_to_uplink,
)


# ============================================================
# 你要统计的误差场景全部写在这里
# ============================================================

MISMATCH_GRID = [
    (2.0, 20.0),
    (2.0, 40.0),
    (2.0, 60.0),
    (2.0, 90.0),
    (4.0, 20.0),
    (4.0, 40.0),
    (4.0, 60.0),
    (4.0, 90.0),
]


POWER_NEUTRAL = True
USE_SHRINKAGE = True
SEED = 42
EPS = 1e-12


def make_xi_key(std_amp_db, std_phase_deg):
    return f"A{float(std_amp_db):.1f}_P{float(std_phase_deg):.1f}"


def compute_s_xi_for_one_csi_snr(
    H_UL_clean,
    C_BS,
    C_UE,
    csi_snr_db,
    use_shrinkage=True,
    eps=1e-12,
):
    """
    H_UL_clean: [B, Nt, Nr] complex
    C_BS: [Nt, Nt] complex
    C_UE: [Nr, Nr] complex
    csi_snr_db: scalar
    """

    # 1. True downlink channel
    H_DL_true = build_downlink_from_uplink(
        H_UL_clean,
        C_BS,
        C_UE,
    )  # [B, Nr, Nt]

    # 2. Noisy uplink CSI
    H_UL_est = add_awgn_to_uplink(
        H_UL_clean,
        csi_snr_db,
    )  # [B, Nt, Nr]

    # 3. Optional SNR-aware shrinkage
    if use_shrinkage:
        rho = 10.0 ** (float(csi_snr_db) / 10.0)
        alpha = rho / (1.0 + rho)
        H_UL_tilde = alpha * H_UL_est
    else:
        H_UL_tilde = H_UL_est

    # 4. Coarse downlink estimate
    H0 = np.transpose(H_UL_tilde, (0, 2, 1))  # [B, Nr, Nt]

    # 5. Residual
    Delta = H_DL_true - H0

    # 6. RMS amplitudes
    h0_rms = np.sqrt(
        np.mean(np.abs(H0) ** 2, axis=(1, 2)) + eps
    )

    delta_rms = np.sqrt(
        np.mean(np.abs(Delta) ** 2, axis=(1, 2)) + eps
    )

    # 7. Relative residual scale
    r = delta_rms / (h0_rms + eps)

    return r


def main():
    print("Loading dataset...")

    train_data, val_data, test_data, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]

    print("Train:", train_data.shape)
    print(f"Nt={Nt}, Nr={Nr}")

    # --------------------------------------------------------
    # CSI SNR list
    # --------------------------------------------------------
    if SNR_CONFIG["csi_snr_mode"] == "same_as_dl":
        csi_snr_list = SNR_CONFIG["dl_snr_dB_list"]
    elif SNR_CONFIG["csi_snr_mode"] == "fixed":
        csi_snr_list = [SNR_CONFIG["fixed_csi_snr_db"]]
    else:
        raise ValueError(f"Unknown csi_snr_mode: {SNR_CONFIG['csi_snr_mode']}")

    print("CSI SNR list:", csi_snr_list)

    s_xi_table = {}

    # --------------------------------------------------------
    # Loop over all mismatch scenarios
    # --------------------------------------------------------
    for std_amp_db, std_phase_deg in MISMATCH_GRID:
        xi_key = make_xi_key(std_amp_db, std_phase_deg)

        print("\n========================================")
        print(f"Computing s_xi for {xi_key}")
        print(f"std_amp_db    = {std_amp_db}")
        print(f"std_phase_deg = {std_phase_deg}")
        print("========================================")

        # 注意：每个误差场景必须用不同的 mismatch_path
        mismatch_path = os.path.join(
            PATH_CONFIG["results_dir"],
            f"fixed_mismatch_A{std_amp_db:.1f}dB_P{std_phase_deg:.1f}deg.npz",
        )

        C_BS, C_UE = generate_fixed_mismatch(
            Nt=Nt,
            Nr=Nr,
            std_amp_db=std_amp_db,
            std_phase_deg=std_phase_deg,
            power_neutral=POWER_NEUTRAL,
            seed=SEED,
            save_path=mismatch_path,
            force_regenerate=True,
        )

        all_r = []
        per_snr_table = {}

        for csi_snr_db in csi_snr_list:
            print(f"  CSI SNR = {csi_snr_db} dB")

            r = compute_s_xi_for_one_csi_snr(
                H_UL_clean=train_data,
                C_BS=C_BS,
                C_UE=C_UE,
                csi_snr_db=csi_snr_db,
                use_shrinkage=USE_SHRINKAGE,
                eps=EPS,
            )

            snr_key = f"{float(csi_snr_db):.1f}"

            per_snr_table[snr_key] = {
                "s_xi_mean": float(np.mean(r)),
                "s_xi_median": float(np.median(r)),
                "s_xi_std": float(np.std(r)),
            }

            all_r.append(r)

        all_r = np.concatenate(all_r, axis=0)

        s_xi_table[xi_key] = {
            "std_amp_db": float(std_amp_db),
            "std_phase_deg": float(std_phase_deg),
            "power_neutral": bool(POWER_NEUTRAL),
            "use_shrinkage": bool(USE_SHRINKAGE),
            "csi_snr_mode": SNR_CONFIG["csi_snr_mode"],
            "csi_snr_list": list(map(float, csi_snr_list)),

            # 原来的整体平均，保留作为 fallback
            "s_xi_mean": float(np.mean(all_r)),
            "s_xi_median": float(np.median(all_r)),
            "s_xi_std": float(np.std(all_r)),

            # 新增：每个 CSI SNR 单独统计
            "per_snr": per_snr_table,
        }



    # --------------------------------------------------------
    # Save unified table
    # --------------------------------------------------------
    save_path = os.path.join(
        PATH_CONFIG["results_dir"],
        "s_xi_table.json",
    )

    with open(save_path, "w") as f:
        json.dump(s_xi_table, f, indent=2)

    print("\nSaved s_xi table to:")
    print(save_path)


if __name__ == "__main__":
    main()