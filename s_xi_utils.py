import os
import json
import numpy as np


def make_xi_key(std_amp_db, std_phase_deg):
    return f"A{float(std_amp_db):.1f}_P{float(std_phase_deg):.1f}"


def load_s_xi_from_table(
    table_path,
    std_amp_db,
    std_phase_deg,
    csi_snr_db=None,
    use_key="s_xi_mean",
    default=None,
):
    """
    Load s_xi or s_{xi,gamma} from s_xi_table.json.

    If csi_snr_db is None:
        return overall s_xi.
    If csi_snr_db is given:
        return per-SNR s_{xi,gamma}, with linear interpolation.
    """

    xi_key = make_xi_key(std_amp_db, std_phase_deg)

    if not os.path.exists(table_path):
        if default is not None:
            print(f"[s_xi] Table not found: {table_path}")
            print(f"[s_xi] Use default s_xi = {default}")
            return float(default)
        raise FileNotFoundError(f"s_xi table not found: {table_path}")

    with open(table_path, "r") as f:
        table = json.load(f)

    if xi_key not in table:
        if default is not None:
            print(f"[s_xi] Key {xi_key} not found in table.")
            print(f"[s_xi] Use default s_xi = {default}")
            return float(default)
        raise KeyError(
            f"Cannot find key {xi_key} in {table_path}. "
            f"Available keys: {list(table.keys())}"
        )

    item = table[xi_key]

    # --------------------------------------------------------
    # Case 1: no SNR-specific query, use overall s_xi
    # --------------------------------------------------------
    if csi_snr_db is None or "per_snr" not in item:
        if use_key not in item:
            if default is not None:
                return float(default)
            raise KeyError(
                f"Cannot find {use_key} under {xi_key}. "
                f"Available keys: {list(item.keys())}"
            )

        s_xi = float(item[use_key])
        print(
            f"[s_xi] Loaded overall {use_key}={s_xi:.6f} "
            f"for {xi_key}"
        )
        return s_xi

    # --------------------------------------------------------
    # Case 2: use per-SNR s_{xi,gamma}
    # --------------------------------------------------------
    per_snr = item["per_snr"]

    snr_values = sorted([float(k) for k in per_snr.keys()])
    s_values = []

    for snr in snr_values:
        snr_key = f"{snr:.1f}"
        if use_key not in per_snr[snr_key]:
            raise KeyError(
                f"Cannot find {use_key} under {xi_key}/per_snr/{snr_key}."
            )
        s_values.append(float(per_snr[snr_key][use_key]))

    snr_values = np.asarray(snr_values, dtype=float)
    s_values = np.asarray(s_values, dtype=float)

    csi_snr_db = float(csi_snr_db)

    # Linear interpolation; outside range uses boundary value
    s_xi_gamma = float(
        np.interp(
            csi_snr_db,
            snr_values,
            s_values,
            left=s_values[0],
            right=s_values[-1],
        )
    )

    print(
        f"[s_xi] Loaded/interpolated {use_key}={s_xi_gamma:.6f} "
        f"for {xi_key}, CSI SNR={csi_snr_db:.1f} dB"
    )

    return s_xi_gamma