# sa_mult_calibration.py

import os
import json
import numpy as np
import tensorflow as tf

from sa_mult_cal_net import SAMultCalNet, complex_to_ri_np
from config_system import SA_RESCAL_CONFIG, TEST_MISMATCH_CONFIG


def make_xi_key(std_amp_db, std_phase_deg):
    return f"A{float(std_amp_db):.1f}_P{float(std_phase_deg):.1f}"

def ri_to_complex_np(H_ri):
    H_ri = np.asarray(H_ri)
    return (H_ri[..., 0] + 1j * H_ri[..., 1]).astype(np.complex64)

class SAMultCalibrator:
    """
    Wrapper for SA-Mult-Cal variants.

    It supports:
        sa_mult_full_svd
        sa_mult_nores_svd
        sa_mult_nores_nocond_svd

    Input:
        H_UL_est: [B, Nt, Nr] complex numpy
        csi_snr_db: scalar float

    Output:
        H_DL_hat: [B, Nr, Nt] complex numpy
    """

    def __init__(
        self,
        Nt,
        Nr,
        config=None,
        weights_path=None,
        s_xi=None,
    ):
        self.Nt = Nt
        self.Nr = Nr

        # ----------------------------------------------------
        # Use variant config if provided.
        # Otherwise fall back to SA_RESCAL_CONFIG.
        # ----------------------------------------------------
        self.config = dict(SA_RESCAL_CONFIG)

        if config is not None:
            self.config.update(config)

        cfg = self.config

        self.weights_path = weights_path or cfg["weights_path"]

        # If s_xi is manually provided, use it directly.
        # Otherwise, load it from s_xi_table.json.
        self.fixed_s_xi = None if s_xi is None else float(s_xi)

        self.table_path = cfg.get(
            "s_xi_table_path",
            "./results_e2e/s_xi_table.json",
        )
        self.s_xi_key = cfg.get("s_xi_key", "s_xi_mean")
        self.s_xi_default = float(cfg.get("s_xi_default", 0.4))

        if self.fixed_s_xi is not None:
            print(f"[SA-MultCal] Use manually fixed s_xi = {self.fixed_s_xi:.6f}")
            self.s_xi_table = None
        else:
            self.s_xi_table = self._load_s_xi_table()

        print("\n[SA-MultCal] Initialize calibrator")
        print(f"  weights_path   = {self.weights_path}")
        print(f"  eta_add        = {cfg.get('eta_add', 0.0)}")
        print(f"  condition_mode = {cfg.get('condition_mode', 'full')}")
        print(f"  s_xi_key       = {self.s_xi_key}")
        print(f"  s_xi_default   = {self.s_xi_default:.6f}")

        # ----------------------------------------------------
        # Build model according to the selected variant config.
        # ----------------------------------------------------
        self.model = SAMultCalNet(
            Nt=Nt,
            Nr=Nr,
            feature_dim=cfg.get("feature_dim", 64),
            num_blocks=cfg.get("num_blocks", 3),
            cond_dim=cfg.get("cond_dim", 64),
            snr_min_db=cfg.get("snr_min_db", 0.0),
            snr_max_db=cfg.get("snr_max_db", 26.0),
            s_max=cfg.get("s_max", 1.5),
            amp_min_db=cfg.get("amp_min_db", 2.0),
            amp_max_db=cfg.get("amp_max_db", 4.0),
            phase_min_deg=cfg.get("phase_min_deg", 20.0),
            phase_max_deg=cfg.get("phase_max_deg", 90.0),
            amp_clip=cfg.get("amp_clip", 0.5),
            phase_clip=cfg.get("phase_clip", 1.0),
            delta_clip=cfg.get("delta_clip", 3.0),
            eta_add=cfg.get("eta_add", 0.0),
            use_shrinkage=cfg.get("use_shrinkage", True),
            condition_mode=cfg.get("condition_mode", "full"),
        )

        # Build model before loading weights.
        dummy = tf.zeros([1, Nt, Nr, 2], dtype=tf.float32)

        _ = self.model(
            dummy,
            csi_snr_db=tf.constant([10.0], dtype=tf.float32),
            s_xi=tf.constant([self.s_xi_default], dtype=tf.float32),
            std_amp_db=tf.constant(
                [float(TEST_MISMATCH_CONFIG["std_amp_db"])],
                dtype=tf.float32,
            ),
            std_phase_deg=tf.constant(
                [float(TEST_MISMATCH_CONFIG["std_phase_deg"])],
                dtype=tf.float32,
            ),
            training=False,
            return_aux=False,
        )

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"SA-MultCal weights not found: {self.weights_path}"
            )

        print(f"[SA-MultCal] Loading weights from: {self.weights_path}")
        self.model.load_weights(self.weights_path)
        print("[SA-MultCal] Loaded successfully.")

    def _load_s_xi_table(self):
        if not os.path.exists(self.table_path):
            print(f"[SA-ResCalNet] s_xi table not found: {self.table_path}")
            print(f"[SA-ResCalNet] Use default s_xi = {self.s_xi_default:.6f}")
            return None

        with open(self.table_path, "r") as f:
            table = json.load(f)

        print(f"[SA-ResCalNet] Loaded s_xi table from: {self.table_path}")
        return table

    def _get_s_xi(self, csi_snr_db):
        """
        Return the original s_xi only.

        Important:
            Do not multiply s_xi_temp here.
            residual_gain controls the residual strength after model forward.
        """

        if self.fixed_s_xi is not None:
            return float(self.fixed_s_xi)

        if self.s_xi_table is None:
            return float(self.s_xi_default)

        std_amp_db = float(TEST_MISMATCH_CONFIG["std_amp_db"])
        std_phase_deg = float(TEST_MISMATCH_CONFIG["std_phase_deg"])

        xi_key = make_xi_key(std_amp_db, std_phase_deg)

        if xi_key not in self.s_xi_table:
            return self._interpolate_s_xi(std_amp_db, std_phase_deg, csi_snr_db)

        item = self.s_xi_table[xi_key]

        # Per-SNR table format:
        # item["per_snr"]["10.0"]["s_xi_mean"]
        if "per_snr" in item:
            per_snr = item["per_snr"]

            snr_values = sorted([float(k) for k in per_snr.keys()])
            s_values = []

            for snr in snr_values:
                snr_key = f"{snr:.1f}"

                if snr_key not in per_snr:
                    snr_key = str(snr)

                if self.s_xi_key not in per_snr[snr_key]:
                    print(
                        f"[SA-ResCalNet] {self.s_xi_key} not found in "
                        f"per_snr[{snr_key}] of {xi_key}. "
                        f"Use default s_xi = {self.s_xi_default:.6f}"
                    )
                    return float(self.s_xi_default)

                s_values.append(float(per_snr[snr_key][self.s_xi_key]))

            s_xi = float(
                np.interp(
                    float(csi_snr_db),
                    np.asarray(snr_values, dtype=np.float32),
                    np.asarray(s_values, dtype=np.float32),
                    left=s_values[0],
                    right=s_values[-1],
                )
            )

            return float(s_xi)

        # Overall table format:
        # item["s_xi_mean"]
        if self.s_xi_key not in item:
            print(
                f"[SA-ResCalNet] {self.s_xi_key} not found in {xi_key}. "
                f"Use default s_xi = {self.s_xi_default:.6f}"
            )
            return float(self.s_xi_default)

        s_xi = float(item[self.s_xi_key])
        return float(s_xi)
    
    def _read_s_xi_from_item(self, item, csi_snr_db):
        """
        Read s_xi from one table item.
        Supports both overall format and per-SNR format.
        """
        if item is None:
            return None

        # Per-SNR format
        if "per_snr" in item:
            per_snr = item["per_snr"]

            snr_values = sorted([float(k) for k in per_snr.keys()])
            s_values = []

            for snr in snr_values:
                snr_key = f"{snr:.1f}"
                if snr_key not in per_snr:
                    snr_key = str(snr)

                if self.s_xi_key not in per_snr[snr_key]:
                    return None

                s_values.append(float(per_snr[snr_key][self.s_xi_key]))

            return float(
                np.interp(
                    float(csi_snr_db),
                    np.asarray(snr_values, dtype=np.float32),
                    np.asarray(s_values, dtype=np.float32),
                    left=s_values[0],
                    right=s_values[-1],
                )
            )

        # Overall format
        if self.s_xi_key in item:
            return float(item[self.s_xi_key])

        return None
    
    def _interpolate_s_xi(self, std_amp_db, std_phase_deg, csi_snr_db):
        """
        Bilinear interpolation of s_xi over available (amp, phase) grid.
        If the query is outside the grid, clamp to boundary.
        """
        if self.s_xi_table is None:
            print(f"[SA-MultCal] s_xi table is None. Use default {self.s_xi_default:.6f}")
            return float(self.s_xi_default)

        # Parse available grid points
        points = []
        for key in self.s_xi_table.keys():
            # Expected key format: A4.0_P90.0
            try:
                a_str, p_str = key.split("_")
                amp = float(a_str[1:])
                phase = float(p_str[1:])
                points.append((amp, phase, key))
            except Exception:
                continue

        if len(points) == 0:
            print(f"[SA-MultCal] No valid s_xi grid points. Use default {self.s_xi_default:.6f}")
            return float(self.s_xi_default)

        amps = sorted(list(set([p[0] for p in points])))
        phases = sorted(list(set([p[1] for p in points])))

        amp_q = float(std_amp_db)
        phase_q = float(std_phase_deg)

        # Clamp query to grid range
        amp_c = float(np.clip(amp_q, amps[0], amps[-1]))
        phase_c = float(np.clip(phase_q, phases[0], phases[-1]))

        # Find lower/upper amp
        amp_lo = max([a for a in amps if a <= amp_c])
        amp_hi = min([a for a in amps if a >= amp_c])

        # Find lower/upper phase
        phase_lo = max([p for p in phases if p <= phase_c])
        phase_hi = min([p for p in phases if p >= phase_c])

        def get_value(a, p):
            key = make_xi_key(a, p)
            if key not in self.s_xi_table:
                return None
            return self._read_s_xi_from_item(
                self.s_xi_table[key],
                csi_snr_db=csi_snr_db,
            )

        v_ll = get_value(amp_lo, phase_lo)
        v_lh = get_value(amp_lo, phase_hi)
        v_hl = get_value(amp_hi, phase_lo)
        v_hh = get_value(amp_hi, phase_hi)

        values = [v_ll, v_lh, v_hl, v_hh]

        # If some corner is missing, fall back to nearest valid point
        if any(v is None for v in values):
            nearest_key = None
            nearest_dist = float("inf")

            for amp, phase, key in points:
                dist = (amp - amp_c) ** 2 + (phase - phase_c) ** 2
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_key = key

            s_near = self._read_s_xi_from_item(
                self.s_xi_table[nearest_key],
                csi_snr_db=csi_snr_db,
            )

            if s_near is None:
                print(f"[SA-MultCal] Nearest s_xi invalid. Use default {self.s_xi_default:.6f}")
                return float(self.s_xi_default)

            print(
                f"[SA-MultCal] Missing interpolation corner for "
                f"A{amp_q:.1f}_P{phase_q:.1f}. "
                f"Use nearest {nearest_key}, s_xi={s_near:.6f}"
            )
            return float(s_near)

        # Degenerate case: exact amp and phase
        if amp_hi == amp_lo and phase_hi == phase_lo:
            return float(v_ll)

        # Linear interpolation along phase only
        if amp_hi == amp_lo:
            t = 0.0 if phase_hi == phase_lo else (phase_c - phase_lo) / (phase_hi - phase_lo)
            s = (1.0 - t) * v_ll + t * v_lh
            print(
                f"[SA-MultCal] Interpolate s_xi along phase: "
                f"A{amp_lo:.1f}, P{phase_lo:.1f}->{phase_hi:.1f}, "
                f"query P{phase_q:.1f}, s_xi={s:.6f}"
            )
            return float(s)

        # Linear interpolation along amp only
        if phase_hi == phase_lo:
            t = (amp_c - amp_lo) / (amp_hi - amp_lo)
            s = (1.0 - t) * v_ll + t * v_hl
            print(
                f"[SA-MultCal] Interpolate s_xi along amp: "
                f"A{amp_lo:.1f}->{amp_hi:.1f}, P{phase_lo:.1f}, "
                f"query A{amp_q:.1f}, s_xi={s:.6f}"
            )
            return float(s)

        # Bilinear interpolation
        ta = (amp_c - amp_lo) / (amp_hi - amp_lo)
        tp = (phase_c - phase_lo) / (phase_hi - phase_lo)

        s_low = (1.0 - tp) * v_ll + tp * v_lh
        s_high = (1.0 - tp) * v_hl + tp * v_hh
        s = (1.0 - ta) * s_low + ta * s_high

        print(
            f"[SA-MultCal] Bilinear interpolate s_xi for "
            f"A{amp_q:.1f}_P{phase_q:.1f}: "
            f"corners=({amp_lo:.1f},{phase_lo:.1f}), "
            f"({amp_lo:.1f},{phase_hi:.1f}), "
            f"({amp_hi:.1f},{phase_lo:.1f}), "
            f"({amp_hi:.1f},{phase_hi:.1f}), "
            f"s_xi={s:.6f}"
        )

        return float(s)

    def __call__(self, H_UL_est, csi_snr_db):
        H_UL_est = np.asarray(H_UL_est, dtype=np.complex64)
        B = H_UL_est.shape[0]

        H_UL_est_ri = complex_to_ri_np(H_UL_est)

        csi_snr_db_batch = (
            np.ones([B], dtype=np.float32)
            * float(csi_snr_db)
        )

        # Read original s_xi from table.
        s_xi_current = self._get_s_xi(csi_snr_db)

        s_xi_batch = (
            np.ones([B], dtype=np.float32)
            * float(s_xi_current)
        )

        std_amp_db_batch = (
            np.ones([B], dtype=np.float32)
            * float(TEST_MISMATCH_CONFIG["std_amp_db"])
        )

        std_phase_deg_batch = (
            np.ones([B], dtype=np.float32)
            * float(TEST_MISMATCH_CONFIG["std_phase_deg"])
        )

        # Use return_aux=True so we can manually control residual strength.

        # Reconstruct H_hat manually:
        # H_hat = H0 + residual_gain * scale * delta_bar_hat
        H_hat_ri, aux = self.model(
            tf.constant(H_UL_est_ri, dtype=tf.float32),
            csi_snr_db=tf.constant(csi_snr_db_batch, dtype=tf.float32),
            s_xi=tf.constant(s_xi_batch, dtype=tf.float32),
            std_amp_db=tf.constant(std_amp_db_batch, dtype=tf.float32),
            std_phase_deg=tf.constant(std_phase_deg_batch, dtype=tf.float32),
            training=False,
            return_aux=True,
        )

        H_hat = ri_to_complex_np(H_hat_ri.numpy())


        if H_hat.shape != (B, self.Nr, self.Nt):
            raise ValueError(
                f"SA-ResCalNet output shape mismatch: "
                f"H_hat.shape={H_hat.shape}, expected={(B, self.Nr, self.Nt)}"
            )

        return H_hat.astype(np.complex64)