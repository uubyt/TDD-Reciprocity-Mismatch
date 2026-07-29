import numpy as np
import tensorflow as tf

from config_system import (
    MOE_REFINER_CALIB_CONFIG,
    RESIDUAL_REFINER_CONFIG,
)

from MOE_calibration import MoECalibrator
from residual_refiner import (
    ResidualChannelRefiner,
    build_refiner_input_tf,
)


class MoEResidualRefinerCalibrator:
    """
    Frozen pretrained MoE + trained residual refiner.

    H_UL_est
      -> Frozen MoE
      -> H_DL_base
      -> Residual Refiner
      -> H_DL_hat = H_DL_base + delta_H
    """

    def __init__(self, Nt, Nr):
        self.Nt = Nt
        self.Nr = Nr

        # ----------------------------------------------------
        # 1. Load original pretrained MoE
        # ----------------------------------------------------
        base_moe_weights_path = MOE_REFINER_CALIB_CONFIG["base_moe_weights_path"]

        print("[MoE+Refiner] Loading frozen base MoE...")
        self.base_moe = MoECalibrator(
            Nt=Nt,
            Nr=Nr,
            weights_path=base_moe_weights_path,
        )

        self.base_moe.model.trainable = False

        # ----------------------------------------------------
        # 2. Build residual refiner
        # ----------------------------------------------------
        self.refiner = ResidualChannelRefiner(
            hidden_dim=RESIDUAL_REFINER_CONFIG["hidden_dim"],
            num_blocks=RESIDUAL_REFINER_CONFIG["num_blocks"],
            delta_scale=RESIDUAL_REFINER_CONFIG["delta_scale"],
        )

        dummy_input = tf.zeros((1, Nr, Nt, 4), dtype=tf.float32)
        _ = self.refiner(dummy_input, training=False)

        # ----------------------------------------------------
        # 3. Load trained refiner weights
        # ----------------------------------------------------
        refiner_weights_path = MOE_REFINER_CALIB_CONFIG["refiner_weights_path"]

        print(f"[MoE+Refiner] Loading refiner weights from: {refiner_weights_path}")
        self.refiner.load_weights(refiner_weights_path)
        print("[MoE+Refiner] Refiner loaded successfully.")

    def predict_tf(
        self,
        H_UL_est_tf,
        csi_snr_db_batch_tf,
        training=False,
    ):
        """
        TensorFlow inference.

        Args:
            H_UL_est_tf:
                [B, Nt, Nr] complex tensor
            csi_snr_db_batch_tf:
                [B] float tensor

        Returns:
            H_DL_hat_tf:
                [B, Nr, Nt] complex tensor
        """

        # 1. Frozen MoE prediction
        H_DL_base = self.base_moe.predict_tf(
            H_UL_est_tf=H_UL_est_tf,
            csi_snr_db_batch_tf=csi_snr_db_batch_tf,
            gate_weights_batch_tf=None,
            training=False,
        )

        # 2. Residual refinement
        refiner_input = build_refiner_input_tf(
            H_DL_base=H_DL_base,
            H_UL_est=H_UL_est_tf,
        )

        delta_ri = self.refiner(
            refiner_input,
            training=training,
        )

        delta_H = tf.complex(
            delta_ri[..., 0],
            delta_ri[..., 1],
        )

        # 3. Final refined channel
        H_DL_hat_tf = H_DL_base + delta_H

        return H_DL_hat_tf

    def __call__(self, H_UL_est, csi_snr_db):
        """
        Numpy interface for run_traditional_link.py.

        Args:
            H_UL_est:
                [B, Nt, Nr] complex numpy array
            csi_snr_db:
                scalar or [B]

        Returns:
            H_DL_hat:
                [B, Nr, Nt] complex numpy array
        """

        H_UL_est_tf = tf.constant(
            H_UL_est.astype(np.complex64),
            dtype=tf.complex64,
        )

        B = H_UL_est.shape[0]

        if np.isscalar(csi_snr_db):
            csi_snr_db_batch_tf = tf.ones(
                [B],
                dtype=tf.float32,
            ) * float(csi_snr_db)
        else:
            csi_snr_db_batch_tf = tf.constant(
                np.asarray(csi_snr_db, dtype=np.float32),
                dtype=tf.float32,
            )

        H_DL_hat_tf = self.predict_tf(
            H_UL_est_tf=H_UL_est_tf,
            csi_snr_db_batch_tf=csi_snr_db_batch_tf,
            training=False,
        )

        return H_DL_hat_tf.numpy().astype(np.complex64)