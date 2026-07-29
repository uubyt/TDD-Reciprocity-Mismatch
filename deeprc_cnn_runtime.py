import os
import numpy as np
import tensorflow as tf

from deeprc_cnn_calibration import (
    DeepRCCNN,
    complex_to_ri_np,
    ri_to_complex_np,
)


class DeepRCCNNCalibrator:
    def __init__(self, Nt, Nr, config=None):
        from config_system import DEEPRC_CNN_CONFIG

        self.Nt = int(Nt)
        self.Nr = int(Nr)

        cfg = dict(DEEPRC_CNN_CONFIG)
        if config is not None:
            cfg.update(config)

        self.model = DeepRCCNN(
            Nr=self.Nr,
            Nt=self.Nt,
            base_channels=cfg.get("base_channels", 64),
        )

        dummy = tf.zeros([1, self.Nr, self.Nt, 2], dtype=tf.float32)
        _ = self.model(dummy, training=False)

        weights_path = cfg["weights_path"]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"DeepRC-CNN weights not found: {weights_path}"
            )

        self.model.load_weights(weights_path)
        print(f"Loaded DeepRC-CNN weights from: {weights_path}")

    def __call__(self, H_UL_est, csi_snr_db=None):
        H_UL_est = np.asarray(H_UL_est, dtype=np.complex64)

        H0 = np.transpose(H_UL_est, (0, 2, 1)).astype(np.complex64)
        H0_ri = complex_to_ri_np(H0)

        H_hat_ri = self.model(
            tf.constant(H0_ri, dtype=tf.float32),
            training=False,
        )

        H_hat = ri_to_complex_np(H_hat_ri.numpy())

        return H_hat.astype(np.complex64)