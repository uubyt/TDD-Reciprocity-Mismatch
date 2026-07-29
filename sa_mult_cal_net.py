# sa_mult_cal_net.py
# Distribution-conditioned multiplicative diagonal calibration network
#
# Input:
#   H_UL_est_ri:   [B, Nt, Nr, 2]
#   csi_snr_db:    [B]
#   s_xi:          [B]
#   std_amp_db:    [B]
#   std_phase_deg: [B]
#   nr_db:    [B]
#   s_xi:          [B]
#   std_amp_db:    [B]
#   std#
# Output:
#   H_DL_hat_ri:   [B, Nr, Nt, 2]
#
# Main model:
#   H0 = alpha(gamma) * H_UL_est^T
#   H_hat = diag(d_r) H0 diag(d_t) + eta_add * g * s_xi * Delta_bar

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# Complex / real-imag utilities
# ============================================================

def complex_to_ri_np(H):
    """
    Convert complex numpy/tensor array to real-imag representation.

    Input:
        H: [...], complex
    Return:
        H_ri: [..., 2], float32
    """
    if isinstance(H, tf.Tensor):
        H = H.numpy()

    H = np.asarray(H)
    return np.stack([np.real(H), np.imag(H)], axis=-1).astype(np.float32)


def ri_to_complex_tf(H_ri):
    """
    H_ri: [..., 2]
    Return complex tensor [...]
    """
    H_ri = tf.cast(H_ri, tf.float32)
    return tf.complex(H_ri[..., 0], H_ri[..., 1])


def complex_to_ri_tf(H):
    """
    H: complex tensor [...]
    Return:
        H_ri: [..., 2], float32
    """
    H = tf.cast(H, tf.complex64)
    return tf.stack([tf.math.real(H), tf.math.imag(H)], axis=-1)


# ============================================================
# Basic blocks
# ============================================================

class ConvResBlock(layers.Layer):
    """
    Simple residual CNN block with LayerNorm.
    LayerNorm is used instead of BatchNorm to avoid unstable moving statistics
    across different mismatch regimes and SNRs.
    """

    def __init__(self, channels, kernel_size=3, name=None):
        super().__init__(name=name)

        self.conv1 = layers.Conv2D(
            channels,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
        )
        self.norm1 = layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.conv2 = layers.Conv2D(
            channels,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
        )
        self.norm2 = layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.norm1(y)
        y = tf.nn.gelu(y)

        y = self.conv2(y)
        y = self.norm2(y)

        return tf.nn.gelu(x + y)


class ConditionEncoder(layers.Layer):
    """
    Encode known error statistics and CSI SNR into a condition vector.

    Input:
        xi_norm: [B, 4]
            [snr_norm, s_xi_norm, amp_norm, phase_norm]

    Output:
        z_xi: [B, cond_dim]
    """

    def __init__(self, cond_dim=64, name=None):
        super().__init__(name=name)

        self.fc1 = layers.Dense(cond_dim, activation=tf.nn.gelu)
        self.fc2 = layers.Dense(cond_dim, activation=tf.nn.gelu)
        self.fc3 = layers.Dense(cond_dim)

    def call(self, xi_norm, training=False):
        z = self.fc1(xi_norm)
        z = self.fc2(z)
        z = self.fc3(z)
        return z


# ============================================================
# Main model
# ============================================================

class SAMultCalNet(tf.keras.Model):
    """
    Single-model distribution-conditioned multiplicative calibration network.

    Core structure:
        H0 = alpha(gamma) * H_UL_est^T

        d_r, d_t = f(H0, sigma_a, sigma_phi, gamma, s_xi)

        H_mult = diag(d_r) H0 diag(d_t)

        H_hat = H_mult + eta_add * g * s_xi * Delta_bar

    Compared with SA-ResCalNet:
        SA-ResCalNet mainly learns additive residual:
            H_hat = H0 + residual

        SAMultCalNet mainly learns multiplicative diagonal calibration:
            H_hat = diag(d_r) H0 diag(d_t) + small residual
    """

    def __init__(
        self,
        Nt,
        Nr,
        feature_dim=64,
        num_blocks=3,
        cond_dim=64,
        snr_min_db=0.0,
        snr_max_db=26.0,
        s_max=1.5,
        amp_min_db=2.0,
        amp_max_db=4.0,
        phase_min_deg=20.0,
        phase_max_deg=90.0,
        amp_clip=0.5,
        phase_clip=1.0,
        delta_clip=3.0,
        eta_add=0.05,
        use_shrinkage=True,

        # condition ablation
        condition_mode="full",
        name="SAMultCalNet",
    ):
        super().__init__(name=name)

        self.Nt = int(Nt)
        self.Nr = int(Nr)

        self.feature_dim = int(feature_dim)
        self.num_blocks = int(num_blocks)
        self.cond_dim = int(cond_dim)

        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)

        self.s_max = float(s_max)

        self.amp_min_db = float(amp_min_db)
        self.amp_max_db = float(amp_max_db)

        self.phase_min_deg = float(phase_min_deg)
        self.phase_max_deg = float(phase_max_deg)

        self.amp_clip = float(amp_clip)
        self.phase_clip = float(phase_clip)
        self.delta_clip = float(delta_clip)
        self.eta_add = float(eta_add)

        self.use_shrinkage = bool(use_shrinkage)
        self.condition_mode = str(condition_mode)

        # Condition encoder
        self.cond_encoder = ConditionEncoder(cond_dim=cond_dim)

        # H0 feature extractor
        # Input feature channels:
        #   H0.real, H0.imag, log|H0|, cos(angle), sin(angle),
        #   snr_map, s_xi_map, amp_map, phase_map
        self.input_proj = layers.Conv2D(
            feature_dim,
            kernel_size=3,
            padding="same",
            activation=tf.nn.gelu,
        )

        self.blocks = [
            ConvResBlock(feature_dim, kernel_size=3, name=f"conv_res_block_{i}")
            for i in range(num_blocks)
        ]

        self.gap = layers.GlobalAveragePooling2D()

        # Heads for multiplicative diagonal gains.
        # Zero initialization makes the initial gain d_r=d_t=1,
        # hence initial H_mult=H0. This stabilizes early training.
        self.row_head = layers.Dense(
            2 * self.Nr,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="row_gain_head",
        )

        self.col_head = layers.Dense(
            2 * self.Nt,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="col_gain_head",
        )

        # Small additive residual head.
        # The final conv is also zero-initialized, so initial residual is zero.
        self.res_conv1 = layers.Conv2D(
            feature_dim,
            kernel_size=3,
            padding="same",
            activation=tf.nn.gelu,
        )
        self.res_conv2 = layers.Conv2D(
            feature_dim,
            kernel_size=3,
            padding="same",
            activation=tf.nn.gelu,
        )
        self.res_out = layers.Conv2D(
            2,
            kernel_size=3,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="small_residual_head",
        )

    # --------------------------------------------------------
    # Normalization utilities
    # --------------------------------------------------------

    def _normalize_scalar(self, x, x_min, x_max):
        x = tf.reshape(tf.cast(x, tf.float32), [-1])
        y = (x - float(x_min)) / (float(x_max) - float(x_min) + 1e-12)
        return tf.clip_by_value(y, 0.0, 1.0)

    def normalize_condition(
        self,
        csi_snr_db,
        s_xi,
        std_amp_db,
        std_phase_deg,
    ):
        """
        Return:
            xi_norm: [B, 4]
        """
        snr_norm = self._normalize_scalar(
            csi_snr_db,
            self.snr_min_db,
            self.snr_max_db,
        )

        s_norm = tf.reshape(tf.cast(s_xi, tf.float32), [-1])
        s_norm = tf.clip_by_value(s_norm / (self.s_max + 1e-12), 0.0, 1.0)

        amp_norm = self._normalize_scalar(
            std_amp_db,
            self.amp_min_db,
            self.amp_max_db,
        )

        phase_norm = self._normalize_scalar(
            std_phase_deg,
            self.phase_min_deg,
            self.phase_max_deg,
        )

        xi_norm = tf.stack(
            [snr_norm, s_norm, amp_norm, phase_norm],
            axis=-1,
        )

        return xi_norm
    

    def apply_condition_mode(self, xi_norm):
        """
        xi_norm: [B, 4]
            [snr_norm, s_xi_norm, amp_norm, phase_norm]

        condition_mode:
            full:
                use all condition inputs.

            no_error:
                keep CSI-SNR condition, remove mismatch condition.
                xi = [snr_norm, 0, 0, 0]

            none:
                remove all condition information from condition encoder and maps.
                xi = [0, 0, 0, 0]

        Note:
            build_H0 still uses the real csi_snr_db for shrinkage.
            This function only controls the condition branch.
        """
        xi_norm = tf.cast(xi_norm, tf.float32)

        if self.condition_mode == "full":
            return xi_norm

        if self.condition_mode == "no_error":
            snr_norm = xi_norm[:, 0:1]
            zeros = tf.zeros_like(xi_norm[:, 1:4])
            return tf.concat([snr_norm, zeros], axis=-1)

        if self.condition_mode == "none":
            return tf.zeros_like(xi_norm)

        raise ValueError(f"Unknown condition_mode: {self.condition_mode}")

    def build_H0(
        self,
        H_UL_est_ri,
        csi_snr_db,
    ):
        """
        H_UL_est_ri: [B, Nt, Nr, 2]

        Return:
            H0:    [B, Nr, Nt] complex
            H0_ri: [B, Nr, Nt, 2]
        """
        H_UL_est = ri_to_complex_tf(H_UL_est_ri)  # [B, Nt, Nr]

        # Reciprocity transpose
        H0 = tf.transpose(H_UL_est, perm=[0, 2, 1])  # [B, Nr, Nt]

        if self.use_shrinkage:
            csi_snr_db = tf.reshape(tf.cast(csi_snr_db, tf.float32), [-1])
            rho = tf.pow(10.0, csi_snr_db / 10.0)
            alpha = rho / (1.0 + rho)
            H0 = tf.cast(alpha[:, None, None], tf.complex64) * H0

        H0_ri = complex_to_ri_tf(H0)

        return H0, H0_ri

    def build_H0_features(
        self,
        H0,
        H0_ri,
        xi_norm,
    ):
        """
        Build CNN input feature map from H0 and normalized conditions.

        H0:      [B, Nr, Nt] complex
        H0_ri:   [B, Nr, Nt, 2]
        xi_norm: [B, 4]

        Return:
            feat: [B, Nr, Nt, 9]
        """
        B = tf.shape(H0_ri)[0]

        abs_H0 = tf.abs(H0)
        log_abs = tf.math.log(abs_H0 + 1e-6)[..., None]

        phase = tf.math.angle(H0)
        cos_phase = tf.cos(phase)[..., None]
        sin_phase = tf.sin(phase)[..., None]

        # Condition maps
        snr_map = tf.ones([B, self.Nr, self.Nt, 1], dtype=tf.float32) * \
            xi_norm[:, None, None, 0:1]

        s_map = tf.ones([B, self.Nr, self.Nt, 1], dtype=tf.float32) * \
            xi_norm[:, None, None, 1:2]

        amp_map = tf.ones([B, self.Nr, self.Nt, 1], dtype=tf.float32) * \
            xi_norm[:, None, None, 2:3]

        phase_map = tf.ones([B, self.Nr, self.Nt, 1], dtype=tf.float32) * \
            xi_norm[:, None, None, 3:4]

        feat = tf.concat(
            [
                H0_ri,
                log_abs,
                cos_phase,
                sin_phase,
                snr_map,
                s_map,
                amp_map,
                phase_map,
            ],
            axis=-1,
        )

        return feat

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def call(
        self,
        H_UL_est_ri,
        csi_snr_db,
        s_xi,
        std_amp_db,
        std_phase_deg,
        training=False,
        return_aux=False,
    ):
        """
        Return:
            H_hat_ri: [B, Nr, Nt, 2]
        """

        # 1. Coarse reciprocity preprocessing
        H0, H0_ri = self.build_H0(
            H_UL_est_ri=H_UL_est_ri,
            csi_snr_db=csi_snr_db,
        )

        # 2. Error-statistics condition encoding
        xi_norm_raw = self.normalize_condition(
            csi_snr_db=csi_snr_db,
            s_xi=s_xi,
            std_amp_db=std_amp_db,
            std_phase_deg=std_phase_deg,
        )

        xi_norm = self.apply_condition_mode(xi_norm_raw)

        z_xi = self.cond_encoder(
            xi_norm,
            training=training,
        )

        # 3. CNN feature extraction from H0
        feat = self.build_H0_features(
            H0=H0,
            H0_ri=H0_ri,
            xi_norm=xi_norm,
        )

        x = self.input_proj(feat)

        for block in self.blocks:
            x = block(x, training=training)

        h_summary = self.gap(x)  # [B, feature_dim]

        q = tf.concat([h_summary, z_xi], axis=-1)

        # 4. Generate row/column complex diagonal gains
        row_param = self.row_head(q)
        row_param = tf.reshape(row_param, [-1, self.Nr, 2])

        col_param = self.col_head(q)
        col_param = tf.reshape(col_param, [-1, self.Nt, 2])

        # Log-amplitude and phase correction
        row_log_amp = self.amp_clip * tf.tanh(row_param[..., 0])
        row_phase = self.phase_clip * tf.tanh(row_param[..., 1])

        col_log_amp = self.amp_clip * tf.tanh(col_param[..., 0])
        col_phase = self.phase_clip * tf.tanh(col_param[..., 1])

        d_r = tf.exp(tf.complex(row_log_amp, row_phase))  # [B, Nr]
        d_t = tf.exp(tf.complex(col_log_amp, col_phase))  # [B, Nt]

        # 5. Multiplicative diagonal calibration
        H_mult = d_r[:, :, None] * H0 * d_t[:, None, :]  # [B, Nr, Nt]
        H_mult_ri = complex_to_ri_tf(H_mult)

        # 6. Small additive residual refinement
        z_map = tf.tile(
            z_xi[:, None, None, :],
            [1, self.Nr, self.Nt, 1],
        )

        res_in = tf.concat(
            [
                H0_ri,
                H_mult_ri,
                z_map,
            ],
            axis=-1,
        )

        r = self.res_conv1(res_in)
        r = self.res_conv2(r)
        delta_bar_ri = self.delta_clip * tf.tanh(self.res_out(r))

        delta_bar_H = tf.complex(
            delta_bar_ri[..., 0],
            delta_bar_ri[..., 1],
        )

        # Scale residual by average channel magnitude and s_xi
        g = tf.sqrt(
            tf.reduce_mean(tf.abs(H0) ** 2, axis=[1, 2], keepdims=True)
            + 1e-12
        )  # [B, 1, 1]

        s_xi_vec = tf.reshape(tf.cast(s_xi, tf.float32), [-1, 1, 1])
        scale = tf.cast(g * s_xi_vec, tf.complex64)

        delta_H = scale * delta_bar_H

        H_hat = H_mult + tf.cast(self.eta_add, tf.complex64) * delta_H
        H_hat_ri = complex_to_ri_tf(H_hat)

        if return_aux:
            aux = {
                "H0": H0,
                "H0_ri": H0_ri,
                "H_mult": H_mult,
                "H_mult_ri": H_mult_ri,
                "H_hat": H_hat,
                "H_hat_ri": H_hat_ri,
                "d_r": d_r,
                "d_t": d_t,
                "delta_H": delta_H,
                "delta_bar_ri": delta_bar_ri,
                "xi_norm_raw": xi_norm_raw,
                "xi_norm": xi_norm,
                "z_xi": z_xi,
                "condition_mode": self.condition_mode,
            }
            return H_hat_ri, aux

        return H_hat_ri


# ============================================================
# Loss functions
# ============================================================

def calibration_nmse_loss_ri(H_hat_ri, H_true_ri):
    """
    NMSE between predicted and true downlink channel.

    H_hat_ri:  [B, Nr, Nt, 2]
    H_true_ri: [B, Nr, Nt, 2]
    """
    H_hat = ri_to_complex_tf(H_hat_ri)
    H_true = ri_to_complex_tf(H_true_ri)

    err = tf.reduce_sum(tf.abs(H_hat - H_true) ** 2, axis=[1, 2])
    power = tf.reduce_sum(tf.abs(H_true) ** 2, axis=[1, 2]) + 1e-12

    return tf.reduce_mean(err / power)


def gram_leakage_loss_ri(H_hat_ri, H_true_ri):
    """
    Compare normalized transmit-side Gram matrices H^H H.
    This is more directly related to SVD precoder directions than pure NMSE.

    H_hat_ri:  [B, Nr, Nt, 2]
    H_true_ri: [B, Nr, Nt, 2]
    """
    H_hat = ri_to_complex_tf(H_hat_ri)
    H_true = ri_to_complex_tf(H_true_ri)

    G_hat = tf.matmul(H_hat, H_hat, adjoint_a=True)   # [B, Nt, Nt]
    G_true = tf.matmul(H_true, H_true, adjoint_a=True)

    diff = G_hat - G_true

    diff_power = tf.reduce_sum(tf.abs(diff) ** 2, axis=[1, 2])
    true_power = tf.reduce_sum(tf.abs(G_true) ** 2, axis=[1, 2]) + 1e-12

    return tf.reduce_mean(diff_power / true_power)


def residual_energy_loss(aux, H_true_ri):
    """
    Penalize excessive additive residual energy.
    This keeps the multiplicative diagonal calibration as the main correction.

    aux["delta_H"]: [B, Nr, Nt] complex
    """
    if aux is None or "delta_H" not in aux:
        return tf.constant(0.0, dtype=tf.float32)

    delta_H = aux["delta_H"]
    H_true = ri_to_complex_tf(H_true_ri)

    err = tf.reduce_sum(tf.abs(delta_H) ** 2, axis=[1, 2])
    power = tf.reduce_sum(tf.abs(H_true) ** 2, axis=[1, 2]) + 1e-12

    return tf.reduce_mean(err / power)


def total_sa_mult_cal_loss(
    model,
    H_UL_est_ri,
    H_DL_true_ri,
    csi_snr_db,
    s_xi,
    std_amp_db,
    std_phase_deg,
    lambda_res=0.0,
    lambda_gram=0.02,
    training=True,
):
    """
    Compatible with the existing SA-ResCalNet training pipeline.

    Return:
        loss, logs

    logs contain both old and new keys:
        "loss", "nmse", "res", "gram",
        "loss_nmse", "loss_res", "loss_gram"
    """

    H_hat_ri, aux = model(
        H_UL_est_ri,
        csi_snr_db=csi_snr_db,
        s_xi=s_xi,
        std_amp_db=std_amp_db,
        std_phase_deg=std_phase_deg,
        training=training,
        return_aux=True,
    )

    loss_nmse = calibration_nmse_loss_ri(
        H_hat_ri=H_hat_ri,
        H_true_ri=H_DL_true_ri,
    )

    loss_gram = gram_leakage_loss_ri(
        H_hat_ri=H_hat_ri,
        H_true_ri=H_DL_true_ri,
    )

    loss_res = residual_energy_loss(
        aux=aux,
        H_true_ri=H_DL_true_ri,
    )

    loss = (
        loss_nmse
        + float(lambda_gram) * loss_gram
        + float(lambda_res) * loss_res
    )

    logs = {
        "loss": loss,
        "nmse": loss_nmse,
        "res": loss_res,
        "gram": loss_gram,

        # Compatibility with your current get_log_value()
        "loss_nmse": loss_nmse,
        "loss_res": loss_res,
        "loss_gram": loss_gram,
    }

    return loss, logs