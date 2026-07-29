import numpy as np
import tensorflow as tf


def complex_to_ri_np(H):
    """
    H: complex array [..., Nr, Nt]
    return: real-imag array [..., Nr, Nt, 2]
    """
    return np.stack([H.real, H.imag], axis=-1).astype(np.float32)


def ri_to_complex_np(H_ri):
    """
    H_ri: [..., Nr, Nt, 2]
    return: complex array [..., Nr, Nt]
    """
    return (H_ri[..., 0] + 1j * H_ri[..., 1]).astype(np.complex64)


def ri_to_complex_tf(H_ri):
    return tf.complex(H_ri[..., 0], H_ri[..., 1])


class PredCaliNetStatic(tf.keras.Model):
    """
    Static adaptation of PredCaliNet for snapshot-based calibration.

    Original idea:
        Past P-frame UL CSI sequence -> LSTM -> FC calibration -> future DL CSI.

    Our adapted task:
        One-frame naive DL CSI H0 = H_UL_est^T -> LSTM/FC -> calibrated DL CSI.
    """

    def __init__(
        self,
        Nr,
        Nt,
        hidden_dim=256,
        num_lstm_layers=3,
        fc_dims=None,
        activation="tanh",
        output_activation=None,
        name="PredCaliNetStatic",
    ):
        super().__init__(name=name)

        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.input_dim = 2 * self.Nr * self.Nt
        self.output_dim = 2 * self.Nr * self.Nt

        if fc_dims is None:
            # Inspired by PredCaliNet FC calibration head.
            fc_dims = [
                8 * self.Nr * self.Nt,
                4 * self.Nr * self.Nt,
                2 * self.Nr * self.Nt,
            ]

        self.lstm_layers = []
        for layer_idx in range(num_lstm_layers):
            return_sequences = layer_idx < num_lstm_layers - 1
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    hidden_dim,
                    return_sequences=return_sequences,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    name=f"lstm_{layer_idx+1}",
                )
            )

        self.fc_layers = []
        for k, dim in enumerate(fc_dims):
            self.fc_layers.append(
                tf.keras.layers.Dense(
                    dim,
                    activation=activation,
                    name=f"fc_calib_{k+1}",
                )
            )

        self.out_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=output_activation,
            name="dl_output",
        )

    def call(self, H0_ri, training=False):
        """
        H0_ri: [B, Nr, Nt, 2]
        output: [B, Nr, Nt, 2]
        """

        B = tf.shape(H0_ri)[0]

        # [B, Nr, Nt, 2] -> [B, 1, 2*Nr*Nt]
        x = tf.reshape(H0_ri, [B, 1, self.input_dim])

        for lstm in self.lstm_layers:
            x = lstm(x, training=training)

        for fc in self.fc_layers:
            x = fc(x, training=training)

        y = self.out_layer(x, training=training)

        # [B, 2*Nr*Nt] -> [B, Nr, Nt, 2]
        y = tf.reshape(y, [B, self.Nr, self.Nt, 2])
        return y


def mse_loss(H_hat_ri, H_true_ri):
    return tf.reduce_mean(tf.square(H_hat_ri - H_true_ri))


def nmse_loss(H_hat_ri, H_true_ri, eps=1e-12):
    H_hat = ri_to_complex_tf(H_hat_ri)
    H_true = ri_to_complex_tf(H_true_ri)

    err = tf.reduce_sum(tf.abs(H_hat - H_true) ** 2, axis=[1, 2])
    den = tf.reduce_sum(tf.abs(H_true) ** 2, axis=[1, 2]) + eps

    return tf.reduce_mean(err / den)


def total_predcalinet_loss(
    model,
    H0_ri,
    H_DL_true_ri,
    loss_type="nmse",
    training=True,
):
    H_hat_ri = model(H0_ri, training=training)

    mse = mse_loss(H_hat_ri, H_DL_true_ri)
    nmse = nmse_loss(H_hat_ri, H_DL_true_ri)

    if loss_type == "mse":
        loss = mse
    elif loss_type == "nmse":
        loss = nmse
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    logs = {
        "loss": loss,
        "mse": mse,
        "nmse": nmse,
    }

    return loss, logs, H_hat_ri