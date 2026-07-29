import numpy as np
import tensorflow as tf


def complex_to_ri_np(H):
    """
    Convert complex numpy array to real-imag representation.

    Input:
        H: complex array, shape [...].

    Output:
        H_ri: float32 array, shape [..., 2].
    """
    H = np.asarray(H, dtype=np.complex64)
    return np.stack([H.real, H.imag], axis=-1).astype(np.float32)


def ri_to_complex_np(H_ri):
    """
    Convert real-imag numpy array to complex numpy array.

    Input:
        H_ri: [..., 2]

    Output:
        H: complex64 array [...]
    """
    H_ri = np.asarray(H_ri, dtype=np.float32)
    return (H_ri[..., 0] + 1j * H_ri[..., 1]).astype(np.complex64)

def ri_to_complex_tf(H_ri):
    """
    Convert real-imag TensorFlow tensor to complex tensor.

    Input:
        H_ri: [..., 2]

    Output:
        H: complex64 tensor [...]
    """
    return tf.complex(H_ri[..., 0], H_ri[..., 1])


class CalinetDNN(tf.keras.Model):
    """
    Black-box Calinet-DNN baseline.

    Input:
        H0_ri: [B, Nr, Nt, 2]
               where H0 = H_UL_est^T.

    Output:
        H_DL_hat_ri: [B, Nr, Nt, 2]
    """

    def __init__(
        self,
        Nr,
        Nt,
        hidden_dims=(512, 512, 512),
        activation="tanh",
        output_activation=None,
        name="CalinetDNN",
    ):
        super().__init__(name=name)

        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.input_dim = self.Nr * self.Nt * 2
        self.output_dim = self.Nr * self.Nt * 2

        self.hidden_layers = []
        for h in hidden_dims:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    int(h),
                    activation=activation,
                )
            )

        self.out_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=output_activation,
        )

    def call(self, H0_ri, training=False):
        B = tf.shape(H0_ri)[0]

        x = tf.reshape(H0_ri, [B, self.input_dim])

        for layer in self.hidden_layers:
            x = layer(x, training=training)

        y = self.out_layer(x, training=training)

        H_DL_hat_ri = tf.reshape(
            y,
            [B, self.Nr, self.Nt, 2],
        )

        return H_DL_hat_ri


def mse_loss(H_hat_ri, H_true_ri):
    return tf.reduce_mean(tf.square(H_hat_ri - H_true_ri))


def nmse_loss(H_hat_ri, H_true_ri, eps=1e-12):
    H_hat = ri_to_complex_tf(H_hat_ri)
    H_true = ri_to_complex_tf(H_true_ri)

    num = tf.reduce_sum(
        tf.abs(H_hat - H_true) ** 2,
        axis=[1, 2],
    )

    den = tf.reduce_sum(
        tf.abs(H_true) ** 2,
        axis=[1, 2],
    ) + eps

    return tf.reduce_mean(num / den)


def total_calinet_loss(
    model,
    H0_ri,
    H_DL_true_ri,
    loss_type="nmse",
    training=True,
):
    """
    Calinet training loss.

    H0_ri:
        [B, Nr, Nt, 2], input H_UL_est^T.

    H_DL_true_ri:
        [B, Nr, Nt, 2], true effective downlink channel.
    """
    H_hat_ri = model(H0_ri, training=training)

    mse = mse_loss(H_hat_ri, H_DL_true_ri)
    nmse = nmse_loss(H_hat_ri, H_DL_true_ri)

    if loss_type == "mse":
        loss = mse
    elif loss_type == "nmse":
        loss = nmse
    else:
        raise ValueError(f"Unknown Calinet loss_type: {loss_type}")

    logs = {
        "loss": loss,
        "mse": mse,
        "nmse": nmse,
    }

    return loss, logs