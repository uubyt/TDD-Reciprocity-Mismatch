import numpy as np
import tensorflow as tf


def complex_to_ri_np(H):
    return np.stack([H.real, H.imag], axis=-1).astype(np.float32)


def ri_to_complex_np(H_ri):
    return (H_ri[..., 0] + 1j * H_ri[..., 1]).astype(np.complex64)


def ri_to_complex_tf(H_ri):
    return tf.complex(H_ri[..., 0], H_ri[..., 1])


class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, name=None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class DeepRCCNN(tf.keras.Model):
    """
    DeepRC-inspired CNN baseline.

    Input:
        H0_ri: [B, Nr, Nt, 2]
    Output:
        H_hat_ri: [B, Nr, Nt, 2]
    """

    def __init__(self, Nr, Nt, base_channels=64, name="DeepRCCNN"):
        super().__init__(name=name)

        self.Nr = int(Nr)
        self.Nt = int(Nt)

        # MetrNet-style encoder
        self.enc1 = ConvBNReLU(base_channels, 3, strides=1, name="enc1")
        self.enc2 = ConvBNReLU(base_channels * 2, 3, strides=2, name="enc2")
        self.enc3 = ConvBNReLU(base_channels * 4, 3, strides=2, name="enc3")

        # MetrNet-style decoder
        self.up1 = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
            name="up1",
        )
        self.dec1 = ConvBNReLU(base_channels * 2, 3, strides=1, name="dec1")

        self.up2 = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
            name="up2",
        )
        self.dec2 = ConvBNReLU(base_channels, 3, strides=1, name="dec2")

        # residual output, zero init makes initial mapping close to identity
        self.out = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="residual_output",
        )

    def call(self, H0_ri, training=False):
        x0 = H0_ri

        # inspired by DeepRC's tanh regularization for outlier suppression
        x = tf.tanh(H0_ri)

        x = self.enc1(x, training=training)
        x = self.enc2(x, training=training)
        x = self.enc3(x, training=training)

        x = self.up1(x)
        x = self.dec1(x, training=training)

        x = self.up2(x)
        x = self.dec2(x, training=training)

        delta = self.out(x, training=training)

        # residual calibration
        H_hat_ri = x0 + delta

        return H_hat_ri


def mse_loss(H_hat_ri, H_true_ri):
    return tf.reduce_mean(tf.square(H_hat_ri - H_true_ri))


def nmse_loss(H_hat_ri, H_true_ri, eps=1e-12):
    H_hat = ri_to_complex_tf(H_hat_ri)
    H_true = ri_to_complex_tf(H_true_ri)

    err = tf.reduce_sum(tf.abs(H_hat - H_true) ** 2, axis=[1, 2])
    den = tf.reduce_sum(tf.abs(H_true) ** 2, axis=[1, 2]) + eps

    return tf.reduce_mean(err / den)


def total_deeprc_cnn_loss(
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