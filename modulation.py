import numpy as np


def generate_random_bits(batch_size, Ns, num_symbols, bits_per_symbol=2, rng=None):
    if bits_per_symbol != 2:
        raise NotImplementedError("This version currently supports 4QAM/QPSK only.")

    if rng is None:
        rng = np.random.default_rng()

    return rng.integers(
        low=0,
        high=2,
        size=(batch_size, Ns, num_symbols, bits_per_symbol),
        dtype=np.int32,
    )


def qam4_modulate(bits):
    """
    bits: [B, Ns, T, 2]
    return symbols: [B, Ns, T]
    """
    bits = bits.astype(np.float32)
    b0 = bits[..., 0]
    b1 = bits[..., 1]

    real = 1.0 - 2.0 * b0
    imag = 1.0 - 2.0 * b1

    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def qam4_hard_demod(symbols):
    """
    symbols: [B, Ns, T]
    return bits_hat: [B, Ns, T, 2]
    """
    bits_hat = np.zeros(symbols.shape + (2,), dtype=np.int32)
    bits_hat[..., 0] = (np.real(symbols) < 0).astype(np.int32)
    bits_hat[..., 1] = (np.imag(symbols) < 0).astype(np.int32)
    return bits_hat
