# 测试链路系统

import os
import sys


# Allow importing the original ./models package when this folder is placed under the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
tf.config.optimizer.set_jit(False)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import json
import time
import numpy as np


import argparse

SA_MULT_VARIANT_TO_SCHEME = {
    "full": "sa_mult_full_svd",
    "nores": "sa_mult_nores_svd",
    "nocond": "sa_mult_nores_nocond_svd",
}
SA_MULT_SCHEME_TO_VARIANT = {
    v: k for k, v in SA_MULT_VARIANT_TO_SCHEME.items()
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_variant",
        type=str,
        default="full",
        choices=["full", "nores", "nocond", "all"],
        help="Which SA-Mult ablation variant to evaluate.",
    )

    parser.add_argument(
        "--include_moe",
        action="store_true",
        help="Whether to include MoE baselines in this run.",
    )

    parser.add_argument(
        "--include_mcsgd",
        action="store_true",
        help="Whether to include MC-SGD baseline in this run.",
    )

    parser.add_argument(
        "--include_nn",
        action="store_true",
        help="Whether to include Calinet-DNN calibration baseline in this run.",
    )

    parser.add_argument(
        "--include_predcalinet",
        action="store_true",
        help="Whether to include PredCaliNet-static calibration baseline in this run.",
    )

    parser.add_argument(
        "--include_deeprc_cnn",
        action="store_true",
        help="Include DeepRC-inspired CNN calibration baseline.",
    )

    return parser.parse_args()


from config_system import (
    DATA_CONFIG,
    PATH_CONFIG,
    EXTERNAL_PACKAGE_CONFIG,
    TEST_MISMATCH_CONFIG,
    TEST_EXPERIMENT_TAG,
    SNR_CONFIG,
    LINK_CONFIG,
    MCSGD_CONFIG,
    SA_MULT_CONFIGS,
)

from channels import (
    load_and_split_complex_data,
    generate_fixed_mismatch,
    build_downlink_from_uplink,
    add_awgn_to_uplink,
)

from precoding import build_precoder, normalize_precoder, normalize_precoder_equal_stream
from modulation import generate_random_bits, qam4_modulate, qam4_hard_demod
from receiver import downlink_transmit, mmse_equalize
from metrics import (
    compute_ber,
    compute_ser,
    compute_bler,
    compute_evm,
    compute_effective_se,
    gaussian_capacity_with_precoder,
    compute_gram_leakage,
    compute_post_mmse_sinr_stats,
)


def synchronize_tensorflow():
    """
    Synchronize pending TensorFlow GPU operations before timing.

    Some TensorFlow versions provide tf.experimental.async_wait(),
    while others do not. This wrapper keeps the code compatible.
    """
    async_wait = getattr(tf.experimental, "async_wait", None)

    if callable(async_wait):
        async_wait()


def count_total_parameters(model_or_wrapper):
    """
    Count total parameters of a TensorFlow/Keras model or wrapper.

    Returns:
        int: total number of model parameters.
    """
    if model_or_wrapper is None:
        return 0

    # Object itself is a Keras model
    if isinstance(model_or_wrapper, tf.keras.Model):
        return int(model_or_wrapper.count_params())

    # Wrapper itself exposes count_params()
    if hasattr(model_or_wrapper, "count_params"):
        try:
            return int(model_or_wrapper.count_params())
        except Exception:
            pass

    # Common attribute names used inside model wrappers
    candidate_attributes = [
        "model",
        "network",
        "net",
        "calibration_model",
        "calibrator_model",
        "cnn_model",
        "predictor",
        "backbone",
    ]

    for attribute_name in candidate_attributes:
        inner_model = getattr(
            model_or_wrapper,
            attribute_name,
            None,
        )

        if inner_model is None:
            continue

        if hasattr(inner_model, "count_params"):
            try:
                return int(inner_model.count_params())
            except Exception:
                pass

        variables = getattr(inner_model, "variables", None)

        if variables is not None:
            return int(
                np.sum([
                    np.prod(variable.shape)
                    for variable in variables
                ])
            )

    # Wrapper directly exposes variables
    variables = getattr(
        model_or_wrapper,
        "variables",
        None,
    )

    if variables is not None:
        return int(
            np.sum([
                np.prod(variable.shape)
                for variable in variables
            ])
        )

    print(
        f"[Warning] Cannot locate model parameters inside "
        f"{type(model_or_wrapper).__name__}."
    )
    print(
        f"Available attributes: "
        f"{list(vars(model_or_wrapper).keys())}"
    )

    return 0


def get_csi_snr_for_dl_snr(dl_snr_db):
    if SNR_CONFIG["csi_snr_mode"] == "same_as_dl":
        return float(dl_snr_db)

    if SNR_CONFIG["csi_snr_mode"] == "fixed":
        return float(SNR_CONFIG["fixed_csi_snr_db"])

    raise ValueError(f"Unknown csi_snr_mode: {SNR_CONFIG['csi_snr_mode']}")


def build_mcsgd_mismatch_pool(Nt, Nr):
    """
    Build mismatch pool for MC-SGD baseline using the existing utils_1.py.
    """
    calib_dir = EXTERNAL_PACKAGE_CONFIG["calib_package_dir"]
    if calib_dir not in sys.path:
        sys.path.append(calib_dir)

    try:
        from utils_1 import generate_mismatch_pool_tf
    except Exception as exc:
        raise ImportError(
            "MC-SGD requires utils_1.py from calib_package_dir. "
            "Remove 'mc_sgd' from SCHEMES if you do not need it."
        ) from exc

    A_BS_pool, A_UE_pool = generate_mismatch_pool_tf(
        Nt=Nt,
        Nr=Nr,
        std_amp_db=TEST_MISMATCH_CONFIG["std_amp_db"],
        std_phase_deg=TEST_MISMATCH_CONFIG["std_phase_deg"],
        num_samples=MCSGD_CONFIG["num_pool"],
        power_neutral=TEST_MISMATCH_CONFIG["power_neutral"],
        seed=MCSGD_CONFIG["seed"],
    )

    return A_BS_pool, A_UE_pool


def mcsgd_precoder(H_UL_est, dl_snr_linear, Ns, mcsgd_pool):
    """
    Call existing TensorFlow MC-SGD precoder and return numpy W.
    """
    calib_dir = EXTERNAL_PACKAGE_CONFIG["calib_package_dir"]
    if calib_dir not in sys.path:
        sys.path.append(calib_dir)

    from utils_1 import mc_sgd_precoder_from_uplink

    A_BS_pool, A_UE_pool = mcsgd_pool

    W_tf = mc_sgd_precoder_from_uplink(
        H_UL_obs=tf.constant(H_UL_est, dtype=tf.complex64),
        snr_linear=float(dl_snr_linear),
        A_BS_pool=A_BS_pool,
        A_UE_pool=A_UE_pool,
        Ns=Ns,
        num_iters=MCSGD_CONFIG["num_iters"],
        step_size=MCSGD_CONFIG["step_size"],
        mc_samples=MCSGD_CONFIG["mc_samples"],
        init=MCSGD_CONFIG["init"],
        unit="bits",
    )

    W = W_tf.numpy().astype(np.complex64)
    return normalize_precoder_equal_stream(W)


def compute_h_nmse(H_hat, H_true):
    err = np.sum(np.abs(H_hat - H_true) ** 2, axis=(1, 2))
    power = np.sum(np.abs(H_true) ** 2, axis=(1, 2)) + 1e-12
    return float(np.mean(err / power))

def _generate_genie_direct_precoder(
    H_DL_true,
    dl_snr_db,
    genie_direct_precoder,
):
    """
    Perfect-CSI direct link-aware precoder.

    H_DL_true
      -> Direct neural precoder
      -> W_final
    """
    if genie_direct_precoder is None:
        raise ValueError(
            "genie_direct_precoder must be provided "
            "when scheme='genie_direct_precoder'."
        )

    W = genie_direct_precoder(
        H_DL_true=H_DL_true,
        dl_snr_db=dl_snr_db,
    )

    h_nmse = None
    return W, h_nmse


def _generate_deeprc_cnn_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    deeprc_cnn_calibrator,
):
    """
    DeepRC-inspired CNN calibration + SVD precoding.

    H_UL_est -> DeepRC-CNN -> H_DL_hat -> SVD precoder -> W
    """
    if deeprc_cnn_calibrator is None:
        raise ValueError(
            "deeprc_cnn_calibrator must be provided "
            "when scheme='deeprc_cnn_svd'."
        )

    H_prec = deeprc_cnn_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"DeepRC-CNN output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}. "
            f"Expected H_prec to have shape [B, Nr, Nt]."
        )

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse



def _generate_nn_calib_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    nn_calibrator,
):
    """
    NN calibration + SVD equal-power precoding.

    H_UL_est -> NN calibrator -> H_DL_hat -> SVD precoder -> W
    """
    if nn_calibrator is None:
        raise ValueError(
            "nn_calibrator must be provided when scheme='nn_calib_svd'."
        )

    # 1. NN 校准得到下行信道估计
    H_prec = nn_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"NN calibrator output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}. "
            f"Expected H_prec to have shape [B, Nr, Nt]."
        )

    # 2. 用 NN 估计的 H_DL_hat 做 SVD 预编码
    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    # 3. 记录 NN 校准信道的 NMSE
    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse


def _generate_predcalinet_static_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    predcalinet_calibrator,
):
    """
    PredCaliNet-static calibration + SVD precoding.

    H_UL_est
      -> PredCaliNet-static
      -> H_DL_hat
      -> SVD precoder
    """
    if predcalinet_calibrator is None:
        raise ValueError(
            "predcalinet_calibrator must be provided "
            "when scheme='predcalinet_static_svd'."
        )

    H_prec = predcalinet_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"PredCaliNet-static output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}. "
            f"Expected H_prec to have shape [B, Nr, Nt]."
        )

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse


def _generate_sa_rescal_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    sa_mult_calibrator,
):
    """
    SA-ResCalNet calibration + SVD precoding.

    H_UL_est
      -> SA-ResCalNet
      -> H_DL_hat
      -> SVD precoder
    """
    if sa_mult_calibrator is None:
        raise ValueError(
            "sa_mult_calibrator must be provided "
            "when scheme='sa_mult_cal_svd'."
        )

    H_prec = sa_mult_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"SA-ResCalNet output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}."
        )

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse


def _generate_moe_calib_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    moe_calibrator,
):
    """
    MoE calibration + SVD equal-power precoding.

    H_UL_est -> MoE calibrator -> H_DL_hat -> SVD precoder -> W
    """
    if moe_calibrator is None:
        raise ValueError(
            "moe_calibrator must be provided when scheme='moe_calib_svd'."
        )

    # 1. MoE 校准得到下行信道估计
    H_prec = moe_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"MoE calibrator output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}. "
            f"Expected H_prec to have shape [B, Nr, Nt]."
        )

    # 2. 用 MoE 估计的 H_DL_hat 做 SVD 预编码
    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    # 3. 记录 MoE 校准信道的 NMSE
    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse


def _generate_moe_refiner_calib_svd_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    csi_snr_db,
    moe_refiner_calibrator,
):
    """
    MoE + residual refiner calibration + SVD precoding.

    H_UL_est
      -> Frozen MoE
      -> H_DL_base
      -> Refiner
      -> H_DL_hat
      -> SVD precoder
    """
    if moe_refiner_calibrator is None:
        raise ValueError(
            "moe_refiner_calibrator must be provided "
            "when scheme='moe_refiner_calib_svd'."
        )

    H_prec = moe_refiner_calibrator(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
    )

    H_prec = np.asarray(H_prec, dtype=np.complex64)

    if H_prec.shape != H_DL_true.shape:
        raise ValueError(
            f"MoE+Refiner output shape mismatch: "
            f"H_prec.shape={H_prec.shape}, "
            f"H_DL_true.shape={H_DL_true.shape}."
        )

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = compute_h_nmse(H_prec, H_DL_true)

    return W, h_nmse



def _generate_moe_precoder_refiner_precoder(
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    dl_snr_db,
    csi_snr_db,
    moe_precoder_refiner,
):
    """
    Frozen MoE + W-refiner directly outputs final W.

    H_UL_est
      -> MoE
      -> H_DL_hat
      -> SVD W_base
      -> W-refiner
      -> W_final
    """
    if moe_precoder_refiner is None:
        raise ValueError(
            "moe_precoder_refiner must be provided "
            "when scheme='moe_preco_refiner_svd'."
        )

    W = moe_precoder_refiner(
        H_UL_est=H_UL_est,
        csi_snr_db=csi_snr_db,
        dl_snr_db=dl_snr_db,
    )

    # 这个方案直接输出 W，没有新的 H_hat
    h_nmse = None

    return W, h_nmse


def build_qpsk_candidate_table(Ns):
    """
    Build all possible QPSK symbol vectors for ML detection.

    For Ns streams:
        number of candidates = 4^Ns

    return:
        candidate_symbols: [M, Ns] complex64
        candidate_bits:    [M, Ns, 2] int32
    """
    M = 4 ** Ns
    num_bits = 2 * Ns

    candidate_bits_flat = np.zeros((M, num_bits), dtype=np.int32)

    for m in range(M):
        bit_str = format(m, f"0{num_bits}b")
        candidate_bits_flat[m] = np.array(
            [int(b) for b in bit_str],
            dtype=np.int32,
        )

    candidate_bits = candidate_bits_flat.reshape(M, Ns, 2)

    b0 = candidate_bits[..., 0]
    b1 = candidate_bits[..., 1]

    real = 1.0 - 2.0 * b0
    imag = 1.0 - 2.0 * b1

    candidate_symbols = (
        real + 1j * imag
    ) / np.sqrt(2.0)

    return (
        candidate_symbols.astype(np.complex64),
        candidate_bits.astype(np.int32),
    )

def parse_scheme_and_detector(scheme):
    """
    Split scheme name into:
        base precoder scheme
        detector type

    Examples:
        ideal_svd        -> ideal_svd, mmse
        ideal_svd_ml     -> ideal_svd, ml
        moe_preco_refiner_svd_ml
                         -> moe_preco_refiner_svd, ml
    """
    if scheme.endswith("_ml"):
        return scheme[:-3], "ml"

    return scheme, "mmse"


# 最大似然检测器
def ml_detect_qpsk(
    y,
    H_DL_true,
    W,
):
    """
    ML detector for multi-stream QPSK.

    Model:
        y = H_DL_true W s + n
        G = H_DL_true W

    ML rule:
        s_hat = argmin_s || y - G s ||_2^2

    Args:
        y:         [B, Nr, T]
        H_DL_true: [B, Nr, Nt]
        W:         [B, Nt, Ns]

    Returns:
        bits_hat:  [B, Ns, T, 2]
    """
    y = np.asarray(y, dtype=np.complex64)
    H_DL_true = np.asarray(H_DL_true, dtype=np.complex64)
    W = np.asarray(W, dtype=np.complex64)

    B, Nr, T = y.shape
    Ns = W.shape[-1]

    # --------------------------------------------------------
    # 1. Effective channel G = H W
    # --------------------------------------------------------
    G = np.matmul(H_DL_true, W)  # [B, Nr, Ns]

    # --------------------------------------------------------
    # 2. All possible QPSK symbol vectors
    # --------------------------------------------------------
    candidate_symbols, candidate_bits = build_qpsk_candidate_table(Ns)
    # candidate_symbols: [M, Ns]
    # candidate_bits:    [M, Ns, 2]

    # --------------------------------------------------------
    # 3. Predicted received vector for every candidate
    #    y_candidate[b, m, r] = sum_s G[b,r,s] * s_candidate[m,s]
    # --------------------------------------------------------
    y_candidate = np.einsum(
        "brs,ms->bmr",
        G,
        candidate_symbols,
    )  # [B, M, Nr]

    # --------------------------------------------------------
    # 4. Compute Euclidean distance efficiently:
    #    ||y - y_c||^2
    #    = ||y||^2 + ||y_c||^2 - 2 Re(y^H y_c)
    # --------------------------------------------------------
    y_power = np.sum(
        np.abs(y) ** 2,
        axis=1,
    )[:, None, :]  # [B,1,T]

    candidate_power = np.sum(
        np.abs(y_candidate) ** 2,
        axis=2,
    )[:, :, None]  # [B,M,1]

    cross_term = np.real(
        np.einsum(
            "brt,bmr->bmt",
            np.conjugate(y),
            y_candidate,
        )
    )  # [B,M,T]

    distance = (
        y_power
        + candidate_power
        - 2.0 * cross_term
    )  # [B,M,T]

    # --------------------------------------------------------
    # 5. Pick the candidate with minimum distance
    # --------------------------------------------------------
    best_candidate_idx = np.argmin(
        distance,
        axis=1,
    )  # [B,T]

    # candidate_bits[best_candidate_idx]:
    # [B,T,Ns,2]
    bits_hat_bt = candidate_bits[best_candidate_idx]

    # Rearrange to match your original format:
    # [B,Ns,T,2]
    bits_hat = np.transpose(
        bits_hat_bt,
        (0, 2, 1, 3),
    )

    return bits_hat.astype(np.int32)



def compute_bler_wrapper(bits, bits_hat):
    """
    Compatible with either:
    compute_bler(bits, bits_hat)
    or
    compute_bler(bits, bits_hat, block_symbols=...)
    """

    bler = compute_bler(
    bits,
    bits_hat,
    block_symbols=LINK_CONFIG["bler_block_symbols"],
    )
    return bler


def _generate_ideal_svd_precoder(H_DL_true, Ns, dl_snr_linear):
    """
    生成理想SVD预编码器（使用真实的下行信道矩阵）
    
    参数:
        H_DL_true: 真实的下行信道矩阵
        Ns: 流数
        dl_snr_linear: 下行线性信噪比
    
    返回:
        W: 预编码器矩阵
        h_nmse: 信道归一化均方误差（理想情况下为0.0）
    """
    H_prec = H_DL_true

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = 0.0
    return W, h_nmse


def _generate_error_svd_precoder(H_UL_est, H_DL_true, Ns, dl_snr_linear):
    """
    生成基于上行估计的SVD预编码器（考虑信道估计误差）
    
    参数:
        H_UL_est: 估计的上行信道矩阵
        H_DL_true: 真实的下行信道矩阵
        Ns: 流数
        dl_snr_linear: 下行线性信噪比
    
    返回:
        W: 预编码器矩阵
        h_nmse: 信道归一化均方误差
    """
    H_prec = np.transpose(H_UL_est, (0, 2, 1)).astype(np.complex64)

    W = build_precoder(
        H_prec=H_prec,
        Ns=Ns,
        precoder_type=LINK_CONFIG["precoder_type"],
        snr_linear=dl_snr_linear,
    )

    h_nmse = compute_h_nmse(H_prec, H_DL_true)
    return W, h_nmse


def _generate_mc_sgd_precoder(H_UL_est, dl_snr_linear, Ns, mcsgd_pool):
    """
    生成基于蒙特卡洛随机梯度下降的预编码器（考虑信道失配分布）
    
    参数:
        H_UL_est: 估计的上行信道矩阵
        dl_snr_linear: 下行线性信噪比
        Ns: 流数
        mcsgd_pool: MC-SGD预编码器的采样池
    
    返回:
        W: 预编码器矩阵
        h_nmse: 信道归一化均方误差（MC-SGD直接输出W，无H_hat，故为None）
    """
    if mcsgd_pool is None:
        raise ValueError("mcsgd_pool must be provided when scheme='mc_sgd'.")

    W = mcsgd_precoder(
        H_UL_est=H_UL_est,
        dl_snr_linear=dl_snr_linear,
        Ns=Ns,
        mcsgd_pool=mcsgd_pool,
    )

    # MC-SGD directly outputs W, so there is no H_hat.
    h_nmse = None
    return W, h_nmse



def generate_precoder_for_scheme(
    scheme,
    H_UL_est,
    H_DL_true,
    Ns,
    dl_snr_linear,
    dl_snr_db,
    csi_snr_db,
    mcsgd_pool=None,
    nn_calibrator=None,
    moe_calibrator=None,
    moe_refiner_calibrator=None,
    moe_precoder_refiner=None,
    genie_direct_precoder=None,
    sa_mult_calibrators=None,
    predcalinet_calibrator=None,
    deeprc_cnn_calibrator=None,
):
    """
    根据指定方案生成预编码器的统一接口
    
    支持的方案:
        ideal_svd: 使用真实的下行信道矩阵进行SVD分解生成预编码器
        error_svd: 使用估计的上行信道矩阵转置进行SVD分解生成预编码器（考虑信道估计误差）
        mc_sgd: 使用蒙特卡洛随机梯度下降方法生成预编码器（考虑信道失配分布）
    
    参数:
        scheme: 预编码器生成方案，可选值为"ideal_svd"、"error_svd"或"mc_sgd"
        H_UL_est: 估计的上行信道矩阵
        H_DL_true: 真实的下行信道矩阵
        Ns: 流数
        dl_snr_linear: 下行线性信噪比
        mcsgd_pool: MC-SGD预编码器的采样池（仅当scheme="mc_sgd"时需要）
        nn_calibrator: NN校准器（仅当scheme="nn_calib_svd"时需要）
        moe_calibrator: MoE校准器（仅当scheme="moe_calib_svd"时需要）
        predcalinet_calibrator: PredCaliNet-static校准器（仅当scheme="predcalinet_static_svd"时需要）

    返回:
        W: 预编码器矩阵
        h_nmse: 信道归一化均方误差
    """
    if scheme == "ideal_svd":
        return _generate_ideal_svd_precoder(H_DL_true, Ns, dl_snr_linear)

    if scheme == "error_svd":
        return _generate_error_svd_precoder(H_UL_est, H_DL_true, Ns, dl_snr_linear)

    if scheme == "mc_sgd":
        return _generate_mc_sgd_precoder(H_UL_est, dl_snr_linear, Ns, mcsgd_pool)
    
    if scheme == "nn_calib_svd":
        return _generate_nn_calib_svd_precoder(H_UL_est, H_DL_true, Ns, dl_snr_linear, csi_snr_db,
                                                nn_calibrator)
    
    if scheme == "predcalinet_static_svd":
        return _generate_predcalinet_static_svd_precoder(H_UL_est, H_DL_true, Ns, dl_snr_linear, csi_snr_db,
                                                predcalinet_calibrator)
    
    if scheme == "moe_calib_svd":
        return _generate_moe_calib_svd_precoder(H_UL_est, H_DL_true, Ns, dl_snr_linear, csi_snr_db,
                                                moe_calibrator)
    if scheme == "moe_refiner_calib_svd":
        return _generate_moe_refiner_calib_svd_precoder(
            H_UL_est,
            H_DL_true,
            Ns,
            dl_snr_linear,
            csi_snr_db,
            moe_refiner_calibrator,
        )
    if scheme == "moe_preco_refiner_svd":
        return _generate_moe_precoder_refiner_precoder(
            H_UL_est,
            H_DL_true,
            Ns,
            dl_snr_linear,
            dl_snr_db,
            csi_snr_db,
            moe_precoder_refiner,
        )
    if scheme == "genie_direct_precoder":
        return _generate_genie_direct_precoder(
            H_DL_true=H_DL_true,
            dl_snr_db=dl_snr_db,
            genie_direct_precoder=genie_direct_precoder,
        )
    
    if scheme == "deeprc_cnn_svd":
        return _generate_deeprc_cnn_svd_precoder(
            H_UL_est=H_UL_est,
            H_DL_true=H_DL_true,
            Ns=Ns,
            dl_snr_linear=dl_snr_linear,
            csi_snr_db=csi_snr_db,
            deeprc_cnn_calibrator=deeprc_cnn_calibrator,
        )

   


    # if scheme == "sa_rescal_svd":
    #     return _generate_sa_rescal_svd_precoder(
    #         H_UL_est=H_UL_est,
    #         H_DL_true=H_DL_true,
    #         Ns=Ns,
    #         dl_snr_linear=dl_snr_linear,
    #         csi_snr_db=csi_snr_db,
    #         sa_mult_calibrator=sa_mult_calibrator,
    #     )
    if scheme in SA_MULT_SCHEME_TO_VARIANT:
        if sa_mult_calibrators is None or scheme not in sa_mult_calibrators:
            raise ValueError(
                f"Missing SA-Mult calibrator for scheme={scheme}. "
                f"Available: {None if sa_mult_calibrators is None else list(sa_mult_calibrators.keys())}"
            )

        return _generate_sa_rescal_svd_precoder(
            H_UL_est=H_UL_est,
            H_DL_true=H_DL_true,
            Ns=Ns,
            dl_snr_linear=dl_snr_linear,
            csi_snr_db=csi_snr_db,
            sa_mult_calibrator=sa_mult_calibrators[scheme],
        )

    raise ValueError(f"Unknown scheme: {scheme}")


def evaluate_one_scheme(
    scheme,
    H_test,
    C_BS,
    C_UE,
    Nt,
    Nr,
    Ns,
    dl_snr_db,
    dl_snr_linear,
    rng,
    mcsgd_pool=None,
    nn_calibrator=None,
    predcalinet_calibrator=None,
    moe_calibrator=None,
    moe_refiner_calibrator=None,
    moe_precoder_refiner=None,
    genie_direct_precoder=None,
    sa_mult_calibrators=None,
    deeprc_cnn_calibrator=None,
    batch_size=256,
):
    ber_sum = 0.0
    ser_sum = 0.0
    bler_sum = 0.0
    evm_sum = 0.0
    effse_sum = 0.0
    cap_sum = 0.0
    gram_leakage_sum = 0.0
    post_mmse_sinr_mean_db_sum = 0.0
    post_mmse_sinr_min_db_sum = 0.0

    h_nmse_sum = 0.0
    h_nmse_count = 0

    # 推理时间：只统计 H_UL_est -> W
    inference_time_total_sec = 0.0
    inference_num_samples = 0

    # 第一批用于模型预热，不计入时间
    timing_warmup_batches = 1

    num_batches = 0
    num_samples = H_test.shape[0]

    csi_snr_db = get_csi_snr_for_dl_snr(dl_snr_db)
    base_scheme, detector_type = parse_scheme_and_detector(scheme)

    for batch_idx, start in enumerate(
        range(0, num_samples, batch_size)
    ):  
        end = min(start + batch_size, num_samples)

        H_UL_clean = H_test[start:end]
        B = H_UL_clean.shape[0]

        # True downlink channel
        H_DL_true = build_downlink_from_uplink(
            H_UL_clean,
            C_BS,
            C_UE,
        )

        # Noisy uplink CSI
        H_UL_est = add_awgn_to_uplink(
            H_UL_clean,
            csi_snr_db,
        )

        # ----------------------------------------------------
        # Generate precoder and measure online inference time
        #
        # Timing range:
        # H_UL_est -> calibration/network -> SVD -> W
        # ----------------------------------------------------
        synchronize_tensorflow()
        inference_start_time = time.perf_counter()

        W, h_nmse = generate_precoder_for_scheme(
            scheme=base_scheme,
            H_UL_est=H_UL_est,
            H_DL_true=H_DL_true,
            Ns=Ns,
            dl_snr_linear=dl_snr_linear,
            dl_snr_db=dl_snr_db,
            csi_snr_db=csi_snr_db,
            mcsgd_pool=mcsgd_pool,
            nn_calibrator=nn_calibrator,
            predcalinet_calibrator=predcalinet_calibrator,
            moe_calibrator=moe_calibrator,
            moe_refiner_calibrator=moe_refiner_calibrator,
            moe_precoder_refiner=moe_precoder_refiner,
            genie_direct_precoder=genie_direct_precoder,
            sa_mult_calibrators=sa_mult_calibrators,
            deeprc_cnn_calibrator=deeprc_cnn_calibrator,
        )

        # Converting the output to NumPy also forces the result
        # to be available before stopping the timer.
        W = np.asarray(W)

        synchronize_tensorflow()
        inference_elapsed_sec = (
            time.perf_counter() - inference_start_time
        )

        # Exclude the first batch as warm-up.
        if batch_idx >= timing_warmup_batches:
            inference_time_total_sec += inference_elapsed_sec
            inference_num_samples += B

        # Generate 4QAM/QPSK symbols
        bits = generate_random_bits(
            batch_size=B,
            Ns=Ns,
            num_symbols=LINK_CONFIG["num_symbols"],
            bits_per_symbol=LINK_CONFIG["bits_per_symbol"],
            rng=rng,
        )

        s = qam4_modulate(bits)

        # Downlink transmission
        y, noise_var = downlink_transmit(
            H_DL_true=H_DL_true,
            W=W,
            s=s,
            dl_snr_db=dl_snr_db,
            tx_power=LINK_CONFIG["tx_power"],
        )

        # ----------------------------------------------------
        # Diagnostic metrics:
        # 1) Inter-stream Gram leakage
        # 2) Post-MMSE average SINR
        # 3) Post-MMSE weakest-stream SINR
        # ----------------------------------------------------
        gram_leakage = compute_gram_leakage(
            H_DL_true=H_DL_true,
            W=W,
        )

        post_mmse_sinr_mean_db, post_mmse_sinr_min_db = compute_post_mmse_sinr_stats(
            H_DL_true=H_DL_true,
            W=W,
            noise_var=noise_var,
        )

        # ----------------------------------------------------
        # Receiver / detector
        # ----------------------------------------------------
        if detector_type == "mmse":
            # Original receiver:
            # MMSE equalization + hard QPSK demapping
            s_hat = mmse_equalize(
                y=y,
                H_DL_true=H_DL_true,
                W=W,
                noise_var=noise_var,
            )

            bits_hat = qam4_hard_demod(s_hat)

        elif detector_type == "ml":
            # ML discrete detector:
            # Directly detect the most likely QPSK stream vector
            bits_hat = ml_detect_qpsk(
                y=y,
                H_DL_true=H_DL_true,
                W=W,
            )

            # Re-modulate hard decisions only for compatibility with EVM code.
            # 注意：ML 曲线的 EVM 不建议作为核心比较指标；
            # 这一轮主要看 BER / SER / BLER。
            s_hat = qam4_modulate(bits_hat)

        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        ber = compute_ber(bits, bits_hat)
        ser = compute_ser(bits, bits_hat)
        bler = compute_bler_wrapper(bits, bits_hat)
        evm = compute_evm(s, s_hat)
        effse = compute_effective_se(
            bler,
            Ns,
            LINK_CONFIG["bits_per_symbol"],
        )
        cap = gaussian_capacity_with_precoder(
            H_DL_true,
            W,
            dl_snr_linear,
        )

        ber_sum += ber
        ser_sum += ser
        bler_sum += bler
        evm_sum += evm
        effse_sum += effse
        cap_sum += cap
        gram_leakage_sum += gram_leakage
        post_mmse_sinr_mean_db_sum += post_mmse_sinr_mean_db
        post_mmse_sinr_min_db_sum += post_mmse_sinr_min_db

        if h_nmse is not None:
            h_nmse_sum += h_nmse
            h_nmse_count += 1

        num_batches += 1

    result = {
        "ber": ber_sum / num_batches,
        "ser": ser_sum / num_batches,
        "bler": bler_sum / num_batches,
        "evm": evm_sum / num_batches,
        "effse": effse_sum / num_batches,
        "cap_link": cap_sum / num_batches,
        "gram_leakage": gram_leakage_sum / num_batches,
        "post_mmse_sinr_mean_db": (
            post_mmse_sinr_mean_db_sum / num_batches
        ),
        "post_mmse_sinr_min_db": (
            post_mmse_sinr_min_db_sum / num_batches
        ),

        # 当前SNR点下，所有计时样本的总推理时间
        "inference_time_total_sec": float(
            inference_time_total_sec
        ),

        # 当前SNR点下实际参与计时的信道数量
        "inference_num_channels": int(
            inference_num_samples
        ),

        # 当前SNR点下平均每个信道的时间
        "inference_time_ms_per_channel": (
            1000.0
            * inference_time_total_sec
            / inference_num_samples
            if inference_num_samples > 0
            else None
        ),
    }

    if h_nmse_count > 0:
        result["h_nmse"] = h_nmse_sum / h_nmse_count
    else:
        result["h_nmse"] = None

    return result

def get_parameter_count_for_scheme(
    scheme,
    nn_calibrator=None,
    predcalinet_calibrator=None,
    deeprc_cnn_calibrator=None,
    moe_calibrator=None,
    moe_refiner_calibrator=None,
    moe_precoder_refiner=None,
    genie_direct_precoder=None,
    sa_mult_calibrators=None,
):
    """
    Return total model parameters for one scheme.
    """
    base_scheme, _ = parse_scheme_and_detector(scheme)

    # Non-learning methods
    if base_scheme in {
        "ideal_svd",
        "error_svd",
        "mc_sgd",
    }:
        return 0

    if base_scheme == "nn_calib_svd":
        return count_total_parameters(
            nn_calibrator
        )

    if base_scheme == "predcalinet_static_svd":
        return count_total_parameters(
            predcalinet_calibrator
        )

    if base_scheme == "deeprc_cnn_svd":
        return count_total_parameters(
            deeprc_cnn_calibrator
        )

    if base_scheme == "moe_calib_svd":
        return count_total_parameters(
            moe_calibrator
        )

    if base_scheme == "moe_refiner_calib_svd":
        return count_total_parameters(
            moe_refiner_calibrator
        )

    if base_scheme == "moe_preco_refiner_svd":
        return count_total_parameters(
            moe_precoder_refiner
        )

    if base_scheme == "genie_direct_precoder":
        return count_total_parameters(
            genie_direct_precoder
        )

    if base_scheme in SA_MULT_SCHEME_TO_VARIANT:
        if (
            sa_mult_calibrators is None
            or base_scheme not in sa_mult_calibrators
        ):
            return 0

        return count_total_parameters(
            sa_mult_calibrators[base_scheme]
        )

    return 0



def main():
    args = parse_args()

    # ----------------------------------------------------
    # Build schemes for this evaluation run
    # ----------------------------------------------------
    eval_schemes = [
        "ideal_svd",
        "error_svd",
    ]

    if args.include_nn:
        eval_schemes.append("nn_calib_svd")

    if args.include_predcalinet:
        eval_schemes.append("predcalinet_static_svd")

    if args.include_deeprc_cnn:
        eval_schemes.append("deeprc_cnn_svd")

    if args.include_mcsgd:
        eval_schemes.append("mc_sgd")

    if args.include_moe:
        eval_schemes.extend([
            "moe_calib_svd",
            "moe_refiner_calib_svd",
        ])


    if args.eval_variant == "all":
        eval_schemes.extend([
            "sa_mult_nores_nocond_svd",
            "sa_mult_nores_svd",
            "sa_mult_full_svd",
        ])
    else:
        eval_schemes.append(
            SA_MULT_VARIANT_TO_SCHEME[args.eval_variant]
        )

    print("Loading dataset...")

    _, _, test_data, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]
    Ns = LINK_CONFIG.get("num_streams", Nr)

    print("Test:", test_data.shape)
    print(f"Nt={Nt}, Nr={Nr}, Ns={Ns}")
    print("Schemes:", eval_schemes)
    print("Precoder type:", LINK_CONFIG["precoder_type"])

    mismatch_path = os.path.join(
        PATH_CONFIG["results_dir"],
        f"fixed_mismatch_{TEST_EXPERIMENT_TAG}.npz",
    )

    C_BS, C_UE = generate_fixed_mismatch(
        Nt=Nt,
        Nr=Nr,
        std_amp_db=TEST_MISMATCH_CONFIG["std_amp_db"],
        std_phase_deg=TEST_MISMATCH_CONFIG["std_phase_deg"],
        power_neutral=TEST_MISMATCH_CONFIG["power_neutral"],
        seed=42,
        save_path=mismatch_path,
        force_regenerate=False,
    )

    mcsgd_pool = None
    if "mc_sgd" in eval_schemes:
        print("Building MC-SGD mismatch pool...")
        mcsgd_pool = build_mcsgd_mismatch_pool(Nt=Nt, Nr=Nr)
    
    # ===== Optional NN calibration module =====
    nn_calibrator = None
    if "nn_calib_svd" in eval_schemes:
        print("Loading NN calibration module...")
        from nn_calibration import NNCalibrator

        nn_calibrator = NNCalibrator(
            Nt=Nt,
            Nr=Nr,
        )

    # ===== Optional PredCaliNet-static calibration module =====
    predcalinet_calibrator = None
    if "predcalinet_static_svd" in eval_schemes:
        print("Loading PredCaliNet-static calibration module...")
        from predcalinet_calibration import PredCaliNetCalibrator

        predcalinet_calibrator = PredCaliNetCalibrator(
            Nt=Nt,
            Nr=Nr,
        )

    # ===== Optional MoE calibration module =====
    moe_calibrator = None
    if "moe_calib_svd" in eval_schemes:
        print("Loading MoE calibration module...")
        from MOE_calibration import MoECalibrator

        moe_calibrator = MoECalibrator(
            Nt=Nt,
            Nr=Nr,
        )
    
    moe_refiner_calibrator = None
    if "moe_refiner_calib_svd" in eval_schemes:
        print("Loading MoE+Refiner calibration module...")
        from residual_refiner_calibration import MoEResidualRefinerCalibrator

        moe_refiner_calibrator = MoEResidualRefinerCalibrator(
            Nt=Nt,
            Nr=Nr,
        )
    
    moe_precoder_refiner = None
    if "moe_preco_refiner_svd" in eval_schemes:
        print("Loading MoE+Precoder-Refiner calibration module...")
        from precoder_refiner_calibration import MoEPrecoderRefinerCalibrator

        moe_precoder_refiner = MoEPrecoderRefinerCalibrator(
            Nt=Nt,
            Nr=Nr,
            Ns=Ns,
        )
    
    genie_direct_precoder = None
    if "genie_direct_precoder" in eval_schemes:
        print("Loading Perfect-CSI direct link-aware precoder module...")
        from genie_direct_precoder_calibration import GenieDirectPrecoderCalibrator

        genie_direct_precoder = GenieDirectPrecoderCalibrator(
            Nt=Nt,
            Nr=Nr,
            Ns=Ns,
        )

    deeprc_cnn_calibrator = None
    if "deeprc_cnn_svd" in eval_schemes:
        print("Loading DeepRC-inspired CNN calibration module...")
        from deeprc_cnn_runtime import DeepRCCNNCalibrator

        deeprc_cnn_calibrator = DeepRCCNNCalibrator(
            Nt=Nt,
            Nr=Nr,
        )
   
    # ===== SA-Mult-Cal variants =====
    sa_mult_calibrators = {}

    if any(scheme in eval_schemes for scheme in SA_MULT_SCHEME_TO_VARIANT):
        print("Loading SA-Mult calibration modules...")

        from sa_mult_calibration import SAMultCalibrator

        for scheme_name, variant in SA_MULT_SCHEME_TO_VARIANT.items():
            if scheme_name not in eval_schemes:
                continue

            cfg = SA_MULT_CONFIGS[variant]

            print(f"  Loading {scheme_name}")
            print(f"    variant      = {variant}")
            print(f"    weights_path = {cfg['weights_path']}")
            print(f"    eta_add      = {cfg.get('eta_add', 0.0)}")
            print(f"    condition    = {cfg.get('condition_mode', 'full')}")

            sa_mult_calibrators[scheme_name] = SAMultCalibrator(
                Nt=Nt,
                Nr=Nr,
                config=cfg,
            )

    # ====================================================
    # Count model parameters
    # ====================================================
    scheme_parameter_counts = {}

    print("\n" + "=" * 70)
    print("Model parameter counts")
    print("=" * 70)

    for scheme in eval_schemes:
        parameter_count = get_parameter_count_for_scheme(
            scheme=scheme,
            nn_calibrator=nn_calibrator,
            predcalinet_calibrator=predcalinet_calibrator,
            deeprc_cnn_calibrator=deeprc_cnn_calibrator,
            moe_calibrator=moe_calibrator,
            moe_refiner_calibrator=moe_refiner_calibrator,
            moe_precoder_refiner=moe_precoder_refiner,
            genie_direct_precoder=genie_direct_precoder,
            sa_mult_calibrators=sa_mult_calibrators,
        )

        scheme_parameter_counts[scheme] = int(
            parameter_count
        )

        if parameter_count == 0:
            parameter_text = "--"
        elif parameter_count >= 1_000_000:
            parameter_text = (
                f"{parameter_count:,} "
                f"({parameter_count / 1_000_000:.4f} M)"
            )
        elif parameter_count >= 1_000:
            parameter_text = (
                f"{parameter_count:,} "
                f"({parameter_count / 1_000:.4f} K)"
            )
        else:
            parameter_text = f"{parameter_count:,}"

        print(
            f"{scheme:<32s}: {parameter_text}"
        )

    print("=" * 70 + "\n")



    rng = np.random.default_rng(LINK_CONFIG["seed"])




    # Each row corresponds to one scheme at one SNR point.
    csv_rows = []
    results = {
        "snr_db": SNR_CONFIG["dl_snr_dB_list"],
        "schemes": {},
        "parameter_counts": scheme_parameter_counts,
        "config": {
            "eval_variant": args.eval_variant,
            "eval_schemes": eval_schemes,
            "test_mismatch": TEST_MISMATCH_CONFIG,
            "precoder_type": LINK_CONFIG["precoder_type"],
            "num_symbols": LINK_CONFIG["num_symbols"],
            "bits_per_symbol": LINK_CONFIG["bits_per_symbol"],
            "num_streams": Ns,
            "csi_snr_mode": SNR_CONFIG["csi_snr_mode"],
            "fixed_csi_snr_db": SNR_CONFIG["fixed_csi_snr_db"],
            "bler_block_symbols": LINK_CONFIG.get("bler_block_symbols", None),
        }
    }

    for scheme in eval_schemes:
        base_scheme, detector_type = parse_scheme_and_detector(scheme)
        print(f"  Precoder scheme: {base_scheme}")
        print(f"  Detector type: {detector_type}")

        results["schemes"][scheme] = {
            "num_parameters": int(scheme_parameter_counts[scheme]),
            "ber": [],
            "ser": [],
            "bler": [],
            "evm": [],
            "effse": [],
            "cap_link": [],
            "h_nmse": [],
            "gram_leakage": [],
            "post_mmse_sinr_mean_db": [],
            "post_mmse_sinr_min_db": [],

            # 每个SNR点的推理统计
            "inference_time_total_sec": [],
            "inference_num_channels": [],
            "inference_time_ms_per_channel": [],
        }

        for dl_snr_db, dl_snr_linear in zip(
            SNR_CONFIG["dl_snr_dB_list"],
            SNR_CONFIG["dl_snr_linear_list"],
        ):
            print(f"  DL SNR = {dl_snr_db} dB")

            metrics = evaluate_one_scheme(
                scheme=scheme,
                H_test=test_data,
                C_BS=C_BS,
                C_UE=C_UE,
                Nt=Nt,
                Nr=Nr,
                Ns=Ns,
                dl_snr_db=dl_snr_db,
                dl_snr_linear=dl_snr_linear,
                rng=rng,
                mcsgd_pool=mcsgd_pool,
                nn_calibrator=nn_calibrator,
                predcalinet_calibrator=predcalinet_calibrator,
                moe_calibrator=moe_calibrator,
                moe_refiner_calibrator=moe_refiner_calibrator,
                moe_precoder_refiner=moe_precoder_refiner,
                genie_direct_precoder=genie_direct_precoder,
                sa_mult_calibrators=sa_mult_calibrators,
                deeprc_cnn_calibrator=deeprc_cnn_calibrator,
                batch_size=LINK_CONFIG.get("batch_size", 256),
            )

            for key, value in metrics.items():
                if value is None:
                    saved_value = None
                elif key == "inference_num_channels":
                    saved_value = int(value)
                else:
                    saved_value = float(value)

                results["schemes"][scheme][key].append(
                    saved_value
                )

                        # ------------------------------------------------
            # Save one scheme/SNR result as one CSV row
            # ------------------------------------------------
            csv_row = {
                "scheme": scheme,
                "dl_snr_db": float(dl_snr_db),
                "csi_snr_db": float(
                    get_csi_snr_for_dl_snr(dl_snr_db)
                ),
                "batch_size": int(
                    LINK_CONFIG.get("batch_size", 256)
                ),
                "ber": metrics["ber"],
                "ser": metrics["ser"],
                "bler": metrics["bler"],
                "evm": metrics["evm"],
                "effse": metrics["effse"],
                "cap_link": metrics["cap_link"],
                "h_nmse": metrics["h_nmse"],
                "gram_leakage": metrics["gram_leakage"],
                "post_mmse_sinr_mean_db": (
                    metrics["post_mmse_sinr_mean_db"]
                ),
                "post_mmse_sinr_min_db": (
                    metrics["post_mmse_sinr_min_db"]
                ),
                "inference_time_ms_per_channel": (
                    metrics["inference_time_ms_per_channel"]
                ),
            }

            csv_rows.append(csv_row)

            nmse_text = (
                "N/A"
                if metrics["h_nmse"] is None
                else f"{metrics['h_nmse']:.4f}"
            )

            inference_text = (
                "N/A"
                if metrics["inference_time_ms_per_channel"] is None
                else (
                    f"{metrics['inference_time_ms_per_channel']:.6f}"
                )
            )

            print(
                f"    Cap={metrics['cap_link']:.4f}, "
                f"EffSE={metrics['effse']:.4f}, "
                f"BER={metrics['ber']:.4e}, "
                f"BLER={metrics['bler']:.4e}, "
                f"SER={metrics['ser']:.4e}, "
                f"EVM={metrics['evm']:.4f}, "
                f"H-NMSE={nmse_text}, "
                f"Gram Leakage={metrics['gram_leakage']:.4f}, "
                f"Post-MMSE SINR (Mean)={metrics['post_mmse_sinr_mean_db']:.2f} dB, "
                f"Post-MMSE SINR (Min)="
                f"{metrics['post_mmse_sinr_min_db']:.2f} dB, "
                f"Inference={inference_text} ms/channel"
            )

            # ====================================================
            # Overall inference statistics over all SNR points
            # ====================================================
        total_inference_time_sec_all_snrs = float(
            np.sum(
                results["schemes"][scheme][
                    "inference_time_total_sec"
                ]
            )
        )

        total_inference_channels_all_snrs = int(
            np.sum(
                results["schemes"][scheme][
                    "inference_num_channels"
                ]
            )
        )

        if total_inference_channels_all_snrs > 0:
            overall_inference_time_ms_per_channel = (
                1000.0
                * total_inference_time_sec_all_snrs
                / total_inference_channels_all_snrs
            )
        else:
            overall_inference_time_ms_per_channel = None

        results["schemes"][scheme][
            "total_inference_time_sec_all_snrs"
        ] = total_inference_time_sec_all_snrs

        results["schemes"][scheme][
            "total_inference_channels_all_snrs"
        ] = total_inference_channels_all_snrs

        results["schemes"][scheme][
            "overall_inference_time_ms_per_channel"
        ] = overall_inference_time_ms_per_channel

        print(
            f"\n  Overall inference statistics of {scheme}:"
        )

        print(
                f"    Total inference time over all SNRs: "
                f"{total_inference_time_sec_all_snrs:.6f} s"
            )

        print(
            f"    Total timed channels over all SNRs: "
            f"{total_inference_channels_all_snrs}"
        )

        if overall_inference_time_ms_per_channel is not None:
            print(
                f"    Overall time per channel: "
                f"{overall_inference_time_ms_per_channel:.6f} ms/channel\n"
            )
        else:
            print(
                "    Overall time per channel: N/A\n"
            )

        # 注意：放在SNR循环外面，但仍在scheme循环里面
        inference_times = results["schemes"][scheme][
            "inference_time_ms_per_channel"
        ]

        valid_inference_times = [
            value for value in inference_times
            if value is not None
        ]

        if valid_inference_times:
            average_inference_time = float(
                np.mean(valid_inference_times)
            )

            results["schemes"][scheme][
                "average_inference_time_ms_per_channel"
            ] = average_inference_time

            print(
                f"\n  Average inference time of {scheme}: "
                f"{average_inference_time:.6f} ms/channel\n"
            )

    result_tag = SA_MULT_CONFIGS.get("tag", "default")

    if args.eval_variant == "all":
        result_tag = "sa_mult_ablation_all"
    else:
        result_tag = f"sa_mult_{args.eval_variant}"

    result_path = os.path.join(
        PATH_CONFIG["results_dir"],
        f"e2e_link_{TEST_EXPERIMENT_TAG}_{LINK_CONFIG['precoder_type']}_{result_tag}.json",
    )

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {result_path}")


if __name__ == "__main__":
    main()