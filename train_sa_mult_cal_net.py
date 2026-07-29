# 完整的训练代码
import os
import sys
import json
import time
import gc
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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



from config_system import (
    DATA_CONFIG,
    PATH_CONFIG,
    SA_MULT_CONFIGS,
    SA_MULT_TRAIN_CONFIGS,
    TEST_MISMATCH_CONFIG,
    CALINET_CONFIG,
    CALINET_TRAIN_CONFIG,
    PREDCALINET_CONFIG,
    PREDCALINET_TRAIN_CONFIG,
    DEEPRC_CNN_CONFIG,
    DEEPRC_CNN_TRAIN_CONFIG
)



from channels import (
    load_and_split_complex_data,
    generate_fixed_mismatch,
    build_downlink_from_uplink,
    add_awgn_to_uplink,
)

from sa_mult_cal_net import (
    SAMultCalNet,
    complex_to_ri_np,
    total_sa_mult_cal_loss,
)

from calinet_dnn_calibration import (
    CalinetDNN,
    complex_to_ri_np as calinet_complex_to_ri_np,
    total_calinet_loss,
)


from predcalinet_static_calibration import (
    PredCaliNetStatic,
    complex_to_ri_np as pred_complex_to_ri_np,
    total_predcalinet_loss,
)

from deeprc_cnn_calibration import (
    DeepRCCNN,
    complex_to_ri_np as deeprc_complex_to_ri_np,
    total_deeprc_cnn_loss,
)

# ============================================================
# Fixed mismatch training grid
# ============================================================

MISMATCH_GRID = [
    (2.0, 20.0),
    (2.0, 40.0),
    (2.0, 60.0),
    (2.0, 90.0),

    (4.0, 20.0),
    (4.0, 40.0),
    (4.0, 60.0),
    (4.0, 90.0),
]

POWER_NEUTRAL = TEST_MISMATCH_CONFIG.get("power_neutral", True)
SEED = 42

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="sa_mult",
        choices=["sa_mult", "calinet", "predcalinet", "deeprc_cnn"],
        help="Which calibration model to train.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["full", "nores", "nocond", "nores_nogram"],
        help="SA-Mult variant: full, nores, nocond, or nores_nogram.",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def make_xi_key(std_amp_db, std_phase_deg):
    return f"A{float(std_amp_db):.1f}_P{float(std_phase_deg):.1f}"

def make_deeprc_cnn_batch(
    H_UL_clean_batch,
    C_BS,
    C_UE,
    csi_snr_db,
):
    H_DL_true = build_downlink_from_uplink(
        H_UL_clean_batch,
        C_BS,
        C_UE,
    )

    H_UL_est = add_awgn_to_uplink(
        H_UL_clean_batch,
        csi_snr_db,
    )

    H0 = np.transpose(H_UL_est, (0, 2, 1)).astype(np.complex64)

    H0_ri = deeprc_complex_to_ri_np(H0)
    H_DL_true_ri = deeprc_complex_to_ri_np(H_DL_true)

    return H0_ri, H_DL_true_ri


def make_predcalinet_batch(
    H_UL_clean_batch,
    C_BS,
    C_UE,
    csi_snr_db,
):
    """
    PredCaliNet-static baseline batch.

    H_UL_clean_batch: [B, Nt, Nr] complex

    Return:
        H0_ri:        [B, Nr, Nt, 2]
        H_DL_true_ri: [B, Nr, Nt, 2]
    """

    H_DL_true = build_downlink_from_uplink(
        H_UL_clean_batch,
        C_BS,
        C_UE,
    )

    H_UL_est = add_awgn_to_uplink(
        H_UL_clean_batch,
        csi_snr_db,
    )

    # H0 = H_UL_est^T
    H0 = np.transpose(H_UL_est, (0, 2, 1)).astype(np.complex64)

    H0_ri = pred_complex_to_ri_np(H0)
    H_DL_true_ri = pred_complex_to_ri_np(H_DL_true)

    return H0_ri, H_DL_true_ri




def make_predcalinet_optimizer(train_cfg):
    opt_name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))

    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    elif opt_name == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def train_predcalinet(args):
    cfg = PREDCALINET_CONFIG
    train_cfg = PREDCALINET_TRAIN_CONFIG

    print("\n========================================")
    print("Training baseline: PredCaliNet-static")
    print(f"Weights path: {cfg['weights_path']}")
    print(f"Hidden dim: {cfg.get('hidden_dim', 256)}")
    print(f"Num LSTM layers: {cfg.get('num_lstm_layers', 3)}")
    print(f"Loss type: {train_cfg.get('loss_type', 'nmse')}")
    print(f"Optimizer: {train_cfg.get('optimizer', 'adam')}")
    print("========================================\n")

    print("Loading dataset...")

    train_data, val_data, _, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]

    print("Train:", train_data.shape)
    print("Val:", val_data.shape)
    print(f"Nt={Nt}, Nr={Nr}")

    rng = np.random.default_rng(SEED)

    mismatch_cache = build_mismatch_cache(Nt=Nt, Nr=Nr)
    xi_keys = list(mismatch_cache.keys())

    model = PredCaliNetStatic(
        Nr=Nr,
        Nt=Nt,
        hidden_dim=cfg.get("hidden_dim", 256),
        num_lstm_layers=cfg.get("num_lstm_layers", 3),
        fc_dims=cfg.get("fc_dims", None),
        activation=cfg.get("activation", "tanh"),
        output_activation=cfg.get("output_activation", None),
    )

    # Build model
    dummy = tf.zeros([1, Nr, Nt, 2], dtype=tf.float32)
    _ = model(dummy, training=False)

    optimizer = make_predcalinet_optimizer(train_cfg)

    save_path = cfg["weights_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]

    best_val_nmse = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        perm = rng.permutation(len(train_data))

        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_nmse_sum = 0.0
        train_count = 0

        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            idx = perm[start:end]

            H_UL_clean_batch = train_data[idx]
            bsz = H_UL_clean_batch.shape[0]

            # 和 Calinet / SA-Mult 一样，从固定 mismatch grid 中采样训练条件
            xi_key = rng.choice(xi_keys)
            item = mismatch_cache[xi_key]

            C_BS = item["C_BS"]
            C_UE = item["C_UE"]

            csi_snr_db = sample_csi_snr_db(rng, train_cfg)

            H0_ri, H_DL_true_ri = make_predcalinet_batch(
                H_UL_clean_batch,
                C_BS,
                C_UE,
                csi_snr_db,
            )

            with tf.GradientTape() as tape:
                loss, logs, _ = total_predcalinet_loss(
                    model=model,
                    H0_ri=tf.constant(H0_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    loss_type=train_cfg.get("loss_type", "nmse"),
                    training=True,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)

            train_loss_sum += float(logs["loss"].numpy()) * bsz
            train_mse_sum += float(logs["mse"].numpy()) * bsz
            train_nmse_sum += float(logs["nmse"].numpy()) * bsz
            train_count += bsz

        train_loss = train_loss_sum / train_count
        train_mse = train_mse_sum / train_count
        train_nmse = train_nmse_sum / train_count

        val_logs = evaluate_predcalinet_val_loss(
            model=model,
            val_data=val_data,
            mismatch_cache=mismatch_cache,
            batch_size=batch_size,
            train_cfg=train_cfg,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:04d}/{epochs}] | "
            f"Train Loss={train_loss:.6f}, "
            f"Train MSE={train_mse:.6f}, "
            f"Train NMSE={train_nmse:.6f} | "
            f"Val Loss={val_logs['loss']:.6f}, "
            f"Val MSE={val_logs['mse']:.6f}, "
            f"Val NMSE={val_logs['nmse']:.6f} | "
            f"Time={epoch_time:.2f}s"
        )

        if val_logs["nmse"] < best_val_nmse - train_cfg["min_delta"]:
            best_val_nmse = val_logs["nmse"]
            patience_counter = 0
            model.save_weights(save_path)
            print(f"Best DeepRCCNN weights saved to: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["patience"]:
            print("Early stopping triggered.")
            break

        gc.collect()

    total_time = (time.time() - start_time) / 60.0

    print("\nDeepRCCNN training finished.")
    print(f"Total time: {total_time:.2f} min")
    print(f"Best val NMSE: {best_val_nmse:.6f}")
    print(f"Best weights: {save_path}")


def train_deeprc_cnn(args):
    cfg = DEEPRC_CNN_CONFIG
    train_cfg = DEEPRC_CNN_TRAIN_CONFIG

    print("\n========================================")
    print("Training baseline: DeepRCCNN calibration network")
    print(f"Weights path: {cfg['weights_path']}")
    print(f"Base channels: {cfg.get('base_channels', 64)}")
    print(f"Loss type: {train_cfg.get('loss_type', 'nmse')}")
    print(f"Optimizer: {train_cfg.get('optimizer', 'adam')}")
    print("========================================\n")

    print("Loading dataset...")

    train_data, val_data, _, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]

    print("Train:", train_data.shape)
    print("Val:", val_data.shape)
    print(f"Nt={Nt}, Nr={Nr}")

    rng = np.random.default_rng(SEED)

    mismatch_cache = build_mismatch_cache(Nt=Nt, Nr=Nr)
    xi_keys = list(mismatch_cache.keys())

    model = DeepRCCNN(
        Nr=Nr,
        Nt=Nt,
        base_channels=DEEPRC_CNN_CONFIG.get("base_channels", 64),
    )

    # Build model
    dummy = tf.zeros([1, Nr, Nt, 2], dtype=tf.float32)
    _ = model(dummy, training=False)

    optimizer = make_predcalinet_optimizer(train_cfg)

    save_path = cfg["weights_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]

    best_val_nmse = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        perm = rng.permutation(len(train_data))

        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_nmse_sum = 0.0
        train_count = 0

        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            idx = perm[start:end]

            H_UL_clean_batch = train_data[idx]
            bsz = H_UL_clean_batch.shape[0]

            # 和 Calinet / SA-Mult 一样，从固定 mismatch grid 中采样训练条件
            xi_key = rng.choice(xi_keys)
            item = mismatch_cache[xi_key]

            C_BS = item["C_BS"]
            C_UE = item["C_UE"]

            csi_snr_db = sample_csi_snr_db(rng, train_cfg)

            H0_ri, H_DL_true_ri = make_predcalinet_batch(
                H_UL_clean_batch,
                C_BS,
                C_UE,
                csi_snr_db,
            )

            with tf.GradientTape() as tape:
                loss, logs, _ = total_deeprc_cnn_loss(
                    model=model,
                    H0_ri=tf.constant(H0_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    loss_type=train_cfg.get("loss_type", "nmse"),
                    training=True,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)

            train_loss_sum += float(logs["loss"].numpy()) * bsz
            train_mse_sum += float(logs["mse"].numpy()) * bsz
            train_nmse_sum += float(logs["nmse"].numpy()) * bsz
            train_count += bsz

        train_loss = train_loss_sum / train_count
        train_mse = train_mse_sum / train_count
        train_nmse = train_nmse_sum / train_count

        val_logs = evaluate_predcalinet_val_loss(
            model=model,
            val_data=val_data,
            mismatch_cache=mismatch_cache,
            batch_size=batch_size,
            train_cfg=train_cfg,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:04d}/{epochs}] | "
            f"Train Loss={train_loss:.6f}, "
            f"Train MSE={train_mse:.6f}, "
            f"Train NMSE={train_nmse:.6f} | "
            f"Val Loss={val_logs['loss']:.6f}, "
            f"Val MSE={val_logs['mse']:.6f}, "
            f"Val NMSE={val_logs['nmse']:.6f} | "
            f"Time={epoch_time:.2f}s"
        )

        if val_logs["nmse"] < best_val_nmse - train_cfg["min_delta"]:
            best_val_nmse = val_logs["nmse"]
            patience_counter = 0
            model.save_weights(save_path)
            print(f"Best PredCaliNet-static weights saved to: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["patience"]:
            print("Early stopping triggered.")
            break

        gc.collect()

    total_time = (time.time() - start_time) / 60.0

    print("\nPredCaliNet-static training finished.")
    print(f"Total time: {total_time:.2f} min")
    print(f"Best val NMSE: {best_val_nmse:.6f}")
    print(f"Best weights: {save_path}")



def evaluate_predcalinet_val_loss(
    model,
    val_data,
    mismatch_cache,
    batch_size,
    train_cfg,
):
    val_loss_sum = 0.0
    val_mse_sum = 0.0
    val_nmse_sum = 0.0
    val_count = 0

    val_snr_list = train_cfg.get(
        "val_snr_list",
        [0.0, 10.0, 20.0, 30.0, 40.0],
    )

    for xi_key, item in mismatch_cache.items():
        C_BS = item["C_BS"]
        C_UE = item["C_UE"]

        for csi_snr_db in val_snr_list:
            for start in range(0, len(val_data), batch_size):
                end = min(start + batch_size, len(val_data))
                H_UL_clean_batch = val_data[start:end]
                bsz = H_UL_clean_batch.shape[0]

                H0_ri, H_DL_true_ri = make_predcalinet_batch(
                    H_UL_clean_batch,
                    C_BS,
                    C_UE,
                    csi_snr_db,
                )

                loss, logs, _ = total_predcalinet_loss(
                    model=model,
                    H0_ri=tf.constant(H0_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    loss_type=train_cfg.get("loss_type", "nmse"),
                    training=False,
                )

                val_loss_sum += float(logs["loss"].numpy()) * bsz
                val_mse_sum += float(logs["mse"].numpy()) * bsz
                val_nmse_sum += float(logs["nmse"].numpy()) * bsz
                val_count += bsz

    return {
        "loss": val_loss_sum / val_count,
        "mse": val_mse_sum / val_count,
        "nmse": val_nmse_sum / val_count,
    }




def build_mismatch_cache(Nt, Nr):
    """
    Build fixed mismatch realizations for all training mismatch scenarios.

    Each xi = (std_amp_db, std_phase_deg) corresponds to one fixed
    C_BS and C_UE realization. This is different from online random
    mismatch generation.
    """

    mismatch_cache = {}

    for std_amp_db, std_phase_deg in MISMATCH_GRID:
        xi_key = make_xi_key(std_amp_db, std_phase_deg)

        mismatch_path = os.path.join(
            PATH_CONFIG["results_dir"],
            f"fixed_mismatch_{xi_key}.npz",
        )

        C_BS, C_UE = generate_fixed_mismatch(
            Nt=Nt,
            Nr=Nr,
            std_amp_db=std_amp_db,
            std_phase_deg=std_phase_deg,
            power_neutral=POWER_NEUTRAL,
            seed=SEED,
            save_path=mismatch_path,
            force_regenerate=False,
        )

        mismatch_cache[xi_key] = {
            "std_amp_db": float(std_amp_db),
            "std_phase_deg": float(std_phase_deg),
            "C_BS": C_BS,
            "C_UE": C_UE,
        }

    print("\nLoaded fixed mismatch cache:")
    for xi_key in mismatch_cache:
        print(f"  {xi_key}")

    return mismatch_cache


def load_s_xi_table(cfg):
    table_path = cfg.get(
        "s_xi_table_path",
        "./results_e2e/s_xi_table.json",
    )

    if not os.path.exists(table_path):
        raise FileNotFoundError(
            f"s_xi table not found: {table_path}. "
            f"Please run compute_s_xi_table.py first."
        )

    with open(table_path, "r") as f:
        table = json.load(f)

    if not os.path.exists(table_path):
        raise ValueError(
            f"Empty s_xi table at path: {table_path}. "
            f"Please run compute_s_xi_table.py first."
        )
    print("Available s_xi keys:")
    for k in table.keys():
        print(f"  {k}")

    return table
def get_s_xi(
    table,
    std_amp_db,
    std_phase_deg,
    csi_snr_db=None,
    cfg=None,
):
    xi_key = make_xi_key(std_amp_db, std_phase_deg)

    if cfg is None:
        use_key = "s_xi_mean"
    else:
        use_key = cfg.get("s_xi_key", "s_xi_mean")

    if xi_key not in table:
        raise KeyError(
            f"{xi_key} not found in s_xi_table. "
            f"Available keys: {list(table.keys())}"
        )

    item = table[xi_key]

    if csi_snr_db is not None and "per_snr" in item:
        per_snr = item["per_snr"]

        snr_values = sorted([float(k) for k in per_snr.keys()])
        s_values = []

        for snr in snr_values:
            snr_key = f"{snr:.1f}"
            if snr_key not in per_snr:
                snr_key = str(snr)

            if use_key not in per_snr[snr_key]:
                raise KeyError(
                    f"{use_key} not found in per_snr[{snr_key}] "
                    f"for {xi_key}. Available keys: "
                    f"{list(per_snr[snr_key].keys())}"
                )

            s_values.append(float(per_snr[snr_key][use_key]))

        return float(
            np.interp(
                float(csi_snr_db),
                np.asarray(snr_values, dtype=np.float32),
                np.asarray(s_values, dtype=np.float32),
                left=s_values[0],
                right=s_values[-1],
            )
        )

    if use_key not in item:
        raise KeyError(
            f"{use_key} not found in s_xi table item {xi_key}. "
            f"Available keys: {list(item.keys())}"
        )

    return float(item[use_key])


def sample_csi_snr_db(rng, train_cfg):
    return float(
        rng.uniform(
            train_cfg["train_csi_snr_min_db"],
            train_cfg["train_csi_snr_max_db"],
        )
    )


def make_batch(
    H_UL_clean_batch,
    C_BS,
    C_UE,
    csi_snr_db,
):
    """
    H_UL_clean_batch: [B, Nt, Nr] complex

    Return:
        H_UL_est_ri:  [B, Nt, Nr, 2]
        H_DL_true_ri: [B, Nr, Nt, 2]
    """

    H_DL_true = build_downlink_from_uplink(
        H_UL_clean_batch,
        C_BS,
        C_UE,
    )

    H_UL_est = add_awgn_to_uplink(
        H_UL_clean_batch,
        csi_snr_db,
    )

    H_UL_est_ri = complex_to_ri_np(H_UL_est)
    H_DL_true_ri = complex_to_ri_np(H_DL_true)

    return H_UL_est_ri, H_DL_true_ri


def make_calinet_batch(
    H_UL_clean_batch,
    C_BS,
    C_UE,
    csi_snr_db,
):
    """
    Calinet baseline batch.

    H_UL_clean_batch: [B, Nt, Nr] complex

    Return:
        H0_ri:        [B, Nr, Nt, 2]
        H_DL_true_ri: [B, Nr, Nt, 2]
    """

    H_DL_true = build_downlink_from_uplink(
        H_UL_clean_batch,
        C_BS,
        C_UE,
    )

    H_UL_est = add_awgn_to_uplink(
        H_UL_clean_batch,
        csi_snr_db,
    )

    # Calinet-DNN 输入 naive reciprocity channel:
    # H0 = H_UL_est^T
    H0 = np.transpose(H_UL_est, (0, 2, 1)).astype(np.complex64)

    H0_ri = calinet_complex_to_ri_np(H0)
    H_DL_true_ri = calinet_complex_to_ri_np(H_DL_true)

    return H0_ri, H_DL_true_ri



def get_log_value(logs, old_key, new_key):
    if old_key in logs:
        return logs[old_key]

    if new_key in logs:
        return logs[new_key]

    raise KeyError(
        f"Cannot find either '{old_key}' or '{new_key}' in logs. "
        f"Available keys: {list(logs.keys())}"
    )


# ============================================================
# Validation
# ============================================================

def evaluate_val_loss(
    model,
    val_data,
    mismatch_cache,
    s_xi_table,
    batch_size,
    cfg,
    train_cfg,

):
    val_loss_sum = 0.0
    val_nmse_sum = 0.0
    val_res_sum = 0.0
    val_gram_sum = 0.0
    val_count = 0

    val_snr_list = train_cfg.get(
        "val_snr_list",
        [0.0, 10.0, 20.0, 26.0],
    )

    for xi_key, item in mismatch_cache.items():
        C_BS = item["C_BS"]
        C_UE = item["C_UE"]
        std_amp_db = item["std_amp_db"]
        std_phase_deg = item["std_phase_deg"]

        for csi_snr_db in val_snr_list:
            s_xi = get_s_xi(
                s_xi_table,
                std_amp_db,
                std_phase_deg,
                csi_snr_db=csi_snr_db,
                cfg=cfg,
            )

            for start in range(0, len(val_data), batch_size):
                end = min(start + batch_size, len(val_data))
                H_UL_clean_batch = val_data[start:end]
                bsz = H_UL_clean_batch.shape[0]

                H_UL_est_ri, H_DL_true_ri = make_batch(
                    H_UL_clean_batch,
                    C_BS,
                    C_UE,
                    csi_snr_db,
                )

                csi_snr_batch = (
                    np.ones([bsz], dtype=np.float32)
                    * float(csi_snr_db)
                )
                s_xi_batch = (
                    np.ones([bsz], dtype=np.float32)
                    * float(s_xi)
                )
                std_amp_db_batch = (
                    np.ones([bsz], dtype=np.float32)
                    * float(std_amp_db)
                )
                std_phase_deg_batch = (
                    np.ones([bsz], dtype=np.float32)
                    * float(std_phase_deg)
                )

                loss, logs = total_sa_mult_cal_loss(
                    model=model,
                    H_UL_est_ri=tf.constant(H_UL_est_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    csi_snr_db=tf.constant(csi_snr_batch, dtype=tf.float32),
                    s_xi=tf.constant(s_xi_batch, dtype=tf.float32),
                    std_amp_db=tf.constant(std_amp_db_batch, dtype=tf.float32),
                    std_phase_deg=tf.constant(std_phase_deg_batch, dtype=tf.float32),
                    lambda_res=train_cfg.get("lambda_res", 0.0),
                    lambda_gram=train_cfg.get("lambda_gram", 0.0),
                    training=False,
                )

                val_loss_sum += float(logs["loss"].numpy()) * bsz
                val_nmse_sum += float(
                    get_log_value(logs, "loss_nmse", "nmse").numpy()
                ) * bsz
                val_res_sum += float(
                    get_log_value(logs, "loss_res", "res").numpy()
                ) * bsz
                val_gram_sum += float(
                    get_log_value(logs, "loss_gram", "gram").numpy()
                ) * bsz

                val_count += bsz

    return {
        "loss": val_loss_sum / val_count,
        "nmse": val_nmse_sum / val_count,
        "res": val_res_sum / val_count,
        "gram": val_gram_sum / val_count,
    }



def evaluate_calinet_val_loss(
    model,
    val_data,
    mismatch_cache,
    batch_size,
    train_cfg,
):
    val_loss_sum = 0.0
    val_mse_sum = 0.0
    val_nmse_sum = 0.0
    val_count = 0

    val_snr_list = train_cfg.get(
        "val_snr_list",
        [0.0, 10.0, 20.0, 30.0, 40.0],
    )

    for xi_key, item in mismatch_cache.items():
        C_BS = item["C_BS"]
        C_UE = item["C_UE"]

        for csi_snr_db in val_snr_list:
            for start in range(0, len(val_data), batch_size):
                end = min(start + batch_size, len(val_data))
                H_UL_clean_batch = val_data[start:end]
                bsz = H_UL_clean_batch.shape[0]

                H0_ri, H_DL_true_ri = make_calinet_batch(
                    H_UL_clean_batch,
                    C_BS,
                    C_UE,
                    csi_snr_db,
                )

                loss, logs = total_calinet_loss(
                    model=model,
                    H0_ri=tf.constant(H0_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    loss_type=train_cfg.get("loss_type", "nmse"),
                    training=False,
                )

                val_loss_sum += float(logs["loss"].numpy()) * bsz
                val_mse_sum += float(logs["mse"].numpy()) * bsz
                val_nmse_sum += float(logs["nmse"].numpy()) * bsz
                val_count += bsz

    return {
        "loss": val_loss_sum / val_count,
        "mse": val_mse_sum / val_count,
        "nmse": val_nmse_sum / val_count,
    }

# ============================================================
# Training
# ============================================================

def train_sa_mult(args):
    cfg = SA_MULT_CONFIGS[args.variant]
    train_cfg = SA_MULT_TRAIN_CONFIGS[args.variant]

    print("\n========================================")
    print(f"Training variant: {args.variant}")
    print(f"Scheme name: {cfg['scheme_name']}")
    print(f"Weights path: {cfg['weights_path']}")
    print(f"eta_add: {cfg.get('eta_add', 0.0)}")
    print(f"lambda_res: {train_cfg.get('lambda_res', 0.0)}")
    print(f"lambda_gram: {train_cfg.get('lambda_gram', 0.0)}")
    print(f"condition_mode: {cfg.get('condition_mode', 'full')}")
    print("========================================\n")


    print("Loading dataset...")

    train_data, val_data, _, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]

    print("Train:", train_data.shape)
    print("Val:", val_data.shape)
    print(f"Nt={Nt}, Nr={Nr}")

    rng = np.random.default_rng(SEED)

    mismatch_cache = build_mismatch_cache(Nt=Nt, Nr=Nr)
    xi_keys = list(mismatch_cache.keys())

    s_xi_table = load_s_xi_table(cfg)

    model = SAMultCalNet(
        Nt=Nt,
        Nr=Nr,
        feature_dim=cfg.get("feature_dim", 64),
        num_blocks=cfg.get("num_blocks", 3),
        cond_dim=cfg.get("cond_dim", 64),
        snr_min_db=cfg.get("snr_min_db", 0.0),
        snr_max_db=cfg.get("snr_max_db", 26.0),
        amp_min_db=cfg.get("amp_min_db", 2.0),
        amp_max_db=cfg.get("amp_max_db", 4.0),
        phase_min_deg=cfg.get("phase_min_deg", 20.0),
        phase_max_deg=cfg.get("phase_max_deg", 90.0),
        amp_clip=cfg.get("amp_clip", 0.5),
        phase_clip=cfg.get("phase_clip", 1.0),

        # 关键：full 是 0.02，nores 是 0.0
        eta_add=cfg.get("eta_add", 0.0),

        use_shrinkage=cfg.get("use_shrinkage", True),
        condition_mode=cfg.get("condition_mode", "full"),
    )

    # Build model
    dummy = tf.zeros([1, Nt, Nr, 2], dtype=tf.float32)
    _ = model(
        dummy,
        csi_snr_db=tf.constant([10.0], dtype=tf.float32),
        s_xi=tf.constant(
            [float(cfg.get("s_xi_default", 0.4))],
            dtype=tf.float32,
        ),
        std_amp_db=tf.constant([4.0], dtype=tf.float32),
        std_phase_deg=tf.constant([60.0], dtype=tf.float32),
        training=False,
        return_aux=False,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_cfg["learning_rate"]
    )

    save_path = cfg["weights_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]

    best_val_nmse = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        perm = rng.permutation(len(train_data))

        train_loss_sum = 0.0
        train_nmse_sum = 0.0
        train_res_sum = 0.0
        train_gram_sum = 0.0
        train_count = 0

        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            idx = perm[start:end]

            H_UL_clean_batch = train_data[idx]
            bsz = H_UL_clean_batch.shape[0]

            # Fixed-grid mismatch scenario
            xi_key = rng.choice(xi_keys)
            item = mismatch_cache[xi_key]

            C_BS = item["C_BS"]
            C_UE = item["C_UE"]
            std_amp_db = item["std_amp_db"]
            std_phase_deg = item["std_phase_deg"]

            csi_snr_db = sample_csi_snr_db(rng, train_cfg)

            s_xi = get_s_xi(
                s_xi_table,
                std_amp_db,
                std_phase_deg,
                csi_snr_db=csi_snr_db,
                cfg=cfg,
            )

            H_UL_est_ri, H_DL_true_ri = make_batch(
                H_UL_clean_batch,
                C_BS,
                C_UE,
                csi_snr_db,
            )

            csi_snr_batch = (
                np.ones([bsz], dtype=np.float32)
                * float(csi_snr_db)
            )
            s_xi_batch = (
                np.ones([bsz], dtype=np.float32)
                * float(s_xi)
            )
            std_amp_db_batch = (
                np.ones([bsz], dtype=np.float32)
                * float(std_amp_db)
            )
            std_phase_deg_batch = (
                np.ones([bsz], dtype=np.float32)
                * float(std_phase_deg)
            )

            with tf.GradientTape() as tape:
                loss, logs = total_sa_mult_cal_loss(
                    model=model,
                    H_UL_est_ri=tf.constant(H_UL_est_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    csi_snr_db=tf.constant(csi_snr_batch, dtype=tf.float32),
                    s_xi=tf.constant(s_xi_batch, dtype=tf.float32),
                    std_amp_db=tf.constant(std_amp_db_batch, dtype=tf.float32),
                    std_phase_deg=tf.constant(std_phase_deg_batch, dtype=tf.float32),
                    lambda_res=train_cfg.get("lambda_res", 0.0),
                    lambda_gram=train_cfg.get("lambda_gram", 0.0),
                    training=True,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)

            train_loss_sum += float(logs["loss"].numpy()) * bsz
            train_nmse_sum += float(
                get_log_value(logs, "loss_nmse", "nmse").numpy()
            ) * bsz
            train_res_sum += float(
                get_log_value(logs, "loss_res", "res").numpy()
            ) * bsz
            train_gram_sum += float(
                get_log_value(logs, "loss_gram", "gram").numpy()
            ) * bsz

            train_count += bsz

        train_loss = train_loss_sum / train_count
        train_nmse = train_nmse_sum / train_count
        train_res = train_res_sum / train_count
        train_gram = train_gram_sum / train_count

        val_logs = evaluate_val_loss(
            model=model,
            val_data=val_data,
            mismatch_cache=mismatch_cache,
            s_xi_table=s_xi_table,
            batch_size=batch_size,
            cfg=cfg,
            train_cfg=train_cfg,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:04d}/{epochs}] | "
            f"Train Loss={train_loss:.6f}, "
            f"Train NMSE={train_nmse:.6f}, "
            f"Train Res={train_res:.6f}, "
            f"Train Gram={train_gram:.6f} | "
            f"Val Loss={val_logs['loss']:.6f}, "
            f"Val NMSE={val_logs['nmse']:.6f}, "
            f"Val Res={val_logs['res']:.6f}, "
            f"Val Gram={val_logs['gram']:.6f} | "
            f"Time={epoch_time:.2f}s"
        )

        if val_logs["nmse"] < best_val_nmse - train_cfg["min_delta"]:
            best_val_nmse = val_logs["nmse"]
            patience_counter = 0
            model.save_weights(save_path)
            print(f"Best {cfg['scheme_name']} weights saved to: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["patience"]:
            print("Early stopping triggered.")
            break

        gc.collect()

    total_time = (time.time() - start_time) / 60.0

    print("\nTraining finished.")
    print(f"Total time: {total_time:.2f} min")
    print(f"Best val NMSE: {best_val_nmse:.6f}")
    print(f"Best weights: {save_path}")


def make_calinet_optimizer(train_cfg):
    name = str(train_cfg.get("optimizer", "adagrad")).lower()
    lr = float(train_cfg["learning_rate"])

    if name == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=lr)

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)

    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)

    raise ValueError(f"Unknown Calinet optimizer: {name}")


def train_calinet(args):
    cfg = CALINET_CONFIG
    train_cfg = CALINET_TRAIN_CONFIG

    print("\n========================================")
    print("Training baseline: Calinet-DNN")
    print(f"Weights path: {cfg['weights_path']}")
    print(f"Hidden dims: {cfg.get('hidden_dims', [512, 512, 512])}")
    print(f"Activation: {cfg.get('activation', 'tanh')}")
    print(f"Loss type: {train_cfg.get('loss_type', 'nmse')}")
    print(f"Optimizer: {train_cfg.get('optimizer', 'adagrad')}")
    print("========================================\n")

    print("Loading dataset...")

    train_data, val_data, _, info = load_and_split_complex_data(
        data_path=DATA_CONFIG["data_path"],
        train_ratio=DATA_CONFIG["train_ratio"],
        val_ratio=DATA_CONFIG["val_ratio"],
        test_ratio=DATA_CONFIG["test_ratio"],
        random_state=DATA_CONFIG["random_state"],
    )

    Nt = info["Nt"]
    Nr = info["Nr"]

    print("Train:", train_data.shape)
    print("Val:", val_data.shape)
    print(f"Nt={Nt}, Nr={Nr}")

    rng = np.random.default_rng(SEED)

    mismatch_cache = build_mismatch_cache(Nt=Nt, Nr=Nr)
    xi_keys = list(mismatch_cache.keys())

    model = CalinetDNN(
        Nr=Nr,
        Nt=Nt,
        hidden_dims=cfg.get("hidden_dims", [512, 512, 512]),
        activation=cfg.get("activation", "tanh"),
        output_activation=cfg.get("output_activation", None),
    )

    # Build model
    dummy = tf.zeros([1, Nr, Nt, 2], dtype=tf.float32)
    _ = model(dummy, training=False)

    optimizer = make_calinet_optimizer(train_cfg)

    save_path = cfg["weights_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]

    best_val_nmse = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        perm = rng.permutation(len(train_data))

        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_nmse_sum = 0.0
        train_count = 0

        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            idx = perm[start:end]

            H_UL_clean_batch = train_data[idx]
            bsz = H_UL_clean_batch.shape[0]

            # 和 SA-Mult 一样，从相同 mismatch grid 中采样训练条件
            xi_key = rng.choice(xi_keys)
            item = mismatch_cache[xi_key]

            C_BS = item["C_BS"]
            C_UE = item["C_UE"]

            csi_snr_db = sample_csi_snr_db(rng, train_cfg)

            H0_ri, H_DL_true_ri = make_calinet_batch(
                H_UL_clean_batch,
                C_BS,
                C_UE,
                csi_snr_db,
            )

            with tf.GradientTape() as tape:
                loss, logs = total_calinet_loss(
                    model=model,
                    H0_ri=tf.constant(H0_ri, dtype=tf.float32),
                    H_DL_true_ri=tf.constant(H_DL_true_ri, dtype=tf.float32),
                    loss_type=train_cfg.get("loss_type", "nmse"),
                    training=True,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)

            train_loss_sum += float(logs["loss"].numpy()) * bsz
            train_mse_sum += float(logs["mse"].numpy()) * bsz
            train_nmse_sum += float(logs["nmse"].numpy()) * bsz
            train_count += bsz

        train_loss = train_loss_sum / train_count
        train_mse = train_mse_sum / train_count
        train_nmse = train_nmse_sum / train_count

        val_logs = evaluate_calinet_val_loss(
            model=model,
            val_data=val_data,
            mismatch_cache=mismatch_cache,
            batch_size=batch_size,
            train_cfg=train_cfg,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:04d}/{epochs}] | "
            f"Train Loss={train_loss:.6f}, "
            f"Train MSE={train_mse:.6f}, "
            f"Train NMSE={train_nmse:.6f} | "
            f"Val Loss={val_logs['loss']:.6f}, "
            f"Val MSE={val_logs['mse']:.6f}, "
            f"Val NMSE={val_logs['nmse']:.6f} | "
            f"Time={epoch_time:.2f}s"
        )

        if val_logs["nmse"] < best_val_nmse - train_cfg["min_delta"]:
            best_val_nmse = val_logs["nmse"]
            patience_counter = 0
            model.save_weights(save_path)
            print(f"Best Calinet-DNN weights saved to: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["patience"]:
            print("Early stopping triggered.")
            break

        gc.collect()

    total_time = (time.time() - start_time) / 60.0

    print("\nCalinet-DNN training finished.")
    print(f"Total time: {total_time:.2f} min")
    print(f"Best val NMSE: {best_val_nmse:.6f}")
    print(f"Best weights: {save_path}")


def main():
    args = parse_args()

    if args.method == "sa_mult":
        train_sa_mult(args)

    elif args.method == "calinet":
        train_calinet(args)
    elif args.method == "predcalinet":
        train_predcalinet(args)
    elif args.method == "deeprc_cnn":
        train_deeprc_cnn(args)

    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()