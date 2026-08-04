import os

# ============================================================
# Unified traditional MIMO link configuration
# ============================================================

DATA_CONFIG = {
    "data_path": "/home/yt/RF2/dataset/H_UL_norm_0.0dB_0.0deg1.npy",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_state": 42,
}

PATH_CONFIG = {
    "results_dir": "./results_e2e",
    "save_dir": "./saved_models_e2e",
}

os.makedirs(PATH_CONFIG["results_dir"], exist_ok=True)
os.makedirs(PATH_CONFIG["save_dir"], exist_ok=True)


def build_experiment_tag(mismatch_config):
    return (
        f'{mismatch_config["std_amp_db"]:.1f}dB_'
        f'{mismatch_config["std_phase_deg"]:.1f}deg'
    )


# ============================================================
# Test mismatch and SNR configuration
# ============================================================

TEST_MISMATCH_CONFIG = {
    "std_amp_db": 4.0,
    "std_phase_deg": 60.0,
    "power_neutral": True,
}

TEST_EXPERIMENT_TAG = build_experiment_tag(TEST_MISMATCH_CONFIG)

# dl_snr_dB_list: downlink data-transmission SNR.
# csi_snr_mode:
#   "same_as_dl": CSI estimation SNR equals DL SNR for each point.
#   "fixed": CSI estimation SNR is fixed to fixed_csi_snr_db.
SNR_CONFIG = {
    "dl_snr_dB_list": list(range(0, 27, 2)),
    "csi_snr_mode": "same_as_dl",
    "fixed_csi_snr_db": 10.0,
}

SNR_CONFIG["dl_snr_linear_list"] = [
    10 ** (x / 10.0) for x in SNR_CONFIG["dl_snr_dB_list"]
]


# ============================================================
# Link settings
# ============================================================

LINK_CONFIG = {
    "num_symbols": 1024,
    "bits_per_symbol": 2,  # 4QAM/QPSK
    "precoder_type": "svd_water_power",  "svd_waterfilling"
    "tx_power": 1.0,
    "seed": 123,
    "bler_block_symbols": 8,
    "batch_size": 256,
    "num_streams": 2,
}


# ============================================================
# External packages
# ============================================================

EXTERNAL_PACKAGE_CONFIG = {
    "calib_package_dir": "/home/yt/RF2/calib_hdl_svd_package_cross_all",
}


# ============================================================
# Scheme switches and scheme list
# ============================================================

ENABLE_MOE_CALIB = True
ENABLE_MOE_REFINER_CALIB = True        # H-refiner
ENABLE_MOE_PRECODER_REFINER = False    # W-refiner
ENABLE_MCSGD = True
ENABLE_GENIE_DIRECT_PRECODER = False

# Old additive SA-ResCal baseline. Keep False unless you want to test it.
ENABLE_SA_RESCAL = False

# New SA-Mult-Cal ablations.
ENABLE_SA_MULT_FULL = True
ENABLE_SA_MULT_NORES = True
ENABLE_SA_MULT_NOCOND = True
ENABLE_SA_MULT_NORES_NOGRAM = False

# Black-box DNN / Calinet calibration baseline
ENABLE_NN_CALIB = True


SCHEMES = ["ideal_svd"]

if ENABLE_GENIE_DIRECT_PRECODER:
    SCHEMES.append("genie_direct_precoder")

if ENABLE_NN_CALIB:
    SCHEMES.append("nn_calib_svd")

if ENABLE_MOE_CALIB:
    SCHEMES.append("moe_calib_svd")

if ENABLE_MOE_REFINER_CALIB:
    SCHEMES.append("moe_refiner_calib_svd")

if ENABLE_MOE_PRECODER_REFINER:
    SCHEMES.append("moe_preco_refiner_svd")

if ENABLE_SA_RESCAL:
    SCHEMES.append("sa_rescal_svd")

# Error baseline is always useful.
SCHEMES.append("error_svd")

if ENABLE_MCSGD:
    SCHEMES.append("mc_sgd")

if ENABLE_SA_MULT_NOCOND:
    # Step 1: multiplicative-only, without condition and without residual.
    SCHEMES.append("sa_mult_nores_nocond_svd")

if ENABLE_SA_MULT_NORES:
    # Step 2: multiplicative + condition, without residual.
    SCHEMES.append("sa_mult_nores_svd")

if ENABLE_SA_MULT_FULL:
    # Step 3: multiplicative + condition + small residual.
    SCHEMES.append("sa_mult_full_svd")

if ENABLE_SA_MULT_NORES_NOGRAM:
    # Optional training-loss ablation.
    SCHEMES.append("sa_mult_nores_nogram_svd")


# ============================================================
# MC-SGD baseline configuration
# ============================================================

MCSGD_CONFIG = {
    "num_pool": 2000,
    "num_iters": 30,
    "step_size": 0.03,
    "mc_samples": 100,
    "init": "svd",
    "seed": 123,
}


# ============================================================
# MoE calibration configuration
# ============================================================

MOE_CALIB_CONFIG = {
    "calib_package_dir": "/home/yt/RF2/calib_soft_moe_expert_version",
    "train_script": "train_calib.py",

    # Original MoE weights: used by moe_calib_svd and as the base MoE
    # for residual/refiner models.
    "weights_path": (
        "/home/yt/RF2/saved_models_b/UMi_calib_soft_moe/"
        "best_model_expertgrid_A2.0-4.0dB_P20.0-90.0deg_K12_"
        "calib_hdl_svd_soft_moe_expert.weights.h5"
    ),
}


# ============================================================
# End-to-end MoE H-refiner configuration
# ============================================================

E2E_TRAIN_CONFIG = {
    "pretrained_moe_weights_path": MOE_CALIB_CONFIG["weights_path"],

    "save_refiner_weights_path": os.path.join(
        PATH_CONFIG["save_dir"],
        "best_model_expertgrid_A2.0-4.0dB_P20.0-90.0deg_K12_"
        "moe_residual_refiner_e2e_bce.weights.h5",
    ),

    "epochs": 3000,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "patience": 80,
    "min_delta": 1e-4,

    "num_symbols_train": 256,
    "demap_scale": 5.0,

    # Loss = BCE + lambda_nmse * NMSE + lambda_delta * DeltaPenalty
    "lambda_nmse": 2.0,
    "lambda_delta": 0.1,

    # Focus on the BLER waterfall region.
    "train_dl_snr_min_db": 16.0,
    "train_dl_snr_max_db": 26.0,

    "val_dl_snr_db_list": [18.0, 20.0, 22.0, 24.0, 26.0],
}

RESIDUAL_REFINER_CONFIG = {
    "hidden_dim": 32,
    "num_blocks": 2,
    "delta_scale": 0.1,
}

MOE_REFINER_CALIB_CONFIG = {
    "base_moe_weights_path": MOE_CALIB_CONFIG["weights_path"],
    "refiner_weights_path": E2E_TRAIN_CONFIG["save_refiner_weights_path"],
}


# ============================================================
# End-to-end MoE precoder-refiner configuration
# ============================================================

PRECODER_E2E_TRAIN_CONFIG = {
    "pretrained_moe_weights_path": MOE_CALIB_CONFIG["weights_path"],

    "save_precoder_refiner_weights_path": os.path.join(
        PATH_CONFIG["save_dir"],
        "best_model_expertgrid_A2.0-4.0dB_P20.0-90.0deg_K12_"
        "moe_precoder_refiner_e2e_bce.weights.h5",
    ),

    "epochs": 3000,
    "batch_size": 512,
    "learning_rate": 5e-5,
    "patience": 100,
    "min_delta": 1e-4,

    "num_symbols_train": 256,
    "demap_scale": 5.0,

    # Loss = BCE + lambda_delta_w * DeltaWPenalty
    "lambda_delta_w": 0.001,

    # Weakest-stream post-MMSE SINR regularization
    "lambda_weak_sinr": 0.01,
    "weak_sinr_beta": 5.0,

    # Focus on the BLER waterfall region.
    "train_dl_snr_min_db": 16.0,
    "train_dl_snr_max_db": 26.0,

    "val_dl_snr_db_list": [18.0, 20.0, 22.0, 24.0, 26.0],
}

PRECODER_REFINER_CONFIG = {
    "hidden_dim": 256,
    "num_hidden_layers": 3,
    "delta_scale": 0.05,
}

MOE_PRECODER_REFINER_CONFIG = {
    "base_moe_weights_path": MOE_CALIB_CONFIG["weights_path"],
    "precoder_refiner_weights_path": PRECODER_E2E_TRAIN_CONFIG[
        "save_precoder_refiner_weights_path"
    ],
}


# ============================================================
# Perfect-CSI direct link-aware precoder configuration
# ============================================================

DIRECT_PRECODER_CONFIG = {
    "hidden_dim": 128,
    "num_hidden_layers": 3,
}

GENIE_DIRECT_PRECODER_CONFIG = {
    "weights_path": os.path.join(
        PATH_CONFIG["save_dir"],
        "best_model_genie_direct_precoder_e2e_bce_weak_sinr.weights.h5",
    ),
}


# ============================================================
# SA-Mult-Cal base configuration
# ============================================================

# This is the shared base config for all SA-Mult-Cal variants.
# The variants below override only scheme_name, weights_path,
# condition_mode, eta_add, lambda_res, and lambda_gram.
SA_RESCAL_CONFIG = {
    # Kept for backward compatibility with older code.
    "scheme_name": "sa_mult_base_svd",
    "weights_path": "./saved_models_e2e/sa_mult_base.weights.h5",

    # s_xi table
    "s_xi_table_path": os.path.join(PATH_CONFIG["results_dir"], "s_xi_table.json"),
    "s_xi_key": "s_xi_mean",
    "s_xi_default": 0.4,
    "s_max": 1.5,

    # architecture
    "cond_dim": 64,
    "feature_dim": 64,
    "num_blocks": 3,

    # multiplicative gain range
    "amp_clip": 0.5,
    "phase_clip": 1.0,

    # residual branch
    "eta_add": 0.02,

    # condition normalization ranges
    "snr_min_db": 0.0,
    "snr_max_db": 26.0,
    "amp_min_db": 2.0,
    "amp_max_db": 4.0,
    "phase_min_deg": 20.0,
    "phase_max_deg": 90.0,

    # H0 construction
    "use_shrinkage": True,

    # condition mode:
    #   "full": use [CSI SNR, s_xi, amp std, phase std]
    #   "none": zero out all condition channels
    #   "no_error": keep CSI SNR, remove mismatch statistics
    "condition_mode": "full",
}

SA_RESCAL_TRAIN_CONFIG = {
    "epochs": 2000,
    "batch_size": 512,
    "learning_rate": 1e-4,

    # default loss weights; each variant can override them
    "lambda_res": 0.001,
    "lambda_gram": 0.02,

    "patience": 30,
    "min_delta": 1e-5,

    "train_csi_snr_min_db": 0.0,
    "train_csi_snr_max_db": 26.0,

    "val_snr_list": [0.0, 10.0, 20.0, 26.0],
}


# ============================================================
# SA-Mult-Cal ablation variants
# ============================================================

# Step 1:
#   multiplicative-only
#   H_hat = D_r(H0) H0 D_t(H0)
SA_MULT_NOCOND_CONFIG = {
    **SA_RESCAL_CONFIG,
    "scheme_name": "sa_mult_nores_nocond_svd",
    "weights_path": "./saved_models_e2e/sa_mult_nores_nocond.weights.h5",
    "condition_mode": "none",
    "eta_add": 0.0,
}

SA_MULT_NOCOND_TRAIN_CONFIG = {
    **SA_RESCAL_TRAIN_CONFIG,
    "lambda_res": 0.0,
    "lambda_gram": 0.02,
}

# Step 2:
#   multiplicative + condition
#   H_hat = D_r(H0, z_xi) H0 D_t(H0, z_xi)
SA_MULT_NORES_CONFIG = {
    **SA_RESCAL_CONFIG,
    "scheme_name": "sa_mult_nores_svd",
    "weights_path": "./saved_models_e2e/sa_mult_nores.weights.h5",
    "condition_mode": "full",
    "eta_add": 0.0,
}

SA_MULT_NORES_TRAIN_CONFIG = {
    **SA_RESCAL_TRAIN_CONFIG,
    "lambda_res": 0.0,
    "lambda_gram": 0.02,
}

# Step 3:
#   multiplicative + condition + small residual
#   H_hat = D_r(H0, z_xi) H0 D_t(H0, z_xi) + eta*g*s_xi*Delta_bar
SA_MULT_FULL_CONFIG = {
    **SA_RESCAL_CONFIG,
    "scheme_name": "sa_mult_full_svd",
    "weights_path": "./saved_models_e2e/sa_mult_full.weights.h5",
    "condition_mode": "full",
    "eta_add": 0.02,
}

SA_MULT_FULL_TRAIN_CONFIG = {
    **SA_RESCAL_TRAIN_CONFIG,
    "lambda_res": 0.001,
    "lambda_gram": 0.02,
}

# Optional:
#   training-loss ablation for Gram loss
SA_MULT_NORES_NOGRAM_CONFIG = {
    **SA_MULT_NORES_CONFIG,
    "scheme_name": "sa_mult_nores_nogram_svd",
    "weights_path": "./saved_models_e2e/sa_mult_nores_nogram.weights.h5",
}

SA_MULT_NORES_NOGRAM_TRAIN_CONFIG = {
    **SA_MULT_NORES_TRAIN_CONFIG,
    "lambda_gram": 0.0,
}

# Dictionaries used by train_sa_mult_cal.py.
SA_MULT_CONFIGS = {
    "nocond": SA_MULT_NOCOND_CONFIG,
    "nores": SA_MULT_NORES_CONFIG,
    "full": SA_MULT_FULL_CONFIG,
    "nores_nogram": SA_MULT_NORES_NOGRAM_CONFIG,
}

SA_MULT_TRAIN_CONFIGS = {
    "nocond": SA_MULT_NOCOND_TRAIN_CONFIG,
    "nores": SA_MULT_NORES_TRAIN_CONFIG,
    "full": SA_MULT_FULL_TRAIN_CONFIG,
    "nores_nogram": SA_MULT_NORES_NOGRAM_TRAIN_CONFIG,
}

# Dictionaries can also be used by run_traditional_link.py.
SA_MULT_EVAL_CONFIGS = {
    SA_MULT_NOCOND_CONFIG["scheme_name"]: SA_MULT_NOCOND_CONFIG,
    SA_MULT_NORES_CONFIG["scheme_name"]: SA_MULT_NORES_CONFIG,
    SA_MULT_FULL_CONFIG["scheme_name"]: SA_MULT_FULL_CONFIG,
    SA_MULT_NORES_NOGRAM_CONFIG["scheme_name"]: SA_MULT_NORES_NOGRAM_CONFIG,
}


RESULT_CONFIG = {
    "tag": "sa_mult_ablation",
    "add_timestamp": False,
}


# ============================================================
# LS-based Relative Reciprocity Calibration baseline
# ============================================================
# mode:
#   "bs_only":   H_DL_hat = H_UL_est^T diag(b)
#   "two_sided": H_DL_hat = diag(a) H_UL_est^T diag(b)
LS_RC_CONFIG = {
    "mode": "two_sided",
    "num_iter": 50,
    "num_calib_samples": 5000,
}


# ============================================================
# Calinet-DNN black-box AI calibration baseline
# ============================================================

CALINET_CONFIG = {
    "weights_path": os.path.join(
        PATH_CONFIG["save_dir"],
        "calinet_dnn_calibration.weights.h5",
    ),

    # Calinet paper: input layer + 3 hidden layers + output layer.
    # Here hidden_dims controls the three hidden FC layers.
    "hidden_dims": [512, 512, 512],

    "activation": "tanh",

    # The paper uses tanh activation in hidden layers.
    # For output layer, None is more stable for complex-valued channel regression.
    # Set to "tanh" only if all channel coefficients are strictly normalized to [-1,1].
    "output_activation": None,
}


CALINET_TRAIN_CONFIG = {
    # Paper-like setting: Adagrad, lr=0.01, 256 epochs.
    "optimizer": "adagrad",
    "learning_rate": 1e-2,
    "epochs": 256,

    # Paper uses batch_size=4.
    # Your dataset is larger, so 256 is more practical.
    # If you want strict reproduction, change this to 4.
    "batch_size": 256,

    "patience": 40,
    "min_delta": 1e-5,

    # "mse" is closer to the paper.
    # "nmse" is usually fairer for your current channel-normalized comparison.
    "loss_type": "mse",

    # Calinet paper trains over SNR 0--40 dB.
    "train_csi_snr_min_db": 0.0,
    "train_csi_snr_max_db": 40.0,
    "val_snr_list": [18.0, 20.0, 22.0, 24.0, 26.0],
}


# ============================================================
# PredCaliNet-static baseline config
# ============================================================

PREDCALINET_CONFIG = {
    "weights_path": "./saved_models_e2e/predcalinet_static.weights.h5",
    "hidden_dim": 256,
    "num_lstm_layers": 3,
    "fc_dims": None,
    "activation": "tanh",
    "output_activation": None,
}

PREDCALINET_TRAIN_CONFIG = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "epochs": 256,
    "batch_size": 256,
    "patience": 40,
    "min_delta": 1e-5,
    "loss_type": "nmse",
    "train_csi_snr_min_db": 0.0,
    "train_csi_snr_max_db": 40.0,
    "val_snr_list": [18.0, 20.0, 22.0, 24.0, 26.0],
}


DEEPRC_CNN_CONFIG = {
    "scheme_name": "deeprc_cnn_svd",
    "weights_path": "./saved_models_e2e/deeprc_cnn.weights.h5",
    "base_channels": 64,
}

DEEPRC_CNN_TRAIN_CONFIG = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "epochs": 256,
    "batch_size": 256,
    "patience": 40,
    "min_delta": 1e-5,

    "loss_type": "nmse",

    "train_csi_snr_min_db": 0.0,
    "train_csi_snr_max_db": 40.0,
    "val_snr_list": [18.0, 20.0, 22.0, 24.0, 26.0],
}
