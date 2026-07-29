import os
import numpy as np
import tensorflow as tf
import gc
import subprocess
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 4  # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)



import tensorflow as tf
from sionna.channel.tr38901 import AntennaArray, RMa, UMi
from sionna.channel import gen_single_sector_topology


# 参数
Nt = NUM_BS_ANT = 8
Nr = NUM_UT_ANT = 4
NUM_SAMPLES = 50000
SAVE_PATH = "./dataset/H_UL_norm_0.0dB_0.0deg1.npy"
carrier_frequency = 2.14e9 

ut_array = AntennaArray(
    num_rows=1,
    num_cols=int(NUM_UT_ANT / 2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency)
bs_array = AntennaArray(
    num_rows=1,
    num_cols=int(NUM_BS_ANT / 2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency)

# 信道模型
channel = UMi(
    carrier_frequency=carrier_frequency,
    o2i_model="low",
    ut_array=ut_array,
    bs_array=bs_array,
    direction="downlink",
    enable_pathloss=False,
    enable_shadow_fading=False,
    always_generate_lsp=False,
)

# ========== 优化后的生成函数（保留原有逻辑） ==========
def generate_H_UL_optimized(num_samples):
    """
    优化GPU显存占用的版本，保留原有生成逻辑
    """
    H_list = []
    
    for i in range(num_samples):
        # 原有逻辑完全保留
        topology = gen_single_sector_topology(
            batch_size=1,
            num_ut=1,
            scenario="rma"  # 你原来用的是"rma"
        )
        channel.set_topology(*topology)

        # 生成信道
        h, tau = channel(1, 1e6)  # sampling_frequency

        # 原有处理逻辑
        h = tf.reduce_mean(h, axis=-2)  # 平均掉 paths，但保留 time
        H = tf.reshape(h[..., 0], [Nt, Nr])  # 取第一个时间点
        
        # 转换为numpy并添加到列表
        H_list.append(H.numpy())
        
        # ========== 关键优化：显式释放GPU内存 ==========
        # 删除GPU张量
        del h, tau, H
        
        # 每100次循环清理一次计算图
        if (i + 1) % 100 == 0:
            tf.keras.backend.clear_session()
            gc.collect()
        
        # 打印进度
        if (i + 1) % 500 == 0:
            print(f"Generated {i+1}/{num_samples}")
            # 可选：打印显存使用
            # print_gpu_memory()
    
    return np.array(H_list)

# 可选：显存监控函数
def print_gpu_memory():
    """监控GPU显存使用（可选）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        memory_used = int(result.stdout.strip())
        print(f"GPU Memory: {memory_used} MB / 24576 MB ({memory_used/24576*100:.1f}%)")
    except:
        pass

# ========== 保留你原有的所有函数 ==========
def generate_rf_mismatch(Nt, Nr, sigma_A_dB, sigma_phi_deg):
    """
    生成功率中性的 Tx/Rx RF mismatch 矩阵

    Nt: Tx天线数
    Nr: Rx天线数
    sigma_A_dB: 幅度误差标准差(dB)
    sigma_phi_deg: 相位误差标准差(度)

    返回：
        A_tx: [Nt, Nt] 对角复数矩阵
        A_rx: [Nr, Nr] 对角复数矩阵
    """
    # 幅度误差（dB -> 线性）
    A_tx_dB = np.random.normal(0, sigma_A_dB, Nt)
    A_rx_dB = np.random.normal(0, sigma_A_dB, Nr)

    A_tx = 10**(A_tx_dB / 20)
    A_rx = 10**(A_rx_dB / 20)

    # ===== 功率中性归一化 =====
    # 保证每个样本内部，不因为幅度误差整体抬高平均功率
    A_tx = A_tx / np.sqrt(np.mean(A_tx**2) + 1e-12)
    A_rx = A_rx / np.sqrt(np.mean(A_rx**2) + 1e-12)

    # 相位误差（度 -> 弧度）
    phi_tx = np.random.normal(0, np.deg2rad(sigma_phi_deg), Nt)
    phi_rx = np.random.normal(0, np.deg2rad(sigma_phi_deg), Nr)

    # 构造对角复数矩阵
    A_tx_mat = np.diag(A_tx * np.exp(1j * phi_tx))
    A_rx_mat = np.diag(A_rx * np.exp(1j * phi_rx))

    return A_tx_mat, A_rx_mat
# utils/rf_utils.py

def generate_fixed_rf_mismatch(Nt, Nr, sigma_A_dB, sigma_phi_deg, 
                                num_samples=1000, save_path=None, random_state=42):
    """
    生成多个固定的RF误差实例
    
    Returns:
        A_tx: [num_samples, Nt, Nt] 
        A_rx: [num_samples, Nr, Nr]
    """
    np.random.seed(random_state)
    
    A_tx_list = []
    A_rx_list = []
    
    for i in range(num_samples):
        # 调用你原有的函数，生成单个
        A_tx_single, A_rx_single = generate_rf_mismatch(Nt, Nr, sigma_A_dB, sigma_phi_deg)
        A_tx_list.append(A_tx_single)
        A_rx_list.append(A_rx_single)
    
    # 堆叠成 batch
    A_tx = np.stack(A_tx_list, axis=0)  # [num_samples, Nt, Nt]
    A_rx = np.stack(A_rx_list, axis=0)  # [num_samples, Nr, Nr]
    
    print(f"Generated: A_tx shape={A_tx.shape}, A_rx shape={A_rx.shape}")
    
    if save_path:
        np.savez(save_path, A_tx=A_tx, A_rx=A_rx)
        print(f"Saved to {save_path}")
    
    return A_tx, A_rx

def generate_rf_error_features(NUM_SAMPLES, Nt, Nr, sigma_A_dB, sigma_phi_deg):
    features = []
    for _ in range(NUM_SAMPLES):
        A_tx, A_rx = generate_rf_mismatch(Nt, Nr, sigma_A_dB, sigma_phi_deg)

        tx_feat = np.stack([A_tx.diagonal().real, A_tx.diagonal().imag], axis=-1)  # (Nt, 2)
        rx_feat = np.stack([A_rx.diagonal().real, A_rx.diagonal().imag], axis=-1)  # (Nr, 2)

        tx_feat_exp = np.repeat(tx_feat[:, np.newaxis, :], Nr, axis=1)  # (Nt, Nr, 2)
        rx_feat_exp = np.repeat(rx_feat[np.newaxis, :, :], Nt, axis=0)  # (Nt, Nr, 2)

        feat = np.concatenate([tx_feat_exp, rx_feat_exp], axis=-1)      # (Nt, Nr, 4)
        features.append(feat)

    return np.array(features, dtype=np.float32)

def normalize_H(H):
    # H shape: (NUM_SAMPLES, Nr, Nt)
    power = np.sum(np.abs(H)**2, axis=(1,2), keepdims=True)
    return H / np.sqrt(power + 1e-12)

# ========== 主程序 ==========
if __name__ == "__main__":
    # print("="*50)
    # print("Starting optimized GPU generation...")
    # print("="*50)
    # # # RF误差
    sigma_A_dB = 0.0
    sigma_phi_deg = 0.0
    
    # 1. 生成数据（优化版）
    print("\n Generating channel data...")
    H_UL_data = generate_H_UL_optimized(NUM_SAMPLES)
    
    # 保存原始数据
    np.save(SAVE_PATH, H_UL_data)
    print(f"Saved raw data to {SAVE_PATH}_{sigma_A_dB}dB_{sigma_phi_deg}deg")
    
    # 2. 归一化处理（保留原有逻辑）
    print("\nNormalizing data...")
    H_UL_data = np.load(SAVE_PATH)
    H_UL_norm = normalize_H(H_UL_data)
    np.save(f"H_UL_norm_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npy", H_UL_norm)
    print("Saved normalized data to H_UL_norm.npy")
    
    # 3. 构建特征（保留原有逻辑）
    print("\nBuilding features...")
    NUM_SAMPLES, Nt, Nr = H_UL_norm.shape
    
    H_real = H_UL_norm.real
    H_imag = H_UL_norm.imag
    H_features = np.stack([H_real, H_imag], axis=-1).astype(np.float32)
    rf_features = generate_rf_error_features(NUM_SAMPLES, Nt, Nr, sigma_A_dB, sigma_phi_deg).astype(np.float32)
    model_input = np.concatenate([H_features, rf_features], axis=-1).astype(np.float32)
    
    print(f"Input shape: {model_input.shape}")
    
    # # 4. 验证（保留原有打印逻辑）
    # print("\n Sample verification:")
    # print("First sample H real:\n", model_input[0, :, :, 0])
    # print("First sample H imag:\n", model_input[0, :, :, 1])
    # print("First sample Rx real/imag:\n", model_input[0, :, :, 2:4])
    # print("First sample Tx real/imag:\n", model_input[0, :, :, 4:6])
    
    # 5. 保存所有文件
    np.save(f"dataset/H_UL_raw_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npy", H_UL_data)
    np.save(f"dataset/H_UL_norm_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npy", H_UL_norm)
    np.save(f"dataset/model_input_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npy", model_input)
    
    print("\n All files saved:")
    print("  - H_UL_raw.npy (原始数据)")
    print("  - H_UL_norm.npy (归一化数据)")
    print("  - model_input.npy (网络输入)")
    
    # 6. 验证功率归一化
    print("\n Power normalization check:")
    print("Before norm total power per sample:", 
          np.sum(np.abs(H_UL_data)**2, axis=(1,2))[:5])
    print("After norm total power per sample:", 
          np.sum(np.abs(H_UL_norm)**2, axis=(1,2))[:5])
    
    print("\n" + "="*50)
    print(" Dataset generation completed successfully!")
    print("="*50)

    # 训练/设计用误差池
    generate_fixed_rf_mismatch(
        Nt, Nr,
        sigma_A_dB=sigma_A_dB,
        sigma_phi_deg=sigma_phi_deg,
        num_samples=NUM_SAMPLES,
        save_path=f"dataset/train_error_pool_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npz",
        random_state=42
    )

    # 测试评估用固定误差
    generate_fixed_rf_mismatch(
        Nt, Nr,
        sigma_A_dB=sigma_A_dB,
        sigma_phi_deg=sigma_phi_deg,
        num_samples=NUM_SAMPLES,
        save_path=f"dataset/fixed_rf_errors_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npz",
        random_state=123
    )

    # ===== 检查误差前后平均信道功率 =====
    print("\nChecking average channel power before/after RF mismatch...")

    # H_UL_norm shape: [N, Nt, Nr]
    # 先转成 [N, Nr, Nt]，和后面矩阵乘法一致
    H_UL_check = np.transpose(H_UL_norm, (0, 2, 1))   # [N, Nr, Nt]

    err_data = np.load(f"dataset/fixed_rf_errors_{sigma_A_dB}dB_{sigma_phi_deg}deg1.npz")
    A_tx = err_data["A_tx"]   # [N, Nt, Nt]
    A_rx = err_data["A_rx"]   # [N, Nr, Nr]

    H_true = A_rx @ H_UL_check @ A_tx

    p_H = np.mean(np.sum(np.abs(H_UL_check)**2, axis=(1, 2)))
    p_Htrue = np.mean(np.sum(np.abs(H_true)**2, axis=(1, 2)))

    print("mean ||H_UL||_F^2   =", p_H)
    print("mean ||H_true||_F^2 =", p_Htrue)
 