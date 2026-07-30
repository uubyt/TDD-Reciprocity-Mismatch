# TDD MIMO Reciprocity-Mismatch Calibration

## 1. File Descriptions

| File | Description |
|---|---|
| `generate_data1.py` | Generates the MIMO channel dataset with Sionna and saves the normalized uplink channel data under the dataset directory. |
| `config_system.py` | Stores all project settings, including dataset paths, mismatch parameters, SNR values, training settings, model paths, and link configuration. |
| `channels.py` | Loads and splits the channel dataset, generates fixed RF mismatch matrices, constructs the effective downlink channel, and adds AWGN to the uplink CSI. |
| `compute_s_xi_table.py` | Computes the residual-scale table `s_xi` for different RF mismatch conditions and CSI SNR values. The result is saved as `results_e2e/s_xi_table.json`. |
| `s_xi_utils.py` | Loads and interpolates `s_xi` values from the saved `s_xi` table. |
| `sa_mult_cal_net.py` | Defines the SA-Mult-Cal network, condition encoder, multiplicative calibration heads, residual branch, and training losses. |
| `sa_mult_calibration.py` | Loads a trained SA-Mult-Cal model and performs downlink channel calibration during link evaluation. |
| `train_sa_mult_cal_net.py` | Main training script for SA-Mult-Cal and the included baseline calibration networks. |
| `calinet_dnn_calibration.py` | Defines the fully connected Calinet-DNN baseline and its loss functions. |
| `predcalinet_static_calibration.py` | Defines the static PredCaliNet baseline using LSTM and fully connected layers. |
| `deeprc_cnn_calibration.py` | Defines the DeepRC-inspired CNN calibration baseline. |
| `deeprc_cnn_runtime.py` | Loads trained DeepRC-CNN weights and performs inference. |
| `precoding.py` | Implements equal-power SVD precoding, water-filling SVD precoding, and precoder normalization. |
| `modulation.py` | Implements random bit generation, QPSK modulation, and hard-decision QPSK demodulation. |
| `receiver.py` | Implements downlink transmission with AWGN and MMSE equalization. |
| `metrics.py` | Computes BER, SER, BLER, EVM, effective spectral efficiency, capacity, channel NMSE, Gram leakage, and post-MMSE SINR. |
| `run_traditional_link.py` | Runs the complete downlink link-level evaluation and saves the results as JSON files. |
| `plot_traditional_results.py` | Loads the saved JSON results and plots capacity, spectral efficiency, BER, BLER, SER, and EVM curves. |
| `residual_refiner_calibration.py` | Provides the MoE plus residual-refiner inference wrapper. This file requires additional MoE and residual-refiner source files and weights. |

## 2. Environment Requirements

### Required Software

- Python 3
- NumPy
- TensorFlow
- Matplotlib

Dataset generation additionally requires:

- Sionna

### Installation

```bash
pip install numpy tensorflow matplotlib sionna
```

The installed Sionna version must support:

```python
from sionna.channel.tr38901 import AntennaArray, UMi
from sionna.channel import gen_single_sector_topology
```

### Required Directories

Create the required directories before running the project:

```bash
mkdir -p dataset results_e2e saved_models_e2e
```

### Dataset Path

Check the dataset path in `config_system.py`. The current path is:

```text
/home/yt/RF2/dataset/H_UL_norm_0.0dB_0.0deg1.npy
```

Update this path when the project is stored in another directory.

### GPU Configuration

The scripts currently select the following GPUs:

| Script | GPU |
|---|---:|
| `generate_data1.py` | 4 |
| `train_sa_mult_cal_net.py` | 1 |
| `run_traditional_link.py` | 5 |

Change `CUDA_VISIBLE_DEVICES` in these scripts according to the available GPUs.

## 3. Run Commands

Run all commands from the project root.

### Step 1: Generate the Channel Dataset

```bash
python generate_data1.py
```

Expected main dataset:

```text
dataset/H_UL_norm_0.0dB_0.0deg1.npy
```

### Step 2: Compute the `s_xi` Table

```bash
python compute_s_xi_table.py
```

Output:

```text
results_e2e/s_xi_table.json
```

### Step 3: Train the SA-Mult-Cal Ablation Models

Without condition and residual:

```bash
python train_sa_mult_cal_net.py --method sa_mult --variant nocond
```

With condition but without residual:

```bash
python train_sa_mult_cal_net.py --method sa_mult --variant nores
```

Full SA-Mult-Cal model:

```bash
python train_sa_mult_cal_net.py --method sa_mult --variant full
```

Optional model without Gram loss:

```bash
python train_sa_mult_cal_net.py --method sa_mult --variant nores_nogram
```

### Step 4: Evaluate the Trained Models

Evaluate the full model:

```bash
python run_traditional_link.py --eval_variant full
```

Evaluate the model without residual:

```bash
python run_traditional_link.py --eval_variant nores
```

Evaluate the model without condition and residual:

```bash
python run_traditional_link.py --eval_variant nocond
```

Evaluate all three main SA-Mult variants:

```bash
python run_traditional_link.py --eval_variant all
```

### Step 5: Plot the Results

Plot the SA-Mult ablation results:

```bash
python plot_traditional_results.py --plot_mode ablation
```

Plot the main comparison:

```bash
python plot_traditional_results.py --plot_mode main
```

Plot all available schemes:

```bash
python plot_traditional_results.py --plot_mode all
```

## 4. Optional Baseline Training

Train Calinet-DNN:

```bash
python train_sa_mult_cal_net.py --method calinet
```

Train PredCaliNet-static:

```bash
python train_sa_mult_cal_net.py --method predcalinet
```

Train DeepRC-CNN:

```bash
python train_sa_mult_cal_net.py --method deeprc_cnn
```

## 5. Notes

1. Run `compute_s_xi_table.py` before training SA-Mult-Cal.
2. Train the requested model before running link evaluation.
3. Do not enable optional baselines unless their runtime wrappers, external source files, and weight files are available.
4. `generate_data1.py` currently creates a UMi channel model but uses an RMa topology setting. Confirm the intended scenario before final experiments.
5. The fixed-mismatch filenames used by `compute_s_xi_table.py` and `train_sa_mult_cal_net.py` are not identical. Use a consistent naming format if both scripts must reuse exactly the same mismatch realization.
