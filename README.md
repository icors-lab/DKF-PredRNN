# DKF-PredRNN++: Dynamic Physics-Guided Learning for SST Forecasting via Online State Correction

## 1. Description
This repository contains the official PyTorch implementation of **DKF-PredRNN++**, as presented in the paper *"Dynamic Physics-Guided Learning for SST Forecasting via Online State Correction"*.

The proposed model addresses the limitations of static physical priors in deep learning by embedding a **Differentiable Kalman Filter (DKF)** directly into a spatiotemporal recurrent network. This framework enables **Online State Correction**, allowing the model to dynamically adjust physical constraint strength and mitigate error accumulation in long-range sea surface temperature (SST) forecasting.

## Model Architecture
<img width="7361" height="3667" alt="图片1" src="https://github.com/user-attachments/assets/540f1494-ec10-4496-8986-dedc817859d6" />


## 2. Repository Structure
The repository is organized as follows:
- `main.py`: The core script containing model architecture, training loops, and detailed evaluation pipelines.
- `quick_test.py`: A lightweight script designed for reviewers to verify reproducibility and model inference.
- `best_ekf_predrnn_7day_model.pth`: Pre-trained model weights for 7-day lead-time forecasting.
- `sample_test_data.npy`: A micro-sample dataset extracted from the test set for quick verification.
- `requirements.txt`: A list of Python dependencies required to run the code.
- `LICENSE`: The MIT License governing this software.

## 3. Setup and Installation
The code has been tested on **Python 3.9** and **PyTorch 2.5.1**. We recommend using a GPU-enabled environment (e.g., NVIDIA A6000 or RTX 3090) for full training, though the quick test can run on a standard CPU.

**To install the dependencies:**
```bash
pip install -r requirements.txt
```

## 4. Quick Test (Mandatory for Reproducibility)
To verify the model without downloading the massive original dataset or performing long training sessions, we provide a quick verification pipeline. This script loads the micro-sample data and the pre-trained weights to perform a 7-day forward inference pass.

**How to run the quick test:**
1. Clone or download this repository.
2. Run the following command in your terminal:
   ```bash
   python quick_test.py
   ```
3. **Expected Outcome**: 
   - The script will finish in a few seconds.
   - It will display the forecasting metrics (RMSE, R2) for the samples.
   - A spatial visualization comparison map (Prediction vs. Ground Truth) will be saved in the `./results/` folder as `dkf_predrnn_viz_sample_1.png`.
   
### Visualization of Results
<img width="4426" height="9286" alt="dkf_predrnn_sample_1" src="https://github.com/user-attachments/assets/0023c301-6040-4604-ba13-327fbd7d9c39" />



## 5. Full Training and Inference
To reproduce the full experiments described in the paper:

### Data Preparation
The full preprocessed SST dataset (Yellow Sea coastal waters, 2000-2020) is available on Zenodo: 
https://doi.org/10.5281/zenodo.18240836

### Run Training
Place the downloaded .nc file in the ./data/ directory and run:
```bash
python main.py --data_file ./data/sst_final1.nc --save_dir ./results --epochs 50
```

### Run Evaluation (Inference Only)
To evaluate the pre-trained model on the full test set:
```bash
python main.py --epochs 0 --save_dir ./ --data_file ./data/sst_final1.nc
```

## 6. Key Features
- **Dynamic Physics-Guidance**: Uses Kalman Gain to adaptively balance data-driven features and physical constraints.
- **Serial State Correction**: Corrects hidden states at each recurrent step to ensure physical consistency.
- **Robustness**: Specifically optimized for extreme oceanic events (e.g., typhoon-induced cooling).

## 7. Authors and Contact
- **Donglin Fan**: dlfan@glut.edu.cn
- **Hongchang He** (Corresponding Author): HHe_glut@126.com
College of Geomatics and Geoinformation, Guilin University of Technology.

## 8. License
This project is licensed under the **MIT License** - see the LICENSE file for details.
