import os
import time
import torch
import numpy as np

# Import core architecture and utility functions from main.py
from main import EKFPredRNN, load_and_prepare_data, calculate_metrics, visualize_predictions

def run_quick_test():
    print("="*60)
    print("DKF-PredRNN++: Quick Inference Test Pipeline")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1] Target Device: {device}")
    
    # 1. Check and generate micro-sample dataset (sample_test_data.npy)
    sample_path = "sample_test_data.npy"
    if not os.path.exists(sample_path):
        print(f"[2] Sample data not found. Extracting from NetCDF (one-time process)...")
        # Path to the raw dataset (adjust this path if running locally)
        data_file = './data/sst_final1.nc' 
        
        try:
            (train_X, train_y), (val_X, val_y), (test_X, test_y), ocean_mask, data_dict = load_and_prepare_data(data_file)
            sample_dict = {
                'X': test_X[:2],  # Extract 2 sample sequences
                'y': test_y[:2],
                'ocean_mask': ocean_mask,
                'data_min': float(data_dict['metadata']['normalization_min']),
                'data_max': float(data_dict['metadata']['normalization_max'])
            }
            np.save(sample_path, sample_dict)
            print(f"    -> Sample extracted and saved to: {sample_path}")
        except Exception as e:
            print(f"    -> ERROR: Failed to process .nc file. Reason: {e}")
            return
            
    # 2. Load the micro-sample data
    print(f"[2] Loading micro-sample dataset: {sample_path}...")
    sample_data = np.load(sample_path, allow_pickle=True).item()
    test_X = torch.FloatTensor(sample_data['X']).to(device)
    test_y = sample_data['y']
    ocean_mask = sample_data['ocean_mask']
    data_min = sample_data['data_min']
    data_max = sample_data['data_max']
    
    # 3. Initialize model architecture
    print("[3] Initializing DKF-PredRNN++ architecture...")
    model = EKFPredRNN(
        hidden_channels=32,
        state_dim=128,  
        num_layers=4,
        output_days=7
    ).to(device)
    
    # 4. Load pre-trained weights
    # Note: Assume the .pth file is in the root directory for GitHub users
    weights_path = "./best_ekf_predrnn_7day_model.pth"
    if os.path.exists(weights_path):
        print(f"[4] Loading pre-trained weights from: {weights_path}...")
        # Use strict=False and weights_only=True to handle legacy parameters and warnings
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=False)
    else:
        print(f"[4] FAILED: Pre-trained weights not found at {weights_path}.")
        return
    

    # 5. Forward Inference
    print("[5] Running 7-day lead-time forecasting...")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        # Run inference without teacher forcing
        predictions = model(test_X, teacher_forcing_ratio=0.0)
    t1 = time.time()
    
    preds_np = predictions.cpu().numpy()
    
    print(f"    -> Input Dimension:  {test_X.shape}")
    print(f"    -> Output Dimension: {preds_np.shape}")
    print(f"    -> Inference Time:   {t1-t0:.3f} seconds")
    
    # 6. Metrics & Visualization
    print("[6] Calculating error metrics and generating visualization...")
    metrics = calculate_metrics(preds_np, test_y, ocean_mask, data_min, data_max)
    print(f"    ✅ Sample RMSE: {metrics['average']['rmse_celsius']:.4f}°C")
    print(f"    ✅ Sample R2:   {metrics['average']['r2']:.4f}")
    
    # Save the output visualization in the results folder
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    visualize_predictions(preds_np, test_y, ocean_mask, data_min, data_max, num_samples=1, save_dir=save_dir)
    
    print("="*60)
    print("🎉 QUICK TEST PASSED SUCCESSFULLY!")
    print(f"The forecasting maps have been saved to: {save_dir}")
    print("="*60)

if __name__ == "__main__":
    run_quick_test()