# Leads-Adaptive-Network-for-fECG

## Project Introduction
This project (Leads-Adaptive-Network-for-fECG) focuses on research and implementation related to fetal electrocardiogram (fECG) processing based on a leads-adaptive network. It aims to improve the performance of fetal electrocardiogram signal extraction and analysis through an adaptive network model, providing technical support for fetal health monitoring.

## Environment Requirements
To run this project, it is recommended to meet the following environment requirements:
- Python 3.x (Python 3.7 or higher is recommended)
- Related dependent libraries: Can be installed by executing `pip install -r requirements.txt`. Specific dependencies may include but are not limited to `numpy`, `pandas`, `torch`, `tensorflow`, `scikit-learn`, etc. (Please refer to the actual content of `requirements.txt`).

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JHUNyizheng/Leads-Adaptive-Network-for-fECG.git
   cd Leads-Adaptive-Network-for-fECG
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Place your dataset files in the following directories (consistent with the paths used in `Ecg_main.py`):
     - `./ABD5_CSV/`: For ABD5 data files (CSV format)
     - `./ABD12_CSV/`: For ABD12 data files (CSV format)
     - `./ab0_test/`: For mask data files (referenced in `data_pre` function)
   - Ensure all data files are in CSV format as expected by the preprocessing logic.

4. **Train the Model**
   Run the main training script. You can modify parameters like model name, output size, etc., within the script:
   ```bash
   python Ecg_main.py
   ```
   - Key parameters in `Ecg_main.py` that can be adjusted:
     - `model_name`: Choose from 'Unet', 'BiLSTM', 'BiLSTM_S' (default: 'BiLSTM_S')
     - `out_size`: Output sequence length (default: 1024)
     - Training epochs, batch size, and other hyperparameters (modify directly in the training function)

5. **Test the Model**
   The testing process is integrated into `Ecg_main.py`. After training, the model will automatically evaluate on the test set and save metrics (MAE, RMSE, PCC) and statistics to:
   - CSV files (e.g., `BiLSTM_S_TS_0_...csv`)
   - Statistics text file (e.g., `BiLSTM_S_stats.txt`)

## Notes
- The data preprocessing logic in `Ecg_main.py` relies on utility functions from `ECG_tool.py` (e.g., `Compose_filter`, `sliding_window2`, normalization functions).
- Trained models are saved in the specified `save_path` with names formatted as `{model_name}_TS_{test_sample}_{model_state}.pth`.


## Evaluation Metrics
The project adopts common evaluation metrics for electrocardiogram signal processing, which may include Signal-to-Noise Ratio (SNR), Root Mean Square Error (RMSE), Correlation Coefficient (CC), etc., to measure the performance of the model in extracting fECG signals.

## Citation
If the code or research of this project is helpful to you, please consider citing the relevant research papers (if any).

## Contact
If you have any questions or suggestions, you can contact the author through the Issues of the GitHub repository.
