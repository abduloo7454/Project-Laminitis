# Laminitis Risk Analysis and Prediction

This repository contains two Jupyter notebooks for laminitis risk scoring and model-based prediction using multiple ML algorithms.

## Streamlit Web application
- The [link](https://lampred.streamlit.app/#laminitis-risk-prediction-web-app) is available.

## Notebooks
- `Final_Lamintis_ML_Prediction_2_class_2_21_Sep.ipynb`: End-to-end training, cross-validation, model comparison, ROC/confusion plots, model export, and SHAP explainability.
- `Final_Lamintis_score_2_class_2_21_Sep.ipynb`: Logistic-regression-based scoring system with point allocation and risk estimation table generation.

## Project structure
- `requirements.txt`: Python dependencies.
- `.gitignore`: Ignore caches, outputs, large artifacts.
- Place your dataset file `Laminitis.xlsx` in the project root (same folder as the notebooks). The notebooks expect this filename.

## Quick start
1. Create and activate a virtual environment
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```bash
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Add data
   - Copy `Laminitis.xlsx` to this directory.

4. Launch Jupyter and run a notebook
   ```bash
   python -m pip install jupyter
   jupyter notebook
   ```
   Open one of the notebooks and run all cells in order.

## Outputs
- The prediction notebook saves models and figures to folders that may be created at runtime:
  - `final_models/`: saved scalers and models (`*.pkl`, `*.h5`).
  - `saved_model/`: additional serialized models and feature lists.
  - `explainability/`: SHAP plots.
  - Generated figures like `avg_conf_matrix_*.png`, `KFold_All_Model_ROC_Curves.png`.

These folders and large files are ignored via `.gitignore` to keep the repo lightweight. If you need to commit artifacts, remove the corresponding lines from `.gitignore`.

## Notes
- The notebooks read `Laminitis.xlsx` and will error if it is missing. For GitHub, consider adding a tiny anonymized sample (e.g., `Laminitis_sample.xlsx`) and adjusting paths in the notebook for demos.
- GPU is not required; TensorFlow will run on CPU by default.
- SHAP computations for non-tree models can be slow; reduce `nsamples` if needed.

## Reproducibility
- Random seeds are set in various estimators, but full determinism is not guaranteed due to algorithmic and parallelism differences.

## License
Add a license of your choice (e.g., MIT) if this is going public.
