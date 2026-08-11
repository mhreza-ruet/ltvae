## ChemInfo-LatentRep

# LSTM-Transformer-VAE (LTVAE) for Molecular SMILES

This project implements a hybrid Variational Autoencoder (**LTVAE**) for molecular SMILES reconstruction and generation. The model uses a **bi-directional LSTM encoder**, the **VAE reparameterization trick**, and a **Transformer decoder**. It supports teacher-forcing training, greedy decoding, beam-search decoding, chemical validity checks, reconstruction accuracy, Levenshtein distance, and novelty analysis.

---

## Project Structure

- **`data_utils.py`**  
  Utilities for SMILES tokenization, randomized SMILES augmentation, vocabulary creation, cached vocabulary loading, SMILES encoding, sampling, canonicalization, and trainable parameter counting.

- **`dataset.py`**  
  PyTorch dataset wrapper for SMILES strings. It encodes SMILES into token sequences, handles padding and special tokens, and supports on-the-fly training augmentation.

- **`model_bs.py`**  
  Model definitions:
  - Positional encoding
  - BiLSTM encoder
  - Transformer decoder
  - VAE wrapper (`LSTM_VAE_Trans`)
  - Greedy decoding
  - Beam-search decoding

- **`train.py`**  
  Training and validation workflow with:
  - Teacher forcing
  - KL annealing
  - Optional token corruption and label smoothing
  - CUDA AMP support
  - Validation with teacher forcing
  - Beam-search validation metrics
  - Early stopping
  - Checkpoint saving

- **`metrics.py`**  
  Evaluation and loss utilities, including reconstruction + KL loss, token accuracy, RDKit chemical validity, Levenshtein distance, canonical SMILES conversion, and novelty metrics.

- **`inference.py`**  
  Reconstruction utilities for trained models. Provides `reconstruct_smiles_table`, beam/greedy inference, validity checks, Levenshtein scoring, and token tensor to SMILES conversion.

- **`device_utils.py`**  
  Device-selection helper for CUDA, Apple MPS, and CPU execution.

- **`plotting.py`**  
  Plotting helper for training curves, validation loss, token accuracy, beam-search accuracy, validity ratio, and average Levenshtein distance.

- **`vocab.json`**  
  Cached SMILES token vocabulary used for consistent encoding and decoding across training and inference.

### Data and Model Artifacts

- **`Data/`**  
  Contains CSV files used for training, validation, and testing:
  - `Train_1.csv`, `Train_2.csv`, `Train_3.csv`
  - `Valid_1.csv`
  - `Test.csv`
  - `Test_pubchem.csv`
  - `dye.csv`
  - `union_586k.csv`

- **`checkpoints/`**  
  Contains saved model artifacts:
  - `best_model.pth`
  - `property_head_best.pt`
  - `training_curves.png`

- **`outputs/`**  
  Contains reconstruction outputs, dye-domain analysis tables, property histograms, distance/statistical summaries, and figure images generated from model evaluation and analysis.

### Notebooks

> **Note:** Before running any notebook, update local paths for data files, saved checkpoints, and output directories where needed.

- **`main.ipynb`**  
  Main training and reconstruction notebook for the LTVAE model. It builds/loads the vocabulary, configures the model, runs training with KL annealing and beam-search validation, plots training curves, and evaluates reconstruction performance on test and dye SMILES.

- **`Test_code_dyes.ipynb`**  
  Evaluates the trained LTVAE model on the dye SMILES dataset. It reconstructs dye molecules with beam search and reports token-level accuracy, exact SMILES match accuracy, validity ratio, and average Levenshtein distance.

- **`Test_pubchem.ipynb`**  
  Evaluates model reconstruction on a PubChem test subset. It runs batched beam-search reconstruction and computes token-level accuracy, exact-match accuracy, validity ratio, and average Levenshtein distance.

- **`Property_prediction.ipynb`**  
  Analyzes the learned latent chemical space. It trains and evaluates a nonlinear property head using latent vectors and molecular descriptors, then explores property prediction, latent traversal, and property relationships.

- **`Fig_4_prop_sas.ipynb`**  
  Computes molecular property values and synthetic accessibility scores for selected input/output SMILES pairs and generates 2D molecule drawings used for Figure 4 preparation.

---

## How to Run

1. **Set up environment**

   Install the required packages in a Python environment. The project uses Python, PyTorch, RDKit, pandas, NumPy, matplotlib, and Jupyter.

   ```bash
   conda create -n ltvae python=3.11
   conda activate ltvae
   conda install -c conda-forge rdkit pandas numpy matplotlib jupyter
   pip install torch
   ```

2. **Prepare data**

   Put training, validation, and test CSV files under `Data/`. Each file should include a column named `smiles`.

   Training, validation and test data can be downloaded from [here](https://drive.google.com/drive/folders/1DPeCl15xXv-mPysPgoZz5EAOHKTJ6_kI?usp=sharing)

3. **Update configuration**

   Edit the `cfg` dictionary in `main.ipynb` to match your local data paths, checkpoint directory, model size, batch size, number of epochs, and decoding settings.

4. **Train**

   Run the training cells in `main.ipynb`, including:

   ```python
   model, history = run_training(cfg, token_to_idx, idx_to_token)
   ```

5. **Plot training curves**

   ```python
   import plotting as plt

   plt.plot_training_curves(
       history,
       metrics_every=cfg["beam_every"],
       save_path="checkpoints/training_curves.png",
   )
   ```

6. **Inference**

   Evaluate a trained model with beam search:

   ```python
   from inference import reconstruct_smiles_table

   df_rec = reconstruct_smiles_table(
       test_csv="Data/Test.csv",
       model=model,
       token_to_idx=token_to_idx,
       idx_to_token=idx_to_token,
       seq_length=cfg["seq_length"],
       pad_idx=cfg["pad_idx"],
       sos_idx=cfg["sos_idx"],
       eos_idx=cfg["eos_idx"],
       device="cuda",
       mode="beam",
       beam_size=cfg["beam_size"],
   )

   print(df_rec.head())
   ```

---

## Reproducibility

- Training, validation, and test splits are provided as CSV files under `Data/`.
- The SMILES vocabulary is cached in `vocab.json` for consistent tokenization.
- Experiment settings are centralized in the `cfg` dictionary inside the notebooks.
- Model checkpoints and generated training curves are saved under `checkpoints/`.
- Reconstruction tables and analysis outputs are saved under `outputs/`.

### Funding

This work was supported by National Science Foundation (Award # 2344423)
