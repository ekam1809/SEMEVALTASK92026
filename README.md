# SemEval2026-POLAR
This project was worked in collaboration with mhossai6@ualberta.ca
Official repository for UAlberta at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization

This repo contains our systems for the POLAR shared task, covering:

- **Subtask 1** – Binary polarization / hate detection  
- **Subtask 2** – Multi-label hate *types* (5 labels)  
- **Subtask 3** – Multi-label hate *manifestations* (6 labels)

We implement and compare:

- A reproduced **BERT baseline** (organizers’ starter system)  
- Strong **encoder models** (DeBERTa-v3, XLM-R) with MT-based multilingual support  
- **Cross-validated** and **calibrated** models with focal loss  
- A **Qwen2.5-7B** few-shot LLM baseline  
- **Qwen-based data augmentation** (paraphrasing)  
- Final **ensembles** of DeBERTa and XLM-R

---

## Repository structure

```text
.
├── BERT-baseline/           # Organizer-style BERT baseline (all subtasks)
├── method/                  # Single-run DeBERTa & XLM-R systems (no CV)
├── method2/                 # CV + calibration (DeBERTa+MT, XLM-R+MT)
├── method3/                 # Qwen augmentation + CV + calibration
├── llm-baseline/            # Qwen2.5-7B LLM few-shot baseline & utilities
└── README.md                # This file
````

At a high level:

* **BERT-baseline/** – Reproduces the official BERT baseline using the shared-task data.
* **method/** – First improved systems: single-run DeBERTa and XLM-R, with optional MT (for non-English) and simple ensembling.
* **method2/** – Adds K-fold cross-validation, focal loss, temperature scaling + threshold calibration, and ensembling.
* **method3/** – Uses Qwen2.5-7B to *paraphrase* labeled data and train CV models on the augmented corpus.
* **llm-baseline/** – Few-shot prompting of Qwen2.5-7B for all three subtasks (no fine-tuning).

All training and inference are implemented as Jupyter notebooks; you can run them interactively or convert them to scripts.

---

## Environment setup

Tested with:

* Python **3.10+/3.11+**
* CUDA GPU (recommended) for fine-tuning and Qwen2.5-7B

Create a fresh environment (example with `conda`):

```bash
conda create -n polar python=3.10
conda activate polar
```

Install dependencies (minimal set):

```bash
pip install \
  torch \
  transformers \
  accelerate \
  datasets \
  sentencepiece \
  scikit-learn \
  pandas \
  numpy \
  tqdm \
  bitsandbytes
```

If you plan to use the notebooks directly:

```bash
pip install jupyter
```

> **Note:**
>
> * Make sure your PyTorch install is GPU-enabled (e.g., via `pip install torch --index-url https://download.pytorch.org/whl/cu121` or the recommended command from pytorch.org).
> * For Qwen2.5-7B, a GPU with ≥ 16 GB VRAM is strongly recommended; for CPU-only, inference will be very slow.

---

## Data layout

We assume the official POLAR data is available under a directory like:

```text
dev_phase/
  subtask1/
    train/
      eng.csv
      ...
    dev/
      eng.csv
      ...
  subtask2/
    train/
      eng.csv
      ...
    dev/
      eng.csv
      ...
  subtask3/
    train/
      eng.csv
      ...
    dev/
      eng.csv
      ...
```

Each `train` CSV typically contains:

* `id` – unique example ID
* `text` – post text
* **Subtask 1**: `polarization` (0/1)
* **Subtask 2**: 5 binary columns for hate *types*
* **Subtask 3**: 6 binary columns for hate *manifestations*

Each `dev` CSV contains:

* `id`, `text` (no labels)

For Method 3 (augmentation), we create a similar directory:

```text
dev_phase_aug/
  subtask1/...
  subtask2/...
  subtask3/...
```

where each `train` file contains both original and Qwen-paraphrased examples (same label schema).

> **Where to put data:**
> Place `dev_phase/` (and eventually `dev_phase_aug/`) at the repo root or update the `BASE` variable inside notebooks to point to your data directory.

---

## Checkpoints, cache, and outputs

Most notebooks will create and/or expect:

* `cache/` – for:

  * MT outputs (translated text → English)
  * Saved model logits and probabilities
  * Qwen predictions for LLM baselines
* `submissions/` – for:

  * Final CSVs formatted for Codabench / shared-task submission

You can safely delete and regenerate `cache/` and `submissions/`, but keep a copy of any *submitted* runs you care about.

---

## Running the systems

Below is a quick guide to the main workflows. All paths are relative to the repo root.

### 1. BERT baseline

**Notebook:** `BERT-baseline/starter.ipynb`

Steps:

1. Open the notebook in Jupyter:

   ```bash
   jupyter notebook BERT-baseline/starter.ipynb
   ```

2. Set:

   * `BASE = "../dev_phase"`
   * `LANG = "eng"` (or another language if supported)

3. Run all cells. The notebook:

   * Loads the data for each subtask
   * Trains a BERT model (or models) on the train split
   * Evaluates on an internal validation split
   * Writes predictions for the official `dev` set to CSV under `submissions/BERT-baseline/`

This reproduces the official-style baseline.

---

### 2. Method: Single-run DeBERTa & XLM-R

**Main notebooks (per encoder):**

* `method/deberta_train_all_tasks.ipynb`
* `method/xlmr_train_all_tasks.ipynb`
* `method/ensemble_all_tasks.ipynb`

**a. DeBERTa single-run**

1. Open `method/deberta_train_all_tasks.ipynb`.

2. Set configuration cells (near the top), e.g.:

   * `BASE = "../dev_phase"`
   * `LANG = "eng"`
   * `EN_MODEL = "microsoft/deberta-v3-base"`
   * `MAX_LEN`, `EPOCHS`, `LR`, etc.

3. Run all cells:

   * For **Subtask 1**:

     * Train a binary classifier on `text_en` (English or translated)
     * Apply temperature scaling + a global threshold
   * For **Subtasks 2 & 3**:

     * Train multi-label classifiers
     * Fit per-label thresholds on validation data
   * Save dev predictions and thresholds to `cache/` and `submissions/method/deberta/`.

**b. XLM-R single-run**

Repeat the same steps with:

* `method/xlmr_train_all_tasks.ipynb`
* `MODEL_NAME = "xlm-roberta-base"`

**c. Ensemble (DeBERTa + XLM-R)**

1. Open `method/ensemble_all_tasks.ipynb`.
2. Make sure paths to both models’ cached predictions are correct.
3. Run all cells:

   * Loads probabilities from DeBERTa and XLM-R
   * Averages them
   * Applies thresholds
   * Writes ensemble predictions to `submissions/method/ensemble/`.

---

### 3. Method2: Cross-validation + calibration

**Main notebooks:**

* `method2/deberta_mt_cv_train_all_tasks.ipynb`
* `method2/xlmr_cv_train_all_tasks.ipynb`
* `method2/ensemble_cv_xlmr_deberta.ipynb`

This method adds:

* **K-fold CV** (typically K = 3)
* **Focal loss** for multi-label tasks
* **Temperature scaling** and threshold search using **OOF** logits
* Ensembling across folds and across encoders

**a. DeBERTa+MT CV**

1. Open `method2/deberta_mt_cv_train_all_tasks.ipynb`.

2. Set:

   * `BASE = "../dev_phase"`
   * `LANG = "eng"` (or a non-English language with MT configured)
   * `N_FOLDS`, `MAX_LEN`, `EPOCHS`, etc.

3. Run all cells:

   * For each fold:

     * Train on K−1 folds, validate on 1 fold
     * Store OOF logits and labels
   * Fit temperature and thresholds using OOF data
   * Generate calibrated dev predictions
   * Save to `cache/deberta_cv/` and `submissions/method2/deberta_cv/`.

**b. XLM-R CV**

Repeat with `method2/xlmr_cv_train_all_tasks.ipynb`.

**c. CV ensemble (DeBERTa + XLM-R)**

1. Open `method2/ensemble_cv_xlmr_deberta.ipynb`.
2. Run all cells:

   * Loads calibrated probabilities from both CV runs
   * Averages them
   * Applies calibrated thresholds
   * Writes final CSVs to `submissions/method2/ensemble/`.

---

### 4. Method3: Qwen augmentation + CV

**Main notebooks:**

* `method3/qwen_data_augmentation.ipynb`
* `method3/deberta_mt_cv_train_all_tasks.ipynb`
* `method3/xlmr_cv_train_all_tasks.ipynb`
* `method3/ensemble_cv_xlmr_deberta.ipynb`

**a. Qwen paraphrase augmentation**

1. Open `method3/qwen_data_augmentation.ipynb`.

2. Configure Qwen:

   * `MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"`
   * Set device (GPU recommended).

3. Run all cells:

   * For each labeled train example in `dev_phase/`:

     * Prompt Qwen to paraphrase the text *without changing the label*
   * Write augmented CSVs under `../dev_phase_aug/` mirroring the original directory structure.

**b. CV training on augmented data**

1. Open `method3/deberta_mt_cv_train_all_tasks.ipynb` and `method3/xlmr_cv_train_all_tasks.ipynb`.

2. Set:

   * `BASE = "../dev_phase_aug"`
   * Other hyperparameters as desired.

3. Run all cells to train CV models on the augmented corpus and generate calibrated dev predictions.

**c. Final ensemble**

1. Open `method3/ensemble_cv_xlmr_deberta.ipynb`.
2. Run all cells to:

   * Average DeBERTa and XLM-R calibrated probabilities
   * Apply thresholds
   * Save final CSVs to `submissions/method3/ensemble/`.

This is our strongest configuration (MT + augmentation + CV + calibration + ensemble).

---

### 5. LLM baseline (few-shot Qwen)

**Main notebooks:**

* `llm-baseline/qwen2p5_multilingual_llm.ipynb`
* `llm-baseline/qwen_fewshot_predictions.ipynb`

Workflow:

1. Open `llm-baseline/qwen2p5_multilingual_llm.ipynb`.

2. Configure:

   * `MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"`
   * `LANG` and data paths

3. Run:

   * Builds prompts for each subtask
   * Calls Qwen in an instruction-following setup
   * Parses outputs into labels (0/1 vectors)
   * Caches raw outputs and parsed predictions under `cache/qwen/`.

4. Optionally, use `qwen_fewshot_predictions.ipynb` to:

   * Inspect predictions
   * Convert to submission CSV format in `submissions/llm-baseline/`.

---

## Reproducibility notes

* Most notebooks expose:

  * `SEED` or `RANDOM_SEED`
  * `N_FOLDS`, `EPOCHS`, `BATCH_TRAIN_GPU`, etc.
* For full reproducibility:

  * Fix seeds in all relevant libraries (`random`, `numpy`, `torch`).
  * Use a single GPU and avoid non-deterministic CuDNN settings where possible.
* Some variability is expected due to GPU kernels and data loader ordering.
