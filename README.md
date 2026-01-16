# 7015-University-ML-based-on-SLAKE-Med-VQA-dataset

Please Download SLAKE dataset : https://www.med-vqa.com/slake/#gt-Download


# Med-VQA (SLAKE) — BLIP Fine-tuning (Colab)

This script fine-tunes **BLIP** (`Salesforce/blip-vqa-base`) on the **SLAKE Medical Visual Question Answering (Med-VQA)** dataset. It provides an end-to-end pipeline from dataset extraction and exploratory visualization to training, evaluation, and checkpoint saving.

> The code is exported from a Google Colab notebook and contains Colab-specific commands such as `!pip install ...` and `drive.mount(...)`, so it is best run on **Google Colab with a GPU**.

---

## What it does

- **Environment setup**: installs common ML dependencies (Transformers, PyTorch, Accelerate, Matplotlib/Seaborn, etc.)
- **Dataset preparation**:
  - reads `Slake.zip` from Google Drive and extracts it to a local working directory
  - automatically discovers `train.json` and the image folder (`imgs/` or `images/`) by walking the extracted directory
  - uses only samples with `q_lang == 'en'` (English questions)
- **Exploratory visualization**:
  - question type distribution (Closed: Yes/No vs Open-Ended)
  - top-10 most frequent answers
  - random sample visualization (image + question + answer)
- **Training strategy**:
  - model: `BlipForQuestionAnswering`
  - freezes the **vision encoder** to improve stability and reduce GPU memory usage
  - **gradient accumulation** (small physical batch, larger effective batch)
  - **early stopping** (stop if accuracy does not improve for `PATIENCE` epochs)
  - saves the best checkpoint (by test accuracy)
- **Metrics**:
  - robust accuracy (special handling for Yes/No)
  - token-level F1 score
- **Outputs**:
  - per-epoch logs: accuracy / F1 / open vs closed accuracy / loss
  - best model weights saved to `CHECKPOINT_PATH`

---

## Recommended environment

- **Google Colab + GPU** (e.g., T4)

The notebook installs dependencies automatically. If you want to run locally, convert the Colab magic commands:

- replace `!pip install ...` with your shell/pip installation
- remove or replace `from google.colab import drive` and `drive.mount(...)`

---

## Dataset setup

By default, the script expects a ZIP in Google Drive:

- `ZIP_PATH = '/content/drive/MyDrive/MV_Project/Slake.zip'`

After extraction, it searches recursively for:

- `train.json`
- an image directory named `imgs/` or `images/`

You do not need to manually specify deeper subdirectories as long as these exist somewhere inside the extracted folder.

---

## Quick start (Colab)

1. Upload `Slake.zip` to Google Drive and ensure it matches `ZIP_PATH`.
2. Adjust config variables at the top of the script (see below).
3. Run the script cells in order (Step 1 → Step 6).
4. During training, whenever the **test accuracy** reaches a new best value, the script saves a checkpoint to `CHECKPOINT_PATH`.

---

## Key configuration

Edit these at the top of the script:

- `ZIP_PATH`: dataset zip path (Google Drive)
- `EXTRACT_PATH`: extraction directory
- `CHECKPOINT_PATH`: where to save the best model weights (`.pth`)
- `BATCH_SIZE`: physical batch size (GPU-memory sensitive)
- `GRAD_ACCUM_STEPS`: gradient accumulation steps
  - effective batch size = `BATCH_SIZE * GRAD_ACCUM_STEPS`
- `LEARNING_RATE`: fine-tuning LR (commonly `1e-5` to `5e-5`)
- `NUM_EPOCHS`: max epochs
- `PATIENCE`: early-stopping patience
- `BASELINE_ACC`: baseline accuracy for comparison (script sets `73.8`)

Default settings in the script:

- `BATCH_SIZE = 4`
- `GRAD_ACCUM_STEPS = 8`  → effective batch size = **32**

---

## Implementation details

### Input formatting

- **Image**: RGB image
- **Text input**: the question is formatted as

```text
Question: <question> Answer:
```

### Labels and generation

- labels: answers are tokenized with `max_length=8` and padded/truncated
- prediction: uses `model.generate(..., max_new_tokens=10)` and decodes to text

### Model + optimization

- base model: `Salesforce/blip-vqa-base`
- freezes vision encoder:

```python
for param in model.vision_model.parameters():
    param.requires_grad = False
```

- optimizer: `AdamW` over trainable parameters only

---

## Metrics

### Robust accuracy

- **Closed (Yes/No)**: counts as correct if the generated text contains the correct token (`yes` or `no`).
- **Open-ended**: tolerant matching using containment checks:
  - `truth in pred` or (`pred in truth` and `len(pred) > 2`) counts as correct.

### Token-level F1

- lowercases, removes punctuation, tokenizes by whitespace
- computes precision/recall/F1 based on token overlap

---

## Outputs

Each epoch prints:

- test accuracy (overall)
- test F1 (overall)
- open-ended accuracy
- closed-ended accuracy
- test loss

And:

- when test accuracy improves, the script saves weights to `CHECKPOINT_PATH`.

---

## Known issue (Step 6 plotting/report)

The plotting/report section (Step 6) references keys like `history['total_acc_te']`, `history['total_f1_te']`, etc., but the training loop (Step 5) stores metrics using different keys:

- `tr_loss`, `te_loss`, `te_acc`, `te_f1`, `te_o_acc`, `te_c_acc`

As a result, running Step 6 **as-is** may raise a `KeyError`.

### Minimal fix (rename keys used in Step 6)

Replace:

- `total_acc_te` → `te_acc`
- `total_f1_te` → `te_f1`
- `open_acc_te` → `te_o_acc`
- `closed_acc_te` → `te_c_acc`
- `total_loss_te` → `te_loss`
- `total_loss_tr` → `tr_loss`

And set the epoch range based on an existing key:

```python
epochs = range(1, len(history['te_acc']) + 1)
```

> Note: the current training loop does **not** record train-set accuracy/F1 curves. If you want complete Train-vs-Test plots, you need to either evaluate on the train loader each epoch (slower) or compute training metrics during the loop.

---

## License & citation

- Model: **BLIP** (Salesforce)
- Dataset: **SLAKE** (Medical VQA)

If you use this in a course report or paper, please cite the original BLIP and SLAKE/SLAKE-related papers as required.
