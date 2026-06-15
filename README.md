# DeepTree-AAPred

This repository contains a cleaned project version of the local code that matches the paper:

> DeepTree-AAPred: Binary tree-based deep learning model for anti-angiogenic peptides prediction

The model uses two protein language model branches, ESM-2 and ProtBERT, then combines TextCNN-style convolutional features with BiLSTM features for binary anti-angiogenic peptide prediction.

## Project Layout

```text
data/
  AAIP_135.csv        # processed training split used by the local scripts
  AAIP_28.csv         # processed independent test split
  aaip_135_ori.csv    # original balanced training data copy found locally
  aaip_28_ori.csv     # original balanced test data copy found locally
models/
  .gitkeep            # put local model weights here; weights are intentionally not committed
scripts/
  train_independent.py
  train_5fold.py
src/deeptree_aapred/
  data.py
  model.py
  train_utils.py
```

## Model Weights

Large model weights are not committed to Git. Place local Hugging Face-format directories at:

```text
models/ESM
models/protbert
```

Or point the scripts to existing local directories:

```bash
export ESM_MODEL_PATH=/root/ESM
export PROTBERT_MODEL_PATH=/root/protbert
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Train and Evaluate

Independent test run:

```bash
PYTHONPATH=src python scripts/train_independent.py
```

5-fold validation:

```bash
PYTHONPATH=src python scripts/train_5fold.py
```

Optional arguments can override paths and hyperparameters:

```bash
PYTHONPATH=src python scripts/train_independent.py \
  --esm-model /root/ESM \
  --protbert-model /root/protbert \
  --epochs 10 \
  --batch-size 2
```

## Notes

- The paper reports a train/test distribution of 160/40. The local processed files found in this environment contain 162 training rows and 40 test rows.
- The original scripts returned the fused 128-dimensional feature vector directly to `CrossEntropyLoss`. This cleaned version adds the intended binary classifier layer.

