# G-StageNet
This repository contains the official implementation of **G-StageNet**,
a lightweight hybrid feature fusion framework for multiclass glaucoma
stage classification from fundus images.

## Overview
G-StageNet integrates handcrafted radiomic biomarkers with deep CNN
embeddings using an adaptive feature-level gating mechanism trained
under a triplet-loss objective.

For reproducibility and simplicity, preprocessing, training, feature fusion,
evaluation, and visualization are implemented in a single executable script.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
