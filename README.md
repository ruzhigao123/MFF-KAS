# MFF-KAS: Multimodal Feature Fusion and Kernel-Aware Selection for HLS QoR Prediction

Official implementation of **MFF-KAS** (CDFG + AST + Text fusion experts with **K**ernel-**A**ware expert **S**election via PPO).

> **Note:** The anonymous review link may point to an older repository name (`HEGNN-RCSAC`). This repo contains the MFF-KAS code used in our paper. Please cite the paper and use this repository URL after publication.

## Overview

- **Part 1 — MFF:** Seven modality-specific QoR regressors (all non-empty subsets of `{CDFG, AST, Text}`), each with five metric heads (Latency, LUT, FF, DSP, BRAM).
- **Part 2 — KAS:** Per–QoR-metric policy that selects one frozen expert per unseen kernel at inference time (PPO or supervised training on validation best-expert labels).
/架构图.jpg
**Benchmarks:** MachSuite + PolyBench (22 kernels; 7 held-out test kernels marked † in the paper).

## Repository layout

```
MPM-Innovation-master/
├── src/                    # Expert training & inference (GNN + CodeBERT + MFF)
│   ├── config.py           # Global flags, kernels, paths
│   ├── model.py            # Net / multimodal fusion
│   ├── train.py            # Training & inference loops
│   ├── main.py             # Entry point (train / inference / DSE)
│   └── comp_model/         # Baseline encoders (GNN-DSE, MPM-style, etc.)
├── algorithm_select/       # KAS (selector training & evaluation)
│   ├── select_main.py                  # Main KAS trainer (PPO / supervised)
│   ├── select_model.py                 # Selector network
│   ├── select_dataset.py               # Dataset + labels.json loader
│   ├── build_labels_from_data_csv.py   # Build labels from expert pred CSVs
│   ├── inference_engine*.py            # KAS + expert inference pipelines
│   └── plot_best_expert_heatmap.py     # Fig. 6 style heatmap
├── data/                   # Expert prediction CSVs (for labels / analysis)
├── dse_database/           # ProGraML graphs & HLS configs (large; see below)
└── save_models_and_data/   # Checkpoints (not shipped by default)

> `two_tower_dataset/` and `two_tower_datasets/` (AST `.pt` graphs) are **not** in this repo. See **Data & checkpoints** below.
```

## Environment

- Python 3.9+ (3.9 recommended)
- PyTorch 2.x + CUDA (match your GPU driver)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (CDFG / AST GNN)
- `transformers` + `tokenizers` (CodeBERT)
- `torch-geometric`, `scikit-learn`, `numpy`, `pandas`, `tqdm`, `networkx`

Example install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric transformers scikit-learn pandas tqdm networkx
# Install PyG extensions per https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Data & checkpoints

Large artifacts are **not** included in Git:

| Artifact | Location | Notes |
|----------|----------|--------|
| ProGraML CDFG / configs | `dse_database/` | Generate or obtain from MachSuite/PolyBench DSE pipeline |
| AST `.pt` graphs | `two_tower_datasets/code/ast_final_dataset/` (or `two_tower_dataset/`) | **Not in Git** — download separately; place under repo root |
| Expert weights | `save_models_and_data/` | One checkpoint per modality subset |
| KAS policies | `algorithm_select/save/` or paths in scripts | Per-metric selector |
| Expert pred CSVs | `data/regression_model_pred_{train,test}_<expert>.csv` | For `build_labels_from_data_csv.py` |

Contact the authors or open an issue for a data/checkpoint release link if applicable.

## Quick start (reproduce paper pipeline)

Run from repository root (`MPM-Innovation-master/`).

### 1. Train seven MFF experts

Configure modality / tag in `src/config.py` (or CLI), then:

```bash
cd src
python main.py   # FLAGS.subtask = 'train' in config.py
# or use train_del_0.py / project-specific training script per expert tag
```

Train separately for experts: `all`, `ast`, `cdfg`, `text`, `ast_and_cdfg`, `ast_and_text`, `cdfg_and_text`.

Export train/test predictions to `data/regression_model_pred_{train,test}_<expert>.csv`.

### 2. Build KAS supervision labels

```bash
python algorithm_select/build_labels_from_data_csv.py
# produces labels.json under algorithm_select/ (see script --help)
```

### 3. Train KAS (per QoR metric)

```bash
python algorithm_select/select_main.py --train_mode ppo --target_m_id 0
# target_m_id: 0=Latency(perf), 1=LUT, 2=FF, 3=DSP, 4=BRAM
# repeat for each metric or set target_m_id in src/config.py
```

### 4. Inference & evaluation

```bash
python algorithm_select/inference_engine(加载json版本且数据随机化分).py
# or the inference script version matching your checkpoint layout
python algorithm_select/plot_best_expert_heatmap.py
```

Kernel-level RMSE tables in the paper use macro-average over seven test kernels:  
`atax`, `gemm-p`, `gesummv`, `jacobi-1d`, `jacobi-2d`, `nw`, `spmv-ellpack`.

## Test split (paper)

| Train pool (15 kernels) | Held-out test (†, 7 kernels) |
|-------------------------|------------------------------|
| adi, aes, bicg, … | atax†, gemm-p†, gesummv†, jacobi-1d†, jacobi-2d†, nw†, spmv-ellpack† |

Train/val split on the 15-kernel pool: **8:2** random split; KAS labels from validation errors (Sec. II-B).

## Citation

If you use this code, please cite our paper (bibtex TBD after publication).

## License

Academic research use. See LICENSE file if present; otherwise contact authors.
