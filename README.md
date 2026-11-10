# ScoreCLR

PyTorch implementation of ScoreCLR, self-supervised representation learning where the contrastive
repulsion is replaced by a learned score, and of the two baselines of the paper, SimCLR and LeJEPA,
in one code base: the three methods share the encoder, the data pipeline, the training loop and the
evaluation, and differ only in their objective.

ScoreCLR alternates two updates on the codes `z` of two augmented views:

1. **Score net** (denoising score matching on the detached codes), with `z_t = z + σε` and `ln σ ~ N(p_mean, p_std²)`:

   `‖ε̂(z_t, σ) − ε‖²`

2. **Encoder**, against the frozen score `s = −ε̂/σ` (every term divided by the code dimension):

   `attr_weight · ‖z₀ − z₁‖²  +  ½‖s(z_t)‖²  +  ½‖z_t‖²`

   The repulsion term has gradient `s` with respect to `z`, so descending it pushes codes towards
   lower density. With the Gaussian prior term it estimates `KL(Z ‖ N(0, I))`.

## Layout

```
config/                 one config per row of the paper's tables (config/<row>.yaml)
src/scoreclr/
  data/                 augmentations, datasets, loaders
  models/               ResNet backbone, encoder (backbone + projector + online classifier), score net
  methods/              SSLMethod (base: shared encoder, objective on the two views' codes)
                        and its subclasses ScoreCLR, SimCLR, LeJEPA, BYOL, Barlow Twins
  train/                the training loop shared by every method; optimizers (AdamW, LARS) and schedules
  evaluation/           the evaluation shared by every method: kNN, linear probe, geometry, rank,
                        normality, transfer, and the paper's tables
  utils/                config, distributed, checkpoints, logging
  tests/
scripts/                reproduce_paper.sh, make_imagenet100.py
analysis/               research tooling outside the package: run-to-run diagnostics, functional
                        constants, cost model (paper figure) and measured GPU cost (benchmarks/)
```

A method is a subclass of `scoreclr.methods.SSLMethod` that implements `loss(z0, z1)`; ScoreCLR also
implements `auxiliary_step(z, optimizer)`, the score-net updates the trainer runs on the detached codes
before every encoder update. `method:` in the config selects it.

## Usage

```bash
pip install -e ".[test]"

# train (CIFAR-10 downloads to data.root); any config key can be overridden
python -m scoreclr.train --config config/cifar10_scoreclr.yaml data.root=./datasets

# multi-GPU (DDP): batch_size is per GPU, BatchNorm is synchronized, rank 0 logs and monitors
torchrun --nproc_per_node=2 -m scoreclr.train --config config/imagenet100_scoreclr.yaml \
    data.train_path=/path/to/imagenet100/train data.val_path=/path/to/imagenet100/val

# evaluate a run: linear probe, kNN, geometry / rank / normality, transfer -> <run>/evaluation.json
python -m scoreclr.evaluation --run runs/cifar10_scoreclr --root stl10=./datasets --root dtd=./datasets

pytest
```

Training resumes automatically from `<output_dir>/last.pt`. Training metrics go to
`<output_dir>/log.jsonl`, and to W&B with `wandb.enabled=true`.

## Evaluation protocol

The same for every method and every run (`scoreclr.evaluation`):

- **Linear probe**: 100 epochs, SGD (momentum 0.9, no weight decay, batch 256) on the frozen backbone with
  crop + flip. CIFAR: raw features, lr 0.1 divided by 10 at epochs 60 and 80. ImageNet-100:
  L2-normalized features, lr 10, linear warm-up over 1 epoch then cosine, per step.
- **kNN**: k = 20, temperature 0.07, cosine, memory bank = the train set under the test transform.
- **Geometry, rank, normality**: alignment and uniformity of the projector output, RankMe of the backbone
  features and of the projector output, mean excess kurtosis and SIGReg's Epps-Pulley statistic of the
  ZCA-whitened projector output, on the validation set.
- **Transfer** (Kornblith et al., 2019): logistic regression on L2-normalized frozen features of STL-10 and
  DTD (and CIFAR-10 for ImageNet-100 backbones) at the pretraining resolution, L2 penalty chosen on a
  validation split.

## Reproducing the paper

One command trains every row, evaluates it, prints each table next to the published numbers and draws
the cost figure:

```bash
pip install -r requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu121 && pip install -e .
python scripts/make_imagenet100.py /path/to/imagenet /path/to/imagenet100     # ImageNet-100 split (symlinks)
DATA=./datasets IMAGENET100=/path/to/imagenet100 bash scripts/reproduce_paper.sh  # all rows
bash scripts/reproduce_paper.sh cifar10_scoreclr                                  # or a single row
```

`DATA` receives CIFAR-10, STL-10 and DTD (downloaded by torchvision). CIFAR-10 rows use one GPU,
ImageNet-100 rows `NGPU=2` GPUs with 128 images per GPU. The tables alone:
`python -m scoreclr.evaluation.report runs`; the cost figure: `python analysis/benchmarks/cost_model.py out.pdf`.

| row (`config/<row>.yaml`) | hardware | linear / kNN top-1 (paper) |
|---|---|---|
| `cifar10_scoreclr` | 1 H100, about 3.2 h | 89.33 / 88.53 |
| `cifar10_simclr_adamw` | 1 H100 | 89.45 / 88.56 |
| `cifar10_lejepa` | 1 L40S | 90.07 / 88.35 |
| `cifar10_simclr_lars` | 1 H100 | 90.79 / 89.99 |
| `imagenet100_scoreclr` | 2 H100, about 7 h | 71.28 / 66.02 |
| `imagenet100_simclr_adamw` | 2 A100 | 72.60 / 65.06 |
| `imagenet100_simclr_lars` | 2 L40S | 75.40 / 69.98 |
| `imagenet100_simclr_lars_proj4096` | 2 H100 | 77.02 / 71.60 |

Every result is a single run with seed 5. The objectives, LARS and schedules of the baselines are
ported from solo-learn.

What to expect:

- **Evaluation protocol.** The published kNN of the baseline rows used a memory bank under crop + flip, and
  the ImageNet-100 ScoreCLR probe had no warm-up; the shared protocol above, which this repository applies
  to every method, does not reproduce those numbers exactly (the report marks them `*`).
- **Training is not deterministic** (cuDNN, fp32 without deterministic flags): reruns of one config and
  seed diverge after about 20 epochs, so final accuracies vary across reruns.
- `imagenet100_simclr_lars` was pretrained with NVIDIA DALI data loading; this repository uses torchvision.
- The transfer penalty is selected on a validation split; near-ties can flip between reruns on GPU.
- Epps-Pulley and uniformity are random statistics (seeded here); their last digit can differ.

`analysis/benchmarks/` also measures the GPU time and memory of the objectives against batch size
(`measured_objective_cost.py`, `plot_measured_cost.py`) and the wall-clock time per epoch
(`measured_epoch_time.py`), with the training code itself.
