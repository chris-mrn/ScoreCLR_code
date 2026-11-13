# ScoreCLR

PyTorch implementation of ScoreCLR, self-supervised representation learning where the contrastive
repulsion is replaced by a learned score, and of the baselines of the paper (SimCLR, LeJEPA, BYOL and
Barlow Twins) in one code base: the methods share the encoder, the data pipeline, the training loop and
the evaluation, and differ only in their objective.

ScoreCLR alternates two updates on the codes `z` of two augmented views:

1. **Score net** (denoising score matching on the detached codes), with `z_t = z + σε` and `ln σ ~ N(p_mean, p_std²)`:

   `‖ε̂(z_t, σ) − ε‖²`

2. **Encoder**, against the frozen score `s = −ε̂/σ` (every term divided by the code dimension):

   `attr_weight · ‖z₀ − z₁‖²  +  ½‖s(z_t)‖²  +  ½‖z_t‖²`

   The repulsion term is **ScoreReg** in the paper: the score-based estimate of the entropy term of the
   mutual information. It has gradient `s` with respect to `z`, so descending it pushes codes towards
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
analysis/               tooling outside the package: run-to-run diagnostics, the Poincare-constant
                        estimate (poincare_constant.py), the analytical cost model and the measured
                        GPU cost (benchmarks/)
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

One command trains every row, evaluates it, prints each table next to the paper's numbers and draws
the analytical cost model:

```bash
pip install -r requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu121 && pip install -e .
python scripts/make_imagenet100.py /path/to/imagenet /path/to/imagenet100     # ImageNet-100 split (symlinks)
DATA=./datasets IMAGENET100=/path/to/imagenet100 bash scripts/reproduce_paper.sh  # all rows
bash scripts/reproduce_paper.sh cifar10_scoreclr                                  # or a single row
```

`DATA` receives CIFAR-10, STL-10 and DTD (downloaded by torchvision). CIFAR-10 rows use one GPU,
ImageNet-100 rows `NGPU=2` GPUs with 128 images per GPU. The tables alone:
`python -m scoreclr.evaluation.report runs`; the analytical cost model: `python analysis/benchmarks/cost_model.py out.pdf`.

| row (`config/<row>.yaml`) | seeds in the paper | linear / kNN top-1 (paper) |
|---|---|---|
| `cifar10_scoreclr` | 1, 2, 3 | 88.87 ± 0.44 / 88.27 ± 0.13 |
| `cifar10_simclr_adamw` | 1, 2, 3 | 89.26 ± 0.38 / 88.73 ± 0.18 |
| `cifar10_lejepa` | 1, 2, 3 | 89.67 ± 0.13 / 88.40 ± 0.13 |
| `cifar10_byol` | 1 | 88.90 / 88.37 |
| `cifar10_barlow` | 1 | 87.89 / 87.26 |
| `cifar10_simclr_lars` | 5 | 90.79 / 89.99 |
| `imagenet100_scoreclr` | 1, 2, 3 | 71.23 ± 0.06 / 66.05 ± 0.39 |
| `imagenet100_simclr_adamw` | 1, 2, 3 | 72.39 ± 0.62 / 66.41 ± 0.44 |
| `imagenet100_simclr_lars` | 5 | 75.40 / 69.98 |
| `imagenet100_simclr_lars_proj4096` | 5 | 77.02 / 71.60 |

The ScoreCLR, SimCLR (AdamW) and LeJEPA rows are means ± standard deviations over seeds 1, 2 and 3
(`seed=<n>` overrides the seed of a config); the other rows are single runs. The paper's SimCLR (LARS) rows
were trained with solo-learn itself (its own augmentation pipeline, 16-bit mixed precision); the
`*_simclr_lars` configs re-implement that recipe here, with the loss, LARS and schedules ported from
solo-learn. The other baselines' objectives are also ported from solo-learn.

What to expect:

- **Evaluation protocol.** The kNN numbers of the SimCLR (LARS) rows were computed with a memory bank under
  crop + flip; the shared protocol above, which this repository applies to every method, does not reproduce
  them exactly (the report marks them `*`).
- **Training is not deterministic** (cuDNN, fp32 without deterministic flags): reruns of one config and
  seed diverge after about 20 epochs, so final accuracies vary across reruns.
- `imagenet100_simclr_lars` was pretrained with NVIDIA DALI data loading; this repository uses torchvision.
- The transfer penalty is selected on a validation split; near-ties can flip between reruns on GPU.
- Epps-Pulley and uniformity are random statistics (seeded here); their last digit can differ.

## Other experiments of the paper

Every key of a config can be overridden on the command line (`key.sub=value`); set `output_dir` so that a
variant does not resume another run.

- **Batch size** (Table 5, CIFAR-10, `seed=5`): `optim.batch_size` (per GPU), the encoder learning rate
  `optim.lr` scaled by sqrt(B/128) for both methods, and for ScoreCLR `optim.score_steps` (S) as in the
  caption of Table 5; the score-network learning rate is not scaled.
- **Mixed precision** (Table 6): `optim.precision=fp16`.
- **Ablations** (Appendix C): `loss.num_queries` (K), `optim.score_steps` (S), `loss.p_std`, `optim.grad_clip`,
  `optim.score_grad_clip`.
- **Poincare constant** (Table 3, Appendix C.8), on the codes saved by the evaluation:
  `python analysis/poincare_constant.py --runs scoreclr=RUN1,RUN2,RUN3 simclr=...`.

`analysis/benchmarks/` also measures the GPU time and memory of the objectives against batch size
(`measured_objective_cost.py`, `plot_measured_cost.py`) and the wall-clock time per epoch
(`measured_epoch_time.py`), with the training code itself.

## Licence

MIT (see `LICENSE`), with one exception: `src/scoreclr/methods/lejepa.py` adapts the reference
implementation of LeJEPA's SIGReg (https://github.com/rbalestr-lab/lejepa), which is licensed under
CC BY-NC 4.0, so that file is distributed under CC BY-NC 4.0 (non-commercial use only).
`src/scoreclr/evaluation/stats.py` imports SIGReg from it for the Epps-Pulley statistic. The files ported
from solo-learn (https://github.com/vturrisi/solo-learn, MIT) keep its copyright notice.
