import json

import torch

from scoreclr.train.train import train


def test_train_resume_and_evaluate(tiny_cfg):
    from scoreclr.evaluation.__main__ import main as evaluate

    metrics = train(tiny_cfg)
    for k in ("eval/online_acc1", "eval/knn_acc1"):
        assert 0.0 <= metrics[k] <= 100.0
    out = tiny_cfg.output_dir
    ckpt = torch.load(f"{out}/last.pt", weights_only=False)
    assert ckpt["epoch"] == tiny_cfg.optim.epochs - 1
    assert len(open(f"{out}/log.jsonl").readlines()) == tiny_cfg.optim.epochs

    # resume a finished run with 1 more epoch: continues from the checkpoint
    tiny_cfg.optim.epochs += 1
    train(tiny_cfg)
    assert torch.load(f"{out}/last.pt", weights_only=False)["epoch"] == tiny_cfg.optim.epochs - 1
    results = evaluate(["--run", out, "--targets", "--num_workers", "0", "--probe_epochs", "1"])
    assert 0.0 <= results["linear"]["linear_acc1"] <= 100.0 and results["stats"]["n"] == 32
    assert json.load(open(f"{out}/evaluation.json"))["monitor"]["eval/knn_acc1"] >= 0


def test_exported_weights_are_evaluated_like_the_checkpoint(tiny_cfg, tmp_path):
    from scoreclr.evaluation.__main__ import main as evaluate
    from scoreclr.utils import export_weights

    tiny_cfg.optim.epochs = 1
    train(tiny_cfg)
    out = tiny_cfg.output_dir
    exported = export_weights(f"{out}/last.pt", tmp_path / "w" / "run.pt")
    assert set(torch.load(exported, weights_only=False)) == {"model", "config", "epoch"}
    args = ["--targets", "--num_workers", "0", "--probe_epochs", "1"]
    a = evaluate(["--run", out, *args])
    b = evaluate(["--run", str(tmp_path / "other"), "--checkpoint", str(exported), *args])
    assert a["monitor"]["eval/knn_acc1"] == b["monitor"]["eval/knn_acc1"]


def test_grad_norm_clips_in_place():
    from scoreclr.train.train import grad_norm

    w = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([w], lr=1.0)
    w.grad = torch.full((4,), 3.0)  # norm 6
    assert torch.isclose(grad_norm(opt), torch.tensor(6.0))  # no max_norm: reports, does not clip
    assert torch.isclose(w.grad.norm(), torch.tensor(6.0))
    assert torch.isclose(grad_norm(opt, 1.0), torch.tensor(6.0))  # returns the norm before clipping
    assert torch.isclose(w.grad.norm(), torch.tensor(1.0), atol=1e-5)


def test_train_with_grad_clip(tiny_cfg):
    tiny_cfg.optim.grad_clip = 1.0
    tiny_cfg.optim.epochs = 1
    train(tiny_cfg)
    log = [json.loads(line) for line in open(f"{tiny_cfg.output_dir}/log.jsonl")]
    assert log[0]["train/enc_grad_norm"] > 0 and log[0]["train/score_grad_norm"] > 0
