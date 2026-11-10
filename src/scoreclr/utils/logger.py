import json
from pathlib import Path


class Logger:
    """Prints metrics, appends them to <output_dir>/log.jsonl, and optionally to W&B."""

    def __init__(self, output_dir: Path, cfg):
        self.path = Path(output_dir) / "log.jsonl"
        self.wandb = None
        if cfg.wandb.enabled:
            import wandb

            id_file = Path(output_dir) / "wandb_id.txt"  # same W&B run when resuming
            if not id_file.exists():
                id_file.write_text(wandb.util.generate_id())
            self.wandb = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.get("group"),
                tags=cfg.wandb.get("tags"),
                name=Path(output_dir).name,
                config=cfg.to_dict(),
                dir=str(output_dir),
                resume="allow",
                id=id_file.read_text().strip(),
            )

    def log(self, metrics: dict, step: int) -> None:
        metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        print(json.dumps({"step": step, **metrics}), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
