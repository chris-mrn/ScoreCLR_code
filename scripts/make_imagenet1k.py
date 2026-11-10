# ImageNet-1k (ILSVRC 2012) from the Hugging Face parquet release, ILSVRC/imagenet-1k (gated: accept its
# terms on huggingface.co and `hf auth login` first), as the train/<wnid>/ and val/<wnid>/ JPEG folders
# that data.train_path / data.val_path and make_imagenet100.py expect: 1,281,167 train and 50,000 val
# images. Needs huggingface_hub and pyarrow. Resumable: finished shards are skipped.
import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "ILSVRC/imagenet-1k"
SPLITS = {"train": "train", "validation": "val"}


def convert(job):
    shard, split, out, wnids = job
    done = out / ".done" / f"{shard.name}.done"
    if done.exists():
        return 0
    n = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=512, columns=["image", "label"]):
        for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            name = os.path.basename(image["path"] or f"{shard.stem}_{n}.JPEG")
            (out / split / wnids[label] / name).write_bytes(image["bytes"])
            n += 1
    done.touch()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("download", type=Path, help="where the parquet shards are downloaded")
    p.add_argument("out", type=Path, help="the ImageFolder tree to write")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    snapshot_download(REPO, repo_type="dataset", local_dir=args.download, max_workers=args.workers,
                      allow_patterns=["classes.py", "data/train-*.parquet", "data/validation-*.parquet"])
    # label i of the release is the i-th wnid of classes.py (the sorted wnid order of ILSVRC 2012)
    wnids = re.findall(r'"(n\d{8})"', (args.download / "classes.py").read_text())
    assert len(wnids) == len(set(wnids)) == 1000, len(wnids)

    jobs = []
    for prefix, split in SPLITS.items():
        for w in wnids:
            (args.out / split / w).mkdir(parents=True, exist_ok=True)
        jobs += [(s, split, args.out, wnids) for s in sorted((args.download / "data").glob(f"{prefix}-*.parquet"))]
    (args.out / ".done").mkdir(exist_ok=True)
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(convert, jobs), 1):
            print(f"{i}/{len(jobs)} shards ({n} images)", flush=True)
    for split in SPLITS.values():
        print(split, sum(len(os.listdir(args.out / split / w)) for w in wnids), "images")


if __name__ == "__main__":
    main()
