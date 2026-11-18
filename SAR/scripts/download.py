"""Fetch SARDet-100K from ModelScope.

The official SARDet-100K page offers only Baidu Netdisk and OneDrive, both of
which need an account. ModelScope carries the same release and allows anonymous
download, so it is the only route here that does not require credentials.

Images arrive as three zips totalling about 6.6 GB. We need roughly 17% of them
(the aircraft subset), but a zip cannot be extracted selectively from a remote
server, so the whole archive comes down and prepare_dataset.py takes what it
needs.

    ../.venv/bin/python scripts/download.py            # annotations + images
    ../.venv/bin/python scripts/download.py --annotations-only
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = "wokaikaixinxin/SARDet_100K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--dest", default=str(HERE / "raw"))
    args = ap.parse_args()

    socket.setdefaulttimeout(300)
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    patterns = ["Annotations/*.json"]
    if not args.annotations_only:
        patterns.append("JPEGImages/*.zip")

    path = dataset_snapshot_download(REPO, allow_patterns=patterns,
                                     local_dir=args.dest, max_workers=4)
    print(f"\ndownloaded to {path}")
    for p in sorted(Path(args.dest).rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size / 1e6:10.1f} MB  "
                  f"{p.relative_to(args.dest)}")


if __name__ == "__main__":
    main()
