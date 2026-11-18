"""Download a SARDet-100K archive in byte ranges, resumably.

Three earlier approaches failed, each for a different reason:

  1. The ModelScope SDK restarts rather than resumes. It transferred 24 GB for a
     5.2 GB file across three hours and still failed.
  2. A single long-lived `curl -C -` stalls: the connection stays open and sends
     nothing. `--speed-limit` converts that into a retryable error, but the
     server then stalls the retry too, so it makes little progress.
  3. Any long-running background script here risks being killed, losing the
     process but not the bytes.

So this requests one modest range at a time. A stall costs a single chunk rather
than the transfer, every completed chunk is on disk immediately, and re-running
picks up from whatever is already there. It is deliberately dull and
interruptible.

A note on integrity, learned the hard way: a file of exactly the right length
can still be corrupt. Concurrent writers once produced a 5,242,682,651-byte
train.zip with 2,599 damaged entries. Length is checked here, and the caller
must still verify the zip structure.

    ../.venv/bin/python scripts/fetch_chunked.py train
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BASE = ("https://www.modelscope.cn/api/v1/datasets/wokaikaixinxin/SARDet_100K"
        "/repo?Revision=master&FilePath=JPEGImages%2F")
EXPECTED = {"val": 632524491, "test": 743614368, "train": 5242682651}


def fetch_range(url, start, end, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status not in (200, 206):
            raise urllib.error.HTTPError(url, resp.status, "unexpected status",
                                         resp.headers, None)
        return resp.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(EXPECTED))
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=8)
    args = ap.parse_args()

    total = EXPECTED[args.name]
    url = f"{BASE}{args.name}.zip"
    dest = HERE / "raw" / "JPEGImages"
    dest.mkdir(parents=True, exist_ok=True)
    part = dest / f"{args.name}.zip.part"
    final = dest / f"{args.name}.zip"

    if final.exists() and final.stat().st_size == total:
        print(f"{final.name} already complete")
        return 0

    chunk = args.chunk_mb * 1024 * 1024
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        print(f"partial is larger than expected ({have} > {total}); restarting")
        part.unlink()
        have = 0

    print(f"{args.name}.zip: {have / 1e9:.2f} / {total / 1e9:.2f} GB "
          f"({100 * have / total:.1f}%)")
    started, started_at = have, time.time()

    with open(part, "ab") as fh:
        while have < total:
            end = min(have + chunk - 1, total - 1)
            for attempt in range(1, args.attempts + 1):
                try:
                    data = fetch_range(url, have, end)
                    break
                except Exception as exc:  # stall, reset, 5xx, throttle
                    if attempt == args.attempts:
                        print(f"\n  giving up at byte {have}: {exc}")
                        print("  re-run to continue from here")
                        return 1
                    wait = min(2 ** attempt, 30)
                    print(f"\n  byte {have}: {type(exc).__name__}, "
                          f"retry {attempt}/{args.attempts} in {wait}s")
                    time.sleep(wait)

            if not data:
                print(f"\n  empty response at byte {have}; re-run to continue")
                return 1

            fh.write(data)
            fh.flush()
            have += len(data)

            elapsed = max(time.time() - started_at, 1e-6)
            rate = (have - started) / elapsed / 1e6
            eta = (total - have) / max(rate * 1e6, 1)
            sys.stdout.write(f"\r  {100 * have / total:5.1f}%  "
                             f"{have / 1e9:5.2f}/{total / 1e9:.2f} GB  "
                             f"{rate:5.1f} MB/s  eta {eta / 60:4.1f} min   ")
            sys.stdout.flush()

    print()
    size = part.stat().st_size
    if size != total:
        print(f"size {size}, expected {total}")
        return 1
    part.rename(final)
    print(f"complete -> {final}")
    print("now verify the zip structure: unzip -t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
