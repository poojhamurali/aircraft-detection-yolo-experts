#!/usr/bin/env bash
# Fetch the SARDet-100K image archives, one writer at a time, verified.
#
# Written after two failed attempts, both worth recording:
#
#   1. The ModelScope SDK restarts a failed transfer instead of resuming. It
#      pulled 24 GB over three hours for a 5.2 GB file and still failed, leaving
#      a partial file whose tail was garbage.
#   2. Resuming that partial with curl produced a file of exactly the right
#      length that was still corrupt, because several curl processes were
#      appending to it concurrently with -C -. Size is not integrity.
#
# So: a pid lock allows only one writer, downloads land in a .part file, and
# nothing is moved into place until both its length and its zip structure check
# out. A file that is already present and valid is skipped, which makes this
# safe to re-run.
#
# Usage: bash scripts/fetch_images.sh [val test train]

set -uo pipefail
cd "$(dirname "$0")/.."

BASE="https://www.modelscope.cn/api/v1/datasets/wokaikaixinxin/SARDet_100K/repo?Revision=master&FilePath=JPEGImages%2F"
DEST=raw/JPEGImages
mkdir -p "$DEST"

# A case statement rather than an associative array: macOS ships bash 3.2,
# which has no `declare -A`.
expected_bytes () {
    case "$1" in
        val)   echo 632524491 ;;
        test)  echo 743614368 ;;
        train) echo 5242682651 ;;
        *)     echo "unknown archive: $1" >&2; exit 1 ;;
    esac
}

LOCK=raw/.fetch.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    owner=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
        echo "another fetch is already running (pid $owner). Not starting a second." >&2
        exit 1
    fi
    echo "reclaiming stale lock from dead pid ${owner:-unknown}" >&2
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT

verify () {  # file  expected_bytes
    local f="$1" want="$2" have
    have=$(wc -c < "$f" | tr -d ' ')
    if [ "$have" != "$want" ]; then
        echo "    size $have, expected $want"
        return 1
    fi
    if ! unzip -t "$f" > /tmp/$(basename "$f").test.log 2>&1; then
        echo "    zip structure is damaged ($(grep -ac 'bad zipfile\|error' /tmp/$(basename "$f").test.log) bad entries)"
        return 1
    fi
    return 0
}

for name in "${@:-val test train}"; do
    want="$(expected_bytes "$name")"
    final="$DEST/$name.zip"
    part="$DEST/$name.zip.part"

    echo "=== $name.zip ($((want / 1000000)) MB)"
    if [ -f "$final" ] && verify "$final" "$want"; then
        echo "    already present and valid; skipping"
        continue
    fi

    # A damaged final file is discarded rather than resumed: the damage is
    # somewhere inside it and resuming would only re-append to bad data.
    rm -f "$final"

    for attempt in 1 2 3; do
        echo "    attempt $attempt ..."
        # --speed-limit/--speed-time are what make the retries useful. This
        # server stalls silently: the connection stays open and sends nothing,
        # so curl sees no error and --retry never fires. Treating "under 10 kB/s
        # for 60 s" as a failure converts the stall into a retryable error, and
        # -C - then resumes from the bytes already on disk.
        curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
             --connect-timeout 30 --speed-limit 10000 --speed-time 60 \
             -m 7200 \
             -o "$part" "${BASE}${name}.zip" 2>&1 | tail -1
        if verify "$part" "$want"; then
            mv "$part" "$final"
            echo "    ok -> $final"
            break
        fi
        echo "    invalid; discarding and starting over"
        rm -f "$part"
    done

    [ -f "$final" ] || { echo "    FAILED after 3 attempts" >&2; exit 1; }
done

echo
echo "all archives present and verified:"
ls -l "$DEST"/*.zip | awk '{printf "  %12d  %s\n", $5, $9}'
