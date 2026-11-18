#!/bin/sh
# Parallel, resumable, checksum-verified download of one Hugging Face file.
# Usage: fetch-hf-file.sh <url> <out> <sha256> [parts]
#
# A single connection is throttled here, so ranges are fetched concurrently.
# The CDN sometimes ignores the end of a range and streams to EOF; `head -c`
# caps every attempt at the exact remaining byte count, so a part can never
# overshoot and the assembled file is byte-exact.
set -eu
url=$1
out=$2
sha=$3
parts=${4:-16}

dir=$(dirname "$out")
base=$(basename "$out")
pdir="$dir/parts-$base"
mkdir -p "$pdir"
total=$(curl -fsIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
[ -n "$total" ] || { echo "no content-length for $url" >&2; exit 1; }

chunk=$(( (total + parts - 1) / parts ))
echo "size $total, $parts parts of $chunk"

i=0
while [ "$i" -lt "$parts" ]; do
  s=$((i * chunk)); e=$((s + chunk - 1)); [ "$e" -ge "$total" ] && e=$((total - 1))
  [ "$s" -lt "$total" ] || break
  want=$((e - s + 1)); f="$pdir/part.$i"
  (
    tries=0
    while :; do
      have=$(stat -f%z "$f" 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      tries=$((tries + 1)); [ "$tries" -gt 80 ] && { echo "part $i gave up at $have/$want" >&2; exit 1; }
      remain=$((want - have))
      curl -sfL --retry 5 --retry-all-errors --retry-delay 2 -m 7200 \
        -r "$((s + have))-$e" "$url" | head -c "$remain" >> "$f" || sleep 2
      now=$(stat -f%z "$f" 2>/dev/null || echo 0)
      if [ "$now" -eq "$have" ]; then sleep 5; fi   # no progress: back off
    done
    [ "$(stat -f%z "$f")" -eq "$want" ] || { echo "part $i wrong size" >&2; exit 1; }
  ) &
  i=$((i + 1))
done
wait

rm -f "$out"
i=0; while [ "$i" -lt "$parts" ]; do
  f="$pdir/part.$i"; [ -f "$f" ] && cat "$f" >> "$out"; i=$((i + 1))
done
got=$(shasum -a 256 "$out" | cut -d' ' -f1)
if [ "$got" = "$sha" ] && [ "$(stat -f%z "$out")" -eq "$total" ]; then
  rm -rf "$pdir"; echo "OK $out ($total bytes)"
else
  echo "MISMATCH $out: got $got want $sha" >&2; exit 1
fi
