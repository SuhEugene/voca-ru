#!/bin/sh
# Parallel, resumable, checksum-verified download of one Hugging Face file.
# Usage: fetch-hf-file.sh <url> <out> <sha256> [parts]
# Ranges are fetched concurrently because a single connection is throttled here.
set -eu
url=$1
out=$2
sha=$3
parts=${4:-6}

dir=$(dirname "$out")
base=$(basename "$out")
mkdir -p "$dir/parts-$base"
total=$(curl -fsIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
[ -n "$total" ] || { echo "no content-length for $url" >&2; exit 1; }

chunk=$(( (total + parts - 1) / parts ))
echo "size $total, $parts parts of $chunk"

i=0
while [ "$i" -lt "$parts" ]; do
  s=$((i * chunk)); e=$((s + chunk - 1)); [ "$e" -ge "$total" ] && e=$((total - 1))
  [ "$s" -lt "$total" ] || break
  want=$((e - s + 1)); f="$dir/parts-$base/part.$i"
  (
    tries=0
    while :; do
      have=$(stat -f%z "$f" 2>/dev/null || echo 0)
      [ "$have" -eq "$want" ] && break
      [ "$have" -gt "$want" ] && { rm -f "$f"; have=0; }
      tries=$((tries + 1)); [ "$tries" -gt 60 ] && { echo "part $i gave up" >&2; exit 1; }
      curl -sfL --retry 5 --retry-all-errors --retry-delay 2 -m 7200 \
        -r "$((s + have))-$e" "$url" >> "$f" || sleep 3
    done
  ) &
  i=$((i + 1))
done
wait

rm -f "$out"
i=0
while [ "$i" -lt "$parts" ]; do
  f="$dir/parts-$base/part.$i"
  [ -f "$f" ] && cat "$f" >> "$out"
  i=$((i + 1))
done
got=$(shasum -a 256 "$out" | cut -d' ' -f1)
if [ "$got" = "$sha" ] && [ "$(stat -f%z "$out")" -eq "$total" ]; then
  rm -rf "$dir/parts-$base"; echo "OK $out ($total bytes)"
else
  echo "MISMATCH $out: got $got want $sha" >&2; exit 1
fi
