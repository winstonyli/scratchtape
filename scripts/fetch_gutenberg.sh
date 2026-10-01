#!/bin/bash
# Fetches the six novels of the `novels6` corpus (docs/gpu_step_design.md, round 11) from Project Gutenberg
# into data/gutenberg/ (gitignored), stripping the licence header and footer. Public domain in the US.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/gutenberg
for entry in 84:frankenstein 1342:pride_and_prejudice 98:tale_of_two_cities 345:dracula 1400:great_expectations 2701:moby_dick; do
  id=${entry%%:*}; name=${entry#*:}
  [ -f data/gutenberg/$name.txt ] && continue
  curl -sfL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o data/gutenberg/$name.raw
  sed -e '1,/^\*\*\* START OF/d' -e '/^\*\*\* END OF/,$d' data/gutenberg/$name.raw > data/gutenberg/$name.txt
  rm data/gutenberg/$name.raw
  echo "$name: $(wc -c < data/gutenberg/$name.txt) bytes"
done
