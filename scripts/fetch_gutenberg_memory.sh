#!/bin/bash
# Extra memory-only text for tier_eval `memory_corpus=novels40` (docs/tiers_design.md): eighteen more Project Gutenberg books
# into data/gutenberg_memory/ (gitignored), licence header and footer stripped. Public domain in the US.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/gutenberg_memory
for entry in 514:little_women 145:middlemarch 599:vanity_fair 1023:bleak_house 141:mansfield_park 121:northanger_abbey 821:dombey_and_son \
  967:nicholas_nickleby 963:little_dorrit 786:hard_times 550:silas_marner 507:adam_bede 33:scarlet_letter 421:kidnapped \
  1257:three_musketeers 82:ivanhoe 155:moonstone 583:woman_in_white; do
  id=${entry%%:*}; name=${entry#*:}
  [ -f data/gutenberg_memory/$name.txt ] && continue
  curl -sfL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o data/gutenberg_memory/$name.raw
  title=$(grep -m1 '^Title:' data/gutenberg_memory/$name.raw | tr -d '\r')
  sed -e '1,/^\*\*\* START OF/d' -e '/^\*\*\* END OF/,$d' data/gutenberg_memory/$name.raw > data/gutenberg_memory/$name.txt
  rm data/gutenberg_memory/$name.raw
  echo "$name: $(wc -c < data/gutenberg_memory/$name.txt) bytes  [$title]"
done
