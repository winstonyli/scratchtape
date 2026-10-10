#!/bin/bash
# Extra training-only novels for the `novels22` corpus (docs/tiers_design.md): sixteen more Project Gutenberg books
# into data/gutenberg_extra/ (gitignored), licence header and footer stripped. Public domain in the US.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/gutenberg_extra
for entry in 158:emma 161:sense_and_sensibility 105:persuasion 730:oliver_twist 766:david_copperfield 2852:hound_of_the_baskervilles \
  74:tom_sawyer 76:huckleberry_finn 35:time_machine 36:war_of_the_worlds 110:tess 219:heart_of_darkness 1260:jane_eyre \
  768:wuthering_heights 174:dorian_gray 120:treasure_island; do
  id=${entry%%:*}; name=${entry#*:}
  [ -f data/gutenberg_extra/$name.txt ] && continue
  curl -sfL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o data/gutenberg_extra/$name.raw
  title=$(grep -m1 '^Title:' data/gutenberg_extra/$name.raw | tr -d '\r')
  sed -e '1,/^\*\*\* START OF/d' -e '/^\*\*\* END OF/,$d' data/gutenberg_extra/$name.raw > data/gutenberg_extra/$name.txt
  rm data/gutenberg_extra/$name.raw
  echo "$name: $(wc -c < data/gutenberg_extra/$name.txt) bytes  [$title]"
done
