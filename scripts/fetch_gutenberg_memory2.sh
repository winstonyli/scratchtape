#!/bin/bash
# More memory-only text for tier_eval `memory_corpus=novels90` (docs/tiers_design.md): fifty-one more Project Gutenberg books
# into data/gutenberg_memory2/ (gitignored), licence header and footer stripped. Public domain in the US.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/gutenberg_memory2
for entry in 2554:crime_and_punishment 2600:war_and_peace 1399:anna_karenina 28054:brothers_karamazov 135:les_miserables \
  1184:count_of_monte_cristo 1258:ten_years_later 244:study_in_scarlet 108:return_of_sherlock_holmes 2097:sign_of_the_four \
  580:pickwick_papers 564:edwin_drood 700:old_curiosity_shop 917:barnaby_rudge 883:our_mutual_friend 143:mayor_of_casterbridge \
  153:jude_the_obscure 767:agnes_grey 969:tenant_of_wildfell_hall 9182:villette 394:cranford 4276:north_and_south \
  4274:wives_and_daughters 43:jekyll_and_hyde 5658:lord_jim 2021:nostromo 974:secret_agent 5230:invisible_man \
  159:island_of_doctor_moreau 1013:first_men_in_the_moon 164:twenty_thousand_leagues 103:around_the_world 18857:journey_to_the_centre \
  77:house_of_the_seven_gables 940:last_of_the_mohicans 2788:little_men 215:call_of_the_wild 910:white_fang 113:secret_garden \
  146:little_princess 45:anne_of_green_gables 236:jungle_book 2776:four_million 2166:king_solomons_mines 3155:she \
  521:robinson_crusoe 829:gullivers_travels 6593:tom_jones 86:connecticut_yankee 1837:prince_and_the_pauper 3176:innocents_abroad; do
  id=${entry%%:*}; name=${entry#*:}
  [ -f data/gutenberg_memory2/$name.txt ] && continue
  curl -sfL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o data/gutenberg_memory2/$name.raw
  title=$(grep -m1 '^Title:' data/gutenberg_memory2/$name.raw | tr -d '\r')
  sed -e '1,/^\*\*\* START OF/d' -e '/^\*\*\* END OF/,$d' data/gutenberg_memory2/$name.raw > data/gutenberg_memory2/$name.txt
  rm data/gutenberg_memory2/$name.raw
  echo "$name: $(wc -c < data/gutenberg_memory2/$name.txt) bytes  [$title]"
done
