#!/bin/bash
# More memory-only text for tier_eval `memory_corpus=novels150` (docs/tiers_design.md): fifty-eight more Project Gutenberg books
# into data/gutenberg_memory3/ (gitignored), licence header and footer stripped. Public domain in the US.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/gutenberg_memory3
for entry in \
  46:a_christmas_carol_in \
  968:martin_chuzzlewit \
  882:sketches_by_boz_illustrative \
  3409:barchester_towers \
  619:warden \
  6688:mill_on_the \
  7469:daniel_deronda \
  107:far_from_the_madding \
  122:return_of_the \
  482:woodlanders \
  6053:evelina_or_the_history \
  1079:life_and_opinions \
  5998:waverley_or_tis_sixty \
  7025:rob_roy_complete \
  1900:typee_a_romance_of \
  2833:portrait_of_a \
  209:turn_of_the \
  208:daisy_miller_a_study \
  432:ambassadors \
  284:house_of_mirth \
  541:age_of_innocence \
  4517:ethan_frome \
  24:o_pioneers \
  242:my_ntonia \
  233:sister_carrie_a_novel \
  140:jungle \
  1074:seawolf \
  1056:martin_eden \
  73:red_badge_of \
  102:tragedy_of_puddnhead \
  245:life_on_the_mississippi \
  3177:roughing_it \
  848:black_arrow_a \
  589:catriona \
  864:master_of_ballantrae \
  1142:typhoon \
  2480:under_western_eyes \
  718:tonobungay \
  2641:a_room_with_a \
  2891:howards_end \
  4217:a_portrait_of_the \
  1695:man_who_was \
  204:innocence_of_father \
  139:lost_world \
  834:memoirs_of_sherlock \
  903:white_company \
  711:allan_quatermain \
  558:thirtynine_steps \
  1259:twenty_years_after \
  2610:notredame_de_paris \
  1268:mysterious_island \
  30723:fathers_and_children \
  600:notes_from_the_underground \
  1081:dead_souls \
  996:don_quixote \
  19942:candide \
  2413:madame_bovary \
  1237:father_goriot \
  ; do
  id=${entry%%:*}; name=${entry#*:}
  [ -f data/gutenberg_memory3/$name.txt ] && continue
  curl -sfL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o data/gutenberg_memory3/$name.raw
  title=$(grep -m1 '^Title:' data/gutenberg_memory3/$name.raw | tr -d '\r')
  sed -e '1,/^\*\*\* START OF/d' -e '/^\*\*\* END OF/,$d' data/gutenberg_memory3/$name.raw > data/gutenberg_memory3/$name.txt
  rm data/gutenberg_memory3/$name.raw
  echo "$name: $(wc -c < data/gutenberg_memory3/$name.txt) bytes  [$title]"
done
