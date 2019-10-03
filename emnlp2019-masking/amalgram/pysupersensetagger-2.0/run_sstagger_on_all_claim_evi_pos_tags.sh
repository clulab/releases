#! /bin/bash
for f in $(ls ./input_to_sstagger_output_from_pos_tagger/); do
    $(./sst.sh ./input_to_sstagger_output_from_pos_tagger/$f)
done
