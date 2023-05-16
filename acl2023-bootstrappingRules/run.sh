#!/bin/bash

for iter in {0..5}
do
    # train
    python train.py --lr 1e-5 --pooling avg --batch_size 32 --num_epoch 20 --seed 43 --id $iter --device 0 --warmup_prop 0.3 --data_dir dataset/tacred --info "TACRED EC+RC" 
    # generate model output
    python eval.py saved_models/$iter --device 0 --dataset train
    # generate rules from training
    python collect_rules.py $iter
    python generate_rules.py $iter
    # excute rules on dev (for evluating the rules)
    sbt -J-Xmx4G "runMain shell /data/dev_1pc.json /grammars_$iter/master.yml dev_output_$iter.txt"
    # excute rules on train (for generating new taggings)
    sbt -J-Xmx4G "runMain shell /data/train.json /grammars_$iter/master.yml train_output_$iter.txt"
    #fi
    # eval rules
    python eval_per_rule.py dev_output_$iter.txt kept_rules_$iter.json
    # generate new taggings
    python create_tagging.py --model_output output_$iter.json --kept kept_rules_$iter.json --rule_output train_output_$iter.txt --rule --output tagging_train_$((iter+1)).txt
    # move the new tagging to the dataset folder
    mv tagging_train_$((iter+1)).txt dataset/tacred/
done
    


