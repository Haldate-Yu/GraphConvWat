#!/bin/bash
#for idx in {1..3}
#do
#	for deploy in master dist hydrodist hds hdvar
#	do
#		python train.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds anytown --budget 1 --batch 200
#		python train.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds ctown --budget 5 --batch 120
#		python train.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds richmond --budget 10 --batch 50
#	done
#done
#for idx in {1..15}
#do
#	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds anytown --budget 1 --batch 200
#	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds ctown --budget 5 --batch 120
#	python train.py --epoch 1000 --adj binary --tag ms --deploy xrandom --deterministic --wds richmond --budget 10 --batch 50
#done

#for idx in {1..3}
#do
#	for deploy in master dist hydrodist hds hdvar
#	do
#		python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy $deploy --wds anytown --budget 1 --batch 200
#		python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy $deploy --wds ctown --budget 5 --batch 120
#		python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy $deploy --wds richmond --budget 10 --batch 50
#	done
#done
#for idx in {1..15}
#do
#	python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy xrandom --wds anytown --budget 1 --batch 200
#	python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy xrandom --wds ctown --budget 5 --batch 120
#	python test_Taylor_metrics_for_sensor_placement.py --runid $idx --adj binary --tag ms --deploy xrandom --wds richmond --budget 10 --batch 50
#done

for idx in {1..3}; do
  for deploy in master dist hydrodist hds hdvar; do
    for obs in 0.05 0.1 0.2; do
      python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds anytown --obsrat $obs --batch 200 --runid $idx
      python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds ctown --obsrat $obs --batch 120 --runid $idx
      python train_ratio.py --epoch 2000 --adj binary --tag ms --deploy $deploy --wds richmond --obsrat $obs --batch 50 --runid $idx
    done
  done
done
