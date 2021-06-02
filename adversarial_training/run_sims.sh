#!/bin/bash
python experiments.py --config minmax_10_1_normal_sgd_100 --seed 1 --gpu 0 &
python experiments.py --config minmax_10_1_no_higher_sgd_100 --seed 1 --gpu 0 &
python experiments.py --config minmax_10_1_higher_sgd_100 --seed 1 --gpu 0 &

python experiments.py --config minmax_10_1_normal_sgd_100 --seed 2 --gpu 0 &
python experiments.py --config minmax_10_1_no_higher_sgd_100 --seed 2 --gpu 0 &
python experiments.py --config minmax_10_1_higher_sgd_100 --seed 2 --gpu 0 &

python experiments.py --config minmax_10_1_normal_sgd_100 --seed 3 --gpu 0 &
python experiments.py --config minmax_10_1_no_higher_sgd_100 --seed 3 --gpu 0 &
python experiments.py --config minmax_10_1_higher_sgd_100 --seed 3 --gpu 0 &

python experiments.py --config minmax_10_1_normal_sgd_100 --seed 4 --gpu 1 &
python experiments.py --config minmax_10_1_no_higher_sgd_100 --seed 4 --gpu 1 &
python experiments.py --config minmax_10_1_higher_sgd_100 --seed 4 --gpu 1 &

python experiments.py --config minmax_10_1_normal_sgd_100 --seed 5 --gpu 1 &
python experiments.py --config minmax_10_1_no_higher_sgd_100 --seed 5 --gpu 1 &
python experiments.py --config minmax_10_1_higher_sgd_100 --seed 5 --gpu 1 &