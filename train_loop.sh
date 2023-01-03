#!/bin/bash

# Usage : ./train_loop.sh $dataset_name $train_pct $max_epochs

ds_name=${1-mnist}
t_pct=${2-100}
m_epochs=${3-10}

#all_optims=("sgd" "nesterov" "asgd" "rmsprop" "rprop" "adadelta" "adagrad" "adam" "adamax" "custom_adam" "adam_inverse_sqrt" "adam_cosine" "sag")
all_optims=("sgd" "nesterov" "asgd" "rmsprop" "rprop" "adadelta" "adagrad" "adam" "adamax" "sag")

for weight_decay in 0.0; do {
for lr in 0.001; do {
for dropout in 0.0; do {
for opttmp in ${all_optims[@]}; do {
for random_seed in 0; do {
. train.sh $weight_decay $lr $dropout $opttmp $random_seed $ds_name $t_pct $m_epochs
} done
} done
} done
} done
} done