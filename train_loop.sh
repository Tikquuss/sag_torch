#!/bin/bash

# Usage : ./train_loop.sh

dataset_name=mnist
train_pct=100

for weight_decay in 0.0; do {
for lr in 0.001; do {
for dropout in 0.0; do {
for opt in adam; do {
for random_seed in 0 100; do {
. train.sh $weight_decay $lr $dropout $opt $random_seed $dataset_name $train_pct
} done
} done
} done
} done
} done
