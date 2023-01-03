
#!/bin/bash

none="_None_"

### usage ###
# . train.sh $weight_decay $lr $dropout $opt $random_seed $dataset_name $train_pct

#filename=train.sh 
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

### Main parameters ###
weight_decay=${1-0.0}
lr=${2-0.001}
dropout=${3-0.0}
opt=${4-adam}
random_seed=${5-0}

### Data parameters
dataset_name=${6-mnist}
train_pct=${7-100}
max_epochs=${8-10}

## Other parameters
log_dir="../log_files"

lr_scheduler=$none
#lr_scheduler=reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss

### wandb ###
use_wandb=False
group_name="wd=${weight_decay}-lr=${lr}-d=${dropout}-opt=${opt}"
wandb_entity="ift6512"
wandb_project="dataset=${dataset_name}"

exp_id="${dataset_name}"

#val_metric=val_acc
val_metric=val_loss

### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` ###
#early_stopping_grokking=$none
early_stopping_grokking="patience=int(1000),metric=str(${val_metric}),metric_threshold=float(90.0)"

if [[ $opt == "adam" ]]; then
  opt="${opt},weight_decay=${weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001"
elif [[ $opt == "sag" ]]; then
	opt="sag,weight_decay=${weight_decay},batch_mode=False,init_y_i=False"
elif [[ $opt == "sgd" ]]; then
	opt="sgd,weight_decay=${weight_decay}"
else 
	exit
fi

python train.py \
	--exp_id $exp_id \
	--log_dir "${log_dir}" \
	--c_out 10,10 \
	--hidden_dim 50 \
	--kernel_size 5 \
	--kernel_size_maxPool 2 \
	--dropout $dropout \
	--dataset_name $dataset_name \
	--train_batch_size 512 \
	--val_batch_size 512 \
	--train_pct $train_pct \
	--val_pct 100 \
	--optimizer $opt \
	--lr $lr \
	--lr_scheduler $lr_scheduler \
	--max_epochs $max_epochs \
	--validation_metrics $val_metric \
	--checkpoint_path $none \
	--every_n_epochs 100 \
	--save_top_k -1 \
	--use_wandb $use_wandb \
	--wandb_entity $wandb_entity \
	--wandb_project $wandb_project \
	--group_name $group_name \
	--group_vars $none \
	--accelerator auto \
	--devices auto \
	--random_seed $random_seed \
	--early_stopping_grokking $early_stopping_grokking \
#	--early_stopping_patience 1000000000 \
#	--model_name epoch=88-val_loss=13.6392.ckpt \