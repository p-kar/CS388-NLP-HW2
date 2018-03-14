#!/usr/bin/env bash

DATASET_PATH=/scratch/cluster/pkar/CS388-NLP-HW2/pos/wsj
TRAIN_DIR=/scratch/cluster/pkar/CS388-NLP-HW2/tmp/pos_train_baseline_dir
SPLIT_TYPE=standard
MODE=train
# 0 - no orth. features, 1 - orth. features concatenated to the LSTM input,
# 2 - orth. features concatenated to the LSTM output
ORTH_FEAT_MODE=0

source /u/pkar/Documents/tensorflow_cpu_2.7/bin/activate
python -u pos_bilstm.py $DATASET_PATH $TRAIN_DIR $SPLIT_TYPE $MODE $ORTH_FEAT_MODE
