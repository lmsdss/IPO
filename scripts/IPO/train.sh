#!/bin/bash


# custom config
DATA=/home/data
TRAINER=CoOp # IPO

DATASET=$1
SEED=$2

CFG=vit_b16_ctxv1
SHOTS=1


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/IPO/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/IPO/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi

