#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/CIFM"
MODEL_BASE="/CIFM/ptms/roberta-base"

model_name="cifm"
VV="a100-2401-st-${model_name}"
SEED="0 1 2 3 4"
DATA_PER="1.0"
LR="5e-5"
WARMUP_RATIO=0.1
BS=128
EPOCH_NUM=20
PASTIENT_NUM=5
MAX_LEN=128


# ==============================================================================
ALL_TASK_NAME="claire_v2"
task_type="res"
NCE_W="0.01"
NCE_T="1"
AT_RATE="0.1"
AT_EPISION="0.1"

L2="0"
DP="0"

for task_name in ${ALL_TASK_NAME[@]}
do
for l2 in ${L2[@]}
do
for dp in ${DP[@]}
do
for nce_weight in ${NCE_W[@]}
do
for nce_t in ${NCE_T[@]}
do
for at_rate in ${AT_RATE[@]}
do
for at_epsilon in ${AT_EPISION[@]}
do
for seed in ${SEED[@]}
do
EXP_NO="${VV}_${task_name}_nce-w${nce_weight}-t${nce_t}_adv-e${at_epsilon}-r${at_rate}_d${DATA_PER}_lr${LR}_l2${l2}_WA${WARMUP_RATIO}_bs${BS}_dp${dp}_len${MAX_LEN}_s${seed}"
OUT_DIR="${WORK_DIR}/outputs/${model_name}/${task_name}/${task_name}"
LOG_PATH="${WORK_DIR}/logs/${model_name}/${task_name}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi
echo "VV: ${VV}"
echo "TASK_NAME: ${task_name}"
echo "EXP_NO: ${EXP_NO}"
echo "OUT_DIR: ${OUT_DIR}"
echo "LOG_DIR: ${LOG_PATH}/${EXP_NO}.out"

python -u ${WORK_DIR}/main.py   \
--epochs    ${EPOCH_NUM} \
--patience  ${PASTIENT_NUM} \
--seed      ${seed} \
--fine_tune_task        ${task_name} \
--task_type             ${task_type} \
--dataset_percentage    ${DATA_PER} \
--lr        ${LR}   \
--weight_decay          ${l2}   \
--warmup_ratio  ${WARMUP_RATIO} \
--bs        ${BS}   \
--dropout   ${dp}   \
--max_length    ${MAX_LEN} \
--pretrained_model_path     ${MODEL_BASE} \
--output_dir    ${OUT_DIR} \
--infonce_weight    ${nce_weight} \
--infonce_t     ${nce_t} \
--at_epsilon    ${at_epsilon} \
--at_rate       ${at_rate} \
--batch_sampling_flag \
--tokenizer_add_e_flag \
>> ${LOG_PATH}/${EXP_NO}.out
done
done
done
done
done
done
done
done