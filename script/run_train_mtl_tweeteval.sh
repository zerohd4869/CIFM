#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/SPC"
MODEL_BASE="/SPC/ptms/roberta-base"

model_name="spc"
domain_name="tweet"
VV="a100-1203-mt-${domain_name}-${model_name}"

SEED="0 1 2"
DATA_PER="1.0"
LR="5e-5"
DP="0"
WARMUP_RATIO=0.1
BS=128
EPOCH_NUM=20
PASTIENT_NUM=3
MAX_LEN=256

# ==============================================================================
TASK_NAME="tweeteval_emotion tweeteval_hate tweeteval_irony tweeteval_offensive tweeteval_sentiment tweeteval_stance"
task_type="cls"
L2="0"
NCE_W="0.01"
NCE_T="0.1"
AT_RATE="0.1"
AT_EPISION="1"


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
EXP_NO="${VV}_${domain_name}_nce-w${nce_weight}-t${nce_t}_adv-e${at_epsilon}-r${at_rate}_d${DATA_PER}_lr${LR}_l2${l2}_WA${WARMUP_RATIO}_bs${BS}_dp${dp}_len${MAX_LEN}_s${seed}"
OUT_DIR="${WORK_DIR}/outputs/${model_name}/${domain_name}/${domain_name}"
LOG_PATH="${WORK_DIR}/logs/${model_name}/${domain_name}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi
echo "VV: ${VV}"
echo "TASK_NAME: ${TASK_NAME}"
echo "EXP_NO: ${EXP_NO}"
echo "OUT_DIR: ${OUT_DIR}"
echo "LOG_DIR: ${LOG_PATH}/${EXP_NO}.out"

python -u ${WORK_DIR}/main.py   \
--epochs    ${EPOCH_NUM} \
--patience  ${PASTIENT_NUM} \
--seed      ${seed} \
--fine_tune_task        ${TASK_NAME} \
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
--at_emb_names "word_embedding" \
--batch_sampling_flag \
>> ${LOG_PATH}/${EXP_NO}.out
done
done
done
done
done
done
done
done