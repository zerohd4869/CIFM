# CIFM
This repository contains the official code for ACL 2024 paper "[Representation Learning with Conditional Information Flow Maximization](https://aclanthology.org/2024.acl-long.759/)".

## Highlights
This paper introduces an information-theoretic representation learning framework CIFM to extract noise-invariant sufficient representations for the input data and target task. 
It promotes the learned representations have good feature uniformity and sufficient predictive ability, which can enhance the generalization of pre-trained language models (PLMs) for the target task.

Experiments on 13 language understanding benchmarks demonstrate that CIFM achieves better performance for classification and regression, and the learned representations are more sufficient, robust and transferable.

## News
- **[Sep 2024]**: Added support for the multi-task version of CIFM.
- **[Aug 2024]**: Code is available on [GitHub](https://github.com/zerohd4869/CIFM).
- **[Jun 2024]**: Paper is available on [arXiv](https://arxiv.org/abs/2406.05510).
- **[May 2024]**: Paper is accepted by ACL 2024 (main conference).

## Quick Start

1. Clone the repository
```
git clone https://github.com/zerohd4869/CIFM.git
cd ./CIFM
```

2. Download the data and pre-trained model parameters

Download 13 datasets mentioned in the paper from [here](https://drive.google.com/file/d/17Xin3AKF-BCxn1HzKAI4NaOZNYaVesG-/view?usp=sharing), and extract the files into the `/CIFM/data/` directory.
This repo already contains 7 of these datasets by default, so this step is optional.

Download the `roberta-base` model parameters from [here](https://huggingface.co/FacebookAI/roberta-base) and place them in the `/CIFM/ptms/roberta-base/` directory.
CIFM is a backbone-free representation learning method. When using it, you could choose an appropriate backbone model and initialized parameter checkpoints for your task or dataset.

2. Install dependencies
``` 
# env: Python 3.7.16, Tesla A100 80GB
pip install -r cifm_requirements.txt
```

3. Run examples

For classification:
```
# EmojiEval dataset
nohup bash script/run_train_emojieval.sh >  cifm_roberta_emojieval.out &

# EmotionEval dataset
nohup bash script/run_train_emotioneval.sh >  cifm_roberta_emotioneval.out &

# HatEval dataset
nohup bash script/run_train_hateval.sh >  cifm_roberta_hateval.out &

# IronyEval dataset
nohup bash script/run_train_ironyeval.sh >  cifm_roberta_ironyeval.out &

# OffensEval dataset
nohup bash script/run_train_offenseval.sh >  cifm_roberta_offenseval.out &

# SentiEval dataset
nohup bash script/run_train_sentieval.sh >  cifm_roberta_sentieval.out &

# StanceEval dataset
nohup bash script/run_train_stanceeval.sh >  cifm_roberta_stanceeval.out &

# ISEAR dataset
nohup bash script/run_train_isear.sh >  cifm_roberta_isear.out &

# MELD dataset
nohup bash script/run_train_meld.sh >  cifm_roberta_meld.out &

# GoEmotions dataset
nohup bash script/run_train_goemotions.sh >  cifm_roberta_goemotions.out &
```

For regression:
```
# STS-B dataset
nohup bash script/run_train_sbsb.sh >  cifm_roberta_stsb.out &

# CLAIRE dataset
nohup bash script/run_train_claire.sh >  cifm_roberta_claire.out &

# EmoBank dataset
nohup bash script/run_train_emobank.sh >  cifm_roberta_emobank.out &
```

## Additional Recipes

**Apply for a new task/dataset**

1. Data preparation and loading script. Download the new dataset (take `NewDataset` as an example) and place the unzip files in the `/CIFM/data/` directory. Add the label information of this dataset to the dictionary file `CIFM/data/task2label.json`.
Then, refer to the template `/CIFM/datasets/new_dataset_script.py` to write the corresponding reading script for the dataset and place the file in the `/CIFM/datasets/` directory. Also, add the dataset and task information to the file `CIFM/task.py` at the corresponding location.

2. Refer to the Quick Start section above to write the corresponding sh script and run it.

During the training process for CIFM, the primary hyperparameters for adjustment along with their suggested ranges are as follows:
```
# the default InfoNCE estimator used in IFM
infonce_weight (beta): [0.01, 0.1, 1, 10] for classification, and [0.001, 0.01, 0.1] for regression
infonce_t (tau): [0.1, 0.5, 1]

# the alternative MINE estimator used in IFM
mine_weight (beta): [0.01, 0.1, 1]
mine_mar_weight: [1, 2]

# the adversarial estimator used in CIM
at_epsilon (epsilon): [0.1, 1, 5] 
at_rate: [0.1, 1]

# others
weight_decay: [0, 0.001, 0.01] for classification, and [0] for regression
dropout: [0, 0.2]
batch_sampling_flag: False, True
```
Other hyperparameters can be adjusted based on experimental conditions and specific task requirements, such as epochs, patience, warmup_ratio, bs, dropout, max_length, etc.


**Apply all tasks in a multi-task paradigm**

```
# 6 tasks/datasets in TweetEval
nohup bash script/run_train_mtl_tweeteval.sh >  cifm_roberta_mtl_tweeteval.out &
```

## Citation

If you are interested in this work and want to use the code in this repo, please **star** this repo and **cite** it as:

```
@inproceedings{DBLP:conf/acl/0001WZH24,
  author       = {Dou Hu and
                  Lingwei Wei and
                  Wei Zhou and
                  Songlin Hu},
  title        = {Representation Learning with Conditional Information Flow Maximization},
  booktitle    = {{ACL} {(1)}},
  pages        = {14088--14103},
  publisher    = {Association for Computational Linguistics},
  year         = {2024}
}
```