import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class TaskModule(nn.Module):
    """
    TaskModule: a deterministic head for the downstream task
    """

    def __init__(self, hidden_size, dropout=0.2, output_classes=2):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_classes))

    def forward(self, x):
        output = self.output_layer(x)
        return output


class InfoPLM(nn.Module):
    """
    [ACL'24 Main] Representation Learning with Conditional Information Flow Maximization

    CIFM is an information-theoretic representation learning framework that encompasses two principles: Information Flow Maximization (IFM) and Conditional Information Maximization (CIM).
    In the implementation, maximizing I(X;Z) in IFM can be optimized using mutual information estimators (e.g., InfoNCE and MINE), while minimizing I(X;Z_{\delta}|Y) in CIM can be optimized through gradient-based adversarial training (e.g., FGM).
    For downstream tasks, we leverage pre-trained language models, such as BERT and RoBERTa, as backbone models for fine-tuning.
    """

    def __init__(self, pretrained_model_path="bert-base-chinese", pooling_method="cls",
                 max_length=128, dropout=0.2, infonce_weight=0.0, infonce_temperature=0.1, mine_weight=0.0, mine_mar_weight=1, mine_latent_dim=64,
                 tasks_config=None, output_hidden_states=False, task_type="cls", module_print_flag=False, tokenizer_add_e_flag=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        if tokenizer_add_e_flag:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ["<e>", "<e/>"]})
            self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.tasks_config = tasks_config
        self.pooling_method = pooling_method
        self.output_hidden_states = output_hidden_states
        self.infonce_weight = infonce_weight
        self.mine_weight = mine_weight
        self.mine_mar_weight = mine_mar_weight
        self.task_type = task_type

        if self.infonce_weight > 0:
            from mi_estimator import InfoNCE
            self.infonce_loss = InfoNCE(temperature=infonce_temperature, reduction='mean', negative_mode='unpaired')

        if self.mine_weight > 0:
            from mi_estimator import MINE_Net
            self.mine_nets = nn.ModuleDict(
                dict((k.value, MINE_Net(input_size=self.hidden_size, output_size=v["num_classes"], hidden_size=mine_latent_dim)) for k, v in tasks_config.items())
            )

        self.task_module = nn.ModuleDict(
            dict((k.value,
                  TaskModule(hidden_size=self.hidden_size, dropout=dropout, output_classes=v["num_classes"]))
                 for k, v in tasks_config.items())
        )

        if task_type == "cls":
            # For single- or multi-task setting for classification
            self.task_criterion = CrossEntropyLoss(weight=None, reduction='mean')
        elif task_type == "res":
            # For single- or multi-task setting for regression
            self.task_criterion = MSELoss(size_average=True)
        elif task_type == "multi":
            # For cross-type multi-task setting
            self.task_criterion = {
                "cls": CrossEntropyLoss(weight=None, reduction='mean'),
                "res": MSELoss(size_average=True)
            }

        if module_print_flag: print(self)

    def forward(self, x, label, task):
        tokenized_input = self.tokenizer(text=x, text_pair=None, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        for name, data in tokenized_input.items():
            tokenized_input[name] = tokenized_input[name].to(device)
        tokenized_input["output_hidden_states"] = self.output_hidden_states

        outputs = self.bert(**tokenized_input)

        if self.pooling_method == "cls":
            hidden = outputs.last_hidden_state[:, 0, :]
        else:
            hidden = None
            print("pooling_method error !")
            exit(0)

        if task in self.tasks_config.keys():

            pred = self.task_module[task.value](F.relu(hidden))
            task_criterion = self.task_criterion if self.task_type != "multi" else self.task_criterion[self.tasks_config[task]['task_type']]
            if type(label) == list:
                loss = None
                for i in range(len(label)):
                    if i == 0:
                        loss = task_criterion(pred[:, i], label[i])
                    else:
                        loss += task_criterion(pred[:, i], label[i])
                if len(label) > 1: loss = loss / len(label)
            else:
                loss = task_criterion(pred, label)

            if self.infonce_weight > 0:
                query = hidden
                positive_key = self.bert(**tokenized_input).last_hidden_state[:, 0, :]
                infonce_loss = self.infonce_loss(query, positive_key)

                print("w: {}, ce_loss: {}, infonce_loss: {}".format(self.infonce_weight, loss.item(), infonce_loss.item()))
                loss += infonce_loss * self.infonce_weight

            if self.mine_weight > 0:
                x_sample = hidden
                y_sample = F.one_hot(label, num_classes=self.tasks_config[task]['num_classes']).float()
                y_shuffle = y_sample[torch.randperm(y_sample.shape[0])]

                joint = self.mine_nets[task.value](x_sample, y_sample)
                marginal = torch.exp(self.mine_nets[task.value](x_sample, y_shuffle))
                mine_loss = torch.mean(joint) - self.mine_mar_weight * torch.log(torch.mean(marginal))

                print("w: {}, ce_loss: {}, mine_loss: {}".format(self.mine_weight, loss.item(), mine_loss.item()))
                loss -= mine_loss * self.mine_weight

            return pred, loss, hidden


        else:
            print("The task name {} is undefined in tasks_config!".format(task))
            exit(0)
