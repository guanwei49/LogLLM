import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import random
from model import LogLLM
from customDataset import CustomDataset
from torch import optim


n_epochs_1 = 1
n_epochs_2_1 = 1
n_epochs_2_2 = 1
n_epochs_3 = 2
dataset_name = 'Liberty'  # 'Thunderbird' 'HDFS_v1' 'BGL'   'Liberty'
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size


lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
max_content_len = 100
max_seq_len = 128

data_path = r'/mnt/public/gw/SyslogData/{}/train.csv'.format(dataset_name)

min_less_portion = 0.3

Bert_path = r"/mnt/public/gw/LLM_model/bert-base-uncased"
Llama_path = r"/mnt/public/gw/LLM_model/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0")

print(f'n_epochs_1: {n_epochs_1}\n'
f'n_epochs_2_1: {n_epochs_2_1}\n'
f'n_epochs_2_2: {n_epochs_2_2}\n'
f'n_epochs_3: {n_epochs_3}\n'
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'micro_batch_size: {micro_batch_size}\n'
f'lr_1: {lr_1}\n'
f'lr_2_1: {lr_2_1}\n'
f'lr_2_2: {lr_2_2}\n'
f'lr_3: {lr_3}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'min_less_portion: {min_less_portion}\n'
f'device: {device}')

def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return params



def trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs, lr,num_samples=None):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    normal_tokens = model.Llama_tokenizer('The sequence is normal.')['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.')['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    indexes = [i for i in range(len(dataset))]
    if dataset.num_less/len(dataset) < min_less_portion:
        less_should_num = int((min_less_portion*dataset.num_majority) / (1 - min_less_portion))
        add_num =  less_should_num - dataset.num_less
        indexes = indexes + np.random.choice(dataset.less_indexes , add_num).tolist()

    if num_samples is None:
        total_steps = (len(indexes) * n_epochs) / micro_batch_size
    else:
        num_samples = min(num_samples, len(indexes))
        total_steps = (num_samples * n_epochs) / micro_batch_size
    scheduler_step = int(total_steps/11)  #update 10 times lr

    print(f'scheduler_step: {scheduler_step}')

    steps = 0
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        # 自定义的dataloader
        random.shuffle(indexes)   # 打乱顺序
        end = len(indexes) + 1

        if num_samples is not None:
            end = min(num_samples,end)

        pbar = tqdm(range(micro_batch_size, end, micro_batch_size), desc='Epoch {}/{}'.format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            this_batch_indexes = indexes[bathc_i - micro_batch_size: bathc_i]
            this_batch_seqs, this_batch_labels = dataset.get_batch(this_batch_indexes)

            outputs, targets = model.train_helper(this_batch_seqs, this_batch_labels)

            loss = criterion(outputs, targets)

            loss.backward()
            # print(loss)

            if ((i_th + 1) % gradient_accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()  # 更新网络参数
                optimizer.zero_grad()  # reset grdient # 清空过往梯度

            acc_mask = torch.zeros_like(targets,device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()

            train_loss += loss.item() * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(lr=scheduler.get_last_lr()[0])

            if steps % 10000 ==0:   # every 10000 steps, print loss and acc
                train_loss_epoch = train_loss / total_count
                train_acc_epoch = total_acc / total_acc_count
                print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                      f"[loss: {train_loss_epoch:3f}]"
                      f"[acc: {train_acc_epoch:3f}]")

                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count
            print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]"
                  f"[acc: {train_acc_epoch:3f}]")

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)

    model = LogLLM(Bert_path, Llama_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)
    # model = LogLLM(Bert_path, Llama_path, ft_path= ft_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)

    # phase 1
    print("*" * 10 + "Start training Llama" + "*" * 10)
    model.set_train_only_Llama()
    trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_1, lr_1, num_samples=1000)
    # phase 2-1
    print("*" * 10 + "Start training projector" + "*" * 10)
    model.set_train_only_projector()
    trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_2_1, lr_2_1)
    # phase 2-2
    print("*" * 10 + "Start training projector and Bert" + "*" * 10)
    model.set_train_projectorAndBert()
    trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_2_2, lr_2_2)
    # phase 3
    model.set_finetuning_all()
    print("*" * 10 + "Start training entire model" + "*" * 10)
    trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_3, lr_3)

    model.save_ft_model(ft_path)