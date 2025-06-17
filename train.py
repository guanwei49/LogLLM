import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from model import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler
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



def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    normal_tokens = model.Llama_tokenizer('The sequence is normal.')['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.')['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f'scheduler_step: {scheduler_step}')

    steps = 0
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        pbar = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            inputs= bathc_i['inputs']
            seq_positions= bathc_i['seq_positions']
            labels = bathc_i['labels']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs, targets = model.train_helper(inputs, seq_positions, labels)

            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()
            # print(loss)

            if ((i_th + 1) % gradient_accumulation_steps == 0) or ((i_th + 1) == len(dataloader)):
                # optimizer the net
                optimizer.step()  # 更新网络参数
                optimizer.zero_grad()  # reset grdient # 清空过往梯度

            acc_mask = torch.zeros_like(targets,device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss = loss.item() * gradient_accumulation_steps)

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
    dataset = CustomDataset(data_path, drop_duplicates=False)

    model = LogLLM(Bert_path, Llama_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)
    # model = LogLLM(Bert_path, Llama_path, ft_path= ft_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)

    dataloader_max_samples = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion, max_samples=1000),
        collate_fn=collator,
        drop_last=True
    )
    # phase 1
    print("*" * 10 + "Start training Llama" + "*" * 10)
    model.set_train_only_Llama()
    trainModel(model, dataloader_max_samples, gradient_accumulation_steps, n_epochs_1, lr_1)
    del dataloader_max_samples

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion),
        collate_fn=collator,
        drop_last=True
    )
    # phase 2-1
    print("*" * 10 + "Start training projector" + "*" * 10)
    model.set_train_only_projector()
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_1, lr_2_1)
    # phase 2-2
    print("*" * 10 + "Start training projector and Bert" + "*" * 10)
    model.set_train_projectorAndBert()
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_2, lr_2_2)
    # phase 3
    model.set_finetuning_all()
    print("*" * 10 + "Start training entire model" + "*" * 10)
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_3, lr_3)

    model.save_ft_model(ft_path)