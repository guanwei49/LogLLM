import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
batch_size = 32
dataset_name = 'Liberty'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty‘
data_path = r'/mnt/public/gw/SyslogData/{}/test.csv'.format(dataset_name)

Bert_path = r"/mnt/public/gw/LLM_model/bert-base-uncased"
Llama_path = r"/mnt/public/gw/LLM_model/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0")

print(
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'device: {device}')


def evalModel(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                match = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if match:
                    preds.append(match.group())
                else:
                    print(f'error :{text}')
                    preds.append('')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataloader.dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    print(
        f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    print(f'precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}')


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    evalModel(model, dataloader)