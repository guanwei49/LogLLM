import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomDataset
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


def evalModel(model, dataset, batch_size):
    model.eval()
    pre = 0

    preds = []

    with torch.no_grad():
        indexes = [i for i in range(len(dataset))]
        for bathc_i in tqdm(range(batch_size, len(indexes) + batch_size, batch_size)):
            if bathc_i <= len(indexes):
                this_batch_indexes = list(range(pre, bathc_i))
            else:
                this_batch_indexes = list(range(pre, len(indexes)))
            pre = bathc_i

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes)
            outputs_ids = model(this_batch_seqs)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                matches = re.findall(r' (.*?)\.<|end_of_text|>', text)
                if len(matches) > 0:
                    preds.append(matches[0])
                else:
                    preds.append('')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataset.get_label()

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
    evalModel(model, dataset, batch_size)