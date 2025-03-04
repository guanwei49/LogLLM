import os.path

import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

def merge_data(data):
    merged_data = []

    # 用于记录每个子列表开始的位置
    start_positions = []

    # 当前起始位置
    current_position = 0

    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions

def stack_and_pad_right(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat([torch.ones(tensor.shape[0], dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks

def stack_and_pad_left(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                 torch.ones(tensor.shape[0], dtype=torch.long)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=False,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
)

class LogLLM(nn.Module):
    def __init__(self, Bert_path, Llama_path, ft_path=None, is_train_mode=True, device = torch.device("cuda:0"), max_content_len = 128, max_seq_len = 128):
        super().__init__()
        self.max_content_len = max_content_len  # max length of each log messages (contents)
        self.max_seq_len = max_seq_len   # max length of each log sequence  (log sequence contains some log messages)
        self.device = device
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        self.Llama_model = AutoModelForCausalLM.from_pretrained(Llama_path, quantization_config=bnb_config,
                                                           low_cpu_mem_usage=True,
                                                           device_map=device)  # embedding dim = 4096

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)
        self.Bert_model = BertModel.from_pretrained(Bert_path, quantization_config=bnb_config, low_cpu_mem_usage=True,
                                               device_map=device)

        self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size, device=device)
        # self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size).half().to(device)

        self.instruc_tokens = self.Llama_tokenizer(
            ['Below is a sequence of system log messages:', '. Is this sequence normal or anomalous? \\n'],
            return_tensors="pt", padding=True).to(self.device)

        # if is_train_mode:
        #     self.Bert_model = prepare_model_for_kbit_training(self.Bert_model)
        #     self.Llama_model = prepare_model_for_kbit_training(self.Llama_model)

        if ft_path is not None:
            print(f'Loading peft model from {ft_path}.')
            Llama_ft_path = os.path.join(ft_path, 'Llama_ft')
            Bert_ft_path = os.path.join(ft_path, 'Bert_ft')
            projector_path = os.path.join(ft_path, 'projector.pt')
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.Bert_model = PeftModel.from_pretrained(
                self.Bert_model,
                Bert_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
        else:
            print(f'Creating peft model.')
            Bert_peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                          r=4,
                                          lora_alpha=32,
                                          lora_dropout=0.01)
            self.Bert_model = get_peft_model(self.Bert_model, Bert_peft_config)

            Llama_peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)

    def save_ft_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        Llama_ft_path = os.path.join(path,'Llama_ft')
        Bert_ft_path = os.path.join(path,'Bert_ft')
        projector_path = os.path.join(path,'projector.pt')
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization = True)
        self.Bert_model.save_pretrained(Bert_ft_path, safe_serialization =True)
        torch.save(self.projector.state_dict(), projector_path)


    def set_train_only_projector(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_train_only_Llama(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = False
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def set_train_projectorAndBert(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False


    def set_finetuning_all(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True


    def train_helper(self, sequences_, labels):
        '''
        :param sequences: list of list: [seq, seq, ...,seq]  , seq:[item, ..., item]
        :param labels:  list of labels, label is one of ['anomalous', 'normal']
        :return: Llama_output[label_mask], target_tokens_ids[target_tokens_atts]
        '''

        sequences = [sequence[:self.max_seq_len] for sequence in sequences_]

        batch_size = len(sequences)
        data, seq_positions = merge_data(sequences)
        seq_positions = seq_positions[1:]

        inputs = self.Bert_tokenizer(data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)

        outputs = self.Bert_model(**inputs).pooler_output  # dim = 768
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()

        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels = np.char.add(np.char.add(prefix, labels.astype(f'U{max_len}')), ".")
        answer_tokens = self.Llama_tokenizer(list(labels), padding=True, return_tensors="pt").to(self.device)

        target_tokens_ids = torch.cat([answer_tokens['input_ids'][:, 1:],
                                       torch.full((batch_size, 1), self.Llama_tokenizer.eos_token_id, device=self.device)],
                                      dim=-1)  # add eos token
        target_tokens_atts = answer_tokens['attention_mask'].bool()

        answer_tokens_ids = answer_tokens['input_ids'][:, 1:]  # remove bos token
        answer_tokens_atts = answer_tokens['attention_mask'].bool()[:, 1:]

        if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = self.Llama_model.model.model.embed_tokens(answer_tokens_ids)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = self.Llama_model.model.embed_tokens(answer_tokens_ids)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        embeddings = []
        target_lens = []
        for seq_embedding, answer_embedding, answer_tokens_att in zip(seq_embeddings, answer_embeddings,
                                                                      answer_tokens_atts):
            full_prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_embedding[answer_tokens_att]])
            target_lens.append(answer_tokens_att.sum())
            embeddings.append(full_prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        attention_mask = attention_mask.to(self.device)
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, :-target_lens[i]-1] = 0
        label_mask = label_mask.bool()

        Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        return Llama_output[label_mask], target_tokens_ids[target_tokens_atts]

    def forward(self, sequences_):
        '''
        :param sequences: list of list: [seq, seq, ...,seq]  , seq:[item, ..., item]
        :return: Generated answer (token id).
        '''

        sequences = [sequence[:self.max_seq_len] for sequence in sequences_]

        batch_size = len(sequences)
        data, seq_positions = merge_data(sequences)
        seq_positions = seq_positions[1:]

        inputs = self.Bert_tokenizer(data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)

        outputs = self.Bert_model(**inputs).pooler_output  # dim = 768
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()

        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        prefix = "The sequence is"
        answer_prefix_tokens = self.Llama_tokenizer(prefix, padding=True, return_tensors="pt")['input_ids'][0,1:].to(
            self.device)

        if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.model.embed_tokens(answer_prefix_tokens)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.embed_tokens(answer_prefix_tokens)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]



        promot_embeddings = []
        for seq_embedding in seq_embeddings:
            prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_prefix_tokens_embeddings])
            promot_embeddings.append(prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(promot_embeddings)
        attention_mask = attention_mask.to(self.device)

        pad_token_id = self.Llama_tokenizer.pad_token_id
        eos_token_id = self.Llama_tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.device) if eos_token_id is not None else None

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        this_peer_finished = False
        answer = []
        while not this_peer_finished:
            Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            next_token_logits = Llama_output[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # print(next_tokens)
            answer.append(next_tokens)

            if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
                next_tokens_embeddings = self.Llama_model.model.model.embed_tokens(next_tokens)
            else:
                next_tokens_embeddings = self.Llama_model.model.embed_tokens(next_tokens)

            inputs_embeds = torch.cat([inputs_embeds, next_tokens_embeddings[:,None,:]], dim=1)
            attention_mask = torch.cat([attention_mask, unfinished_sequences[:,None]], dim=1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

                # stop if we exceed the maximum answer length
            if  5 < len(answer):
                this_peer_finished = True

        return torch.stack(answer,dim=1)
