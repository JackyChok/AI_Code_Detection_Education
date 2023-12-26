import os
import csv
import pickle
from typing import Callable, List, Tuple

import gradio as gr
from nltk.data import load as nltk_load
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GLTRPPLCodeDetector:
    def __init__(self):
        self.AUTH_TOKEN = os.environ.get("access_token")
        self.DET_LING_ID = 'Hello-SimpleAI/chatgpt-detector-ling'
        self.NLTK = nltk_load('gltr_ppl/english.pickle')
        self.sent_cut_en = self.NLTK.tokenize
        self.LR_GLTR_EN, self.LR_PPL_EN = [
            pickle.load(open(f'gltr_ppl/{lang}-gpt2-{name}.pkl', 'rb'), encoding='utf-8')
            for lang, name in [('en', 'gltr'), ('en', 'ppl')]
        ]

        self.NAME_EN = 'gpt2'
        self.TOKENIZER_EN = GPT2Tokenizer.from_pretrained(self.NAME_EN)
        self.MODEL_EN = GPT2LMHeadModel.from_pretrained(self.NAME_EN)

        self.CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')
    
    # document here is the python code to be evaluated
    def text_predict_tuple(self, document: str):
        with torch.no_grad():
            feat = self.gpt2_features(document, self.TOKENIZER_EN, self.MODEL_EN, self.sent_cut_en)
        pred_human_gltr, prob_human_gltr, pred_human_ppl, prob_human_ppl = self.lr_predict(*feat, self.LR_GLTR_EN, self.LR_PPL_EN, ['1', '0'])
        return pred_human_gltr, pred_human_ppl
        # !!! order of the return result must match the order of the column header mentioned below
        # return <result, raw> result should be 1 if human, 0 if AI, raw should be full data if it exist
        

    def gpt2_features(
        self, text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, sent_cut: Callable
    ) -> Tuple[List[int], List[float]]:
        # Tokenize
        input_max_length = tokenizer.model_max_length - 2
        token_ids, offsets = list(), list()
        sentences = sent_cut(text)
        for s in sentences:
            tokens = tokenizer.tokenize(s)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            difference = len(token_ids) + len(ids) - input_max_length
            if difference > 0:
                ids = ids[:-difference]
            offsets.append((len(token_ids), len(token_ids) + len(ids)))  # 左开右闭
            token_ids.extend(ids)
            if difference >= 0:
                break

        input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids)
        logits = model(input_ids).logits
        # Shift so that n-1 predict n
        shift_logits = logits[:-1].contiguous()
        shift_target = input_ids[1:].contiguous()
        loss = self.CROSS_ENTROPY(shift_logits, shift_target)

        all_probs = torch.softmax(shift_logits, dim=-1)
        sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
        expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
        indices = torch.where(sorted_ids == expanded_tokens)
        rank = indices[-1]
        counter = [
            rank < 10,
            (rank >= 10) & (rank < 100),
            (rank >= 100) & (rank < 1000),
            rank >= 1000
        ]
        counter = [c.long().sum(-1).item() for c in counter]


        # compute different-level ppl
        text_ppl = loss.mean().exp().item()
        sent_ppl = list()
        for start, end in offsets:
            nll = loss[start: end].sum() / (end - start)
            sent_ppl.append(nll.exp().item())
        max_sent_ppl = max(sent_ppl)
        sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
        if len(sent_ppl) > 1:
            sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
        else:
            sent_ppl_std = 0

        mask = torch.tensor([1] * loss.size(0))
        step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
        max_step_ppl = step_ppl.max(dim=-1)[0].item()
        step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
        if step_ppl.size(0) > 1:
            step_ppl_std = step_ppl.std().item()
        else:
            step_ppl_std = 0
        ppls = [
            text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
            max_step_ppl, step_ppl_avg, step_ppl_std
        ]
        return counter, ppls  # type: ignore


    def lr_predict(
        self, f_gltr: List[int], f_ppl: List[float], lr_gltr: LogisticRegression, lr_ppl: LogisticRegression,
        id_to_label: List[str]
    ) -> List:
        x_gltr = np.asarray([f_gltr])
        gltr_label = lr_gltr.predict(x_gltr)[0]
        gltr_prob = lr_gltr.predict_proba(x_gltr)[0, gltr_label]
        x_ppl = np.asarray([f_ppl])
        ppl_label = lr_ppl.predict(x_ppl)[0]
        ppl_prob = lr_ppl.predict_proba(x_ppl)[0, ppl_label]
        return [id_to_label[gltr_label], gltr_prob, id_to_label[ppl_label], ppl_prob]


# def predict_en(text: str) -> List:
#     with torch.no_grad():
#         feat = gpt2_features(text, TOKENIZER_EN, MODEL_EN, sent_cut_en)
#     out = lr_predict(*feat, LR_GLTR_EN, LR_PPL_EN, ['Human', 'ChatGPT'])
#     return out

    def predict_en(self, csv_file: str, row_start: int, row_end: int, column_name: str, output_file: str) -> None:
        with open(csv_file, 'r', newline='', encoding='ISO-8859-1') as f:
            reader = csv.DictReader(f)
            rows = [row for i, row in enumerate(reader) if row_start <= i < row_end]
        count = 0

        for row in rows:
            text = row[column_name]
            count += 1
            if text == "BLANK":
                continue
            with torch.no_grad():
                feat = self.gpt2_features(text, self.TOKENIZER_EN, self.MODEL_EN, self.sent_cut_en)
            pred_human_gltr, prob_human_gltr, pred_human_ppl, prob_human_ppl = self.lr_predict(*feat, self.LR_GLTR_EN, self.LR_PPL_EN, ['1', '0'])
            print("Done " + str(count))
            row['human_gltr_pred'] = pred_human_gltr
            row['human_gltr_prob'] = str(prob_human_gltr)
            row['human_ppl_pred'] = pred_human_ppl
            row['human_ppl_prob'] = str(prob_human_ppl)

        with open(output_file, 'w', newline='', encoding='ISO-8859-1') as f:
            fieldnames = reader.fieldnames + ['human_gltr_pred', 'human_gltr_prob', 'human_ppl_pred', 'human_ppl_prob']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

# predict_en('merged_file.csv', 0, 2, "GPTAnswer", "v4_gpt.csv")
 
