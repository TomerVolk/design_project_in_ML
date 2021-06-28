import torch
from argparse import Namespace
import random
from dataset_utils import BaseDataset


class LMDataset(BaseDataset):

    def __init__(self, h_params: Namespace, file_path, base_dataset=None):
        self.sentences = []
        super(LMDataset, self).__init__(h_params, file_path, base_dataset)
        # self.sentences = self.sentences[:10]
        # print("Using Only Small DS!!!!!")
        self.ids, self.masks = self.preprocess(self.sentences, False)
        self.vocab_size = self.tokenizer.vocab_size + len(self.tokenizer.added_tokens_encoder)

    def read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for row in f:
                row = row.strip().replace("\n", "")
                self.sentences.append(row)

    def __getitem__(self, item):
        ids, masks = self.ids[item], self.masks[item]
        ids, masks = ids.clone(), masks.clone()
        ids = ids[0]
        masks = masks[0]
        labels = torch.zeros_like(ids, dtype=torch.long) - 100
        counter = 0
        for idx, word in enumerate(ids):
            # if int(ids[idx]) == self.tokenizer.pad_token_id:
            #     break
            # if int(ids[idx]) == self.tokenizer.cls_token_id or int(ids[idx]) == self.tokenizer.sep_token_id:
            #     continue
            p = random.random()
            if p < self.h_params.dropout_prob:
                labels[idx] = int(ids[idx])
                ids[idx] = self.tokenizer.mask_token_id
                # counter += 1
        return ids, masks, labels

    def __len__(self):
        return len(self.sentences)
