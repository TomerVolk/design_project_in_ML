import torch
from argparse import Namespace
import random
from dataset_utils import BaseDataset


class LMDataset(BaseDataset):

    def __init__(self, h_params: Namespace, file_path):
        self.sentences = []
        super(LMDataset, self).__init__(h_params, file_path)
        self.ids, self.masks = self.preprocess(self.sentences, False)

    def read_file(self, file_path):
        with open(file_path, "r") as f:
            for row in f:
                row = row.strip().replace("\n", "")
                self.sentences.append(row)

    def __getitem__(self, item):
        ids, masks = self.ids[item], self.masks[item]
        ids = ids[0]
        masks = masks[0]
        labels = torch.zeros_like(ids, dtype=torch.long) - 100
        counter = 0
        for idx, word in enumerate(ids):
            if int(ids[idx]) == self.tokenizer.pad_token_id:
                break
            p = random.random()
            if p < self.h_params.dropout_prob:
                labels[counter] = int(ids[idx])
                ids[idx] = self.tokenizer.get_vocab()[f"<extra_id_{counter}>"]
                counter += 1
        return ids, masks, labels

    def __len__(self):
        return len(self.sentences)
