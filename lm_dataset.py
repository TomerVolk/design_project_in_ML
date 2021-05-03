from transformers import T5TokenizerFast
from torch.utils.data import Dataset
import spacy
from typing import List
from argparse import Namespace, ArgumentParser
import random


class LMDataset(Dataset):

    def __init__(self, h_params: Namespace):
        self.h_params = h_params
        self.sentences = []
        self.sentences: List[str]
        self.read_file()

        self.special_token = set()
        self.tokenizer = T5TokenizerFast.from_pretrained(self.h_params.T5_model_name)
        self.ids = None
        self.masks = None
        self.preprocess()
        pass

    def read_file(self):
        file_path = self.h_params.data_path
        with open(file_path, "r") as f:
            for row in f:
                row = row.strip().replace("\n", "")
                self.sentences.append(row)

    def preprocess(self):
        nlp = spacy.load("en_core_web_sm")
        for idx, sen in enumerate(self.sentences):
            sen: str
            entities = nlp(sen).ents
            for ent in entities:
                label = f"<{ent.label_}>"
                self.special_token.add(label)
                sen = sen.replace(str(ent), str(label))
            self.sentences[idx] = sen
        self.special_token = sorted(list(self.special_token))
        self.tokenizer.add_tokens(self.special_token)
        tokenized_sen = self.tokenizer.batch_encode_plus(self.sentences, max_length=self.h_params.max_seq_length,
                                                         padding="max_length",
                                                         return_tensors="pt", truncation=True, add_special_tokens=True)
        ids = tokenized_sen.data["input_ids"]
        masks = tokenized_sen.data["attention_mask"]
        self.ids = ids
        self.masks = masks

    def __getitem__(self, item):
        ids, masks = self.ids[item], self.masks[item]
        for idx, word in enumerate(ids):
            if int(ids[idx]) == self.tokenizer.pad_token_id:
                break
            p = random.random()
            counter = 0
            if p < self.h_params.dropout_prob:
                ids[idx] = self.tokenizer.get_vocab()[f"<extra_id_{counter}>"]
                counter += 1
        return ids, masks

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, default="data_sample.txt")
        parser.add_argument('--T5_model_name', type=str, default='t5-base')
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--dropout_prob", type=float, default=0.15)
        parser.add_argument("--learning_rate", type=float, default=3e-5)
        parser.add_argument("--adam_epsilon", type=float, default=1e-6)
        parser.add_argument("--num_train_epochs", type=int, default=1)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--output_dir", type=str, default="results/baseline")
        parser.add_argument("--gpus", type=int, default=0)
        return parser
