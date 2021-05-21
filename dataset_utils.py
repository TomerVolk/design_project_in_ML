from transformers import T5TokenizerFast
from torch.utils.data import Dataset
import spacy
from argparse import Namespace, ArgumentParser


class BaseDataset(Dataset):

    def __init__(self, h_params: Namespace, file_path):
        self.h_params = h_params
        self.read_file(file_path)

        self.special_token = []
        self.tokenizer = T5TokenizerFast.from_pretrained(self.h_params.T5_model_name)

    def read_file(self, file_path):
        raise NotImplementedError

    def preprocess(self, sentences, add_bos):
        nlp = spacy.load("en_core_web_sm")
        added_tokens = []
        for idx, sen in enumerate(sentences):
            sen: str
            if add_bos:
                sen = "<BOS>" + sen
                if "<BOS>" not in self.special_token:
                    added_tokens.append("<BOS>")
                    self.special_token.append("<BOS>")
            entities = nlp(sen).ents
            for ent in entities:
                label = f"<{ent.label_}>"
                if label not in self.special_token:
                    added_tokens.append(label)
                    self.special_token.append(label)
                sen = sen.replace(str(ent), str(label))
            sentences[idx] = sen
        self.tokenizer.add_tokens(added_tokens)
        if not add_bos:
            tokenized_sen = self.tokenizer.batch_encode_plus(sentences, max_length=self.h_params.max_seq_length,
                                                             padding="do_not_pad",
                                                             return_tensors="pt", truncation=True,
                                                             add_special_tokens=True)
            ids = tokenized_sen.data["input_ids"]
            masks = tokenized_sen.data["attention_mask"]
            return ids, masks
        ids, masks = [], []
        for sen in sentences:
            tokenized_sen = self.tokenizer.batch_encode_plus([sen], max_length=self.h_params.max_seq_length,
                                                             padding="do_not_pad",
                                                             return_tensors="pt", truncation=True,
                                                             add_special_tokens=True)
            cur_id = tokenized_sen.data["input_ids"]
            cur_mask = tokenized_sen.data["attention_mask"]
            ids.append(cur_id)
            masks.append(cur_mask)
        return ids, masks

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_path', type=str, default="datasets/clean_dataset.csv")
        parser.add_argument('--dev_path', type=str, default="datasets/clean_dataset.csv")
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
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--gpus", type=int, default=0)
        return parser
