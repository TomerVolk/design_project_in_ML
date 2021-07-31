import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
import pandas as pd


def read_file(file_path):
    df = pd.read_csv(file_path)
    df['labels_text'] = df['labels'].apply(lambda x: x.replace(',', '').replace("'", '')[1:-1])

    sentences_lst = df['text'].to_list()
    keywords_lst = df['labels_text'].to_list()
    return sentences_lst, keywords_lst


class KeyWordsDataset(Dataset):

    def __init__(self, file_path):
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        read_file(file_path)
        self.sentences, self.keywords = read_file(file_path)
        self.model_sent = self.tokenizer.batch_encode_plus(self.sentences, return_tensors="pt", max_length=100,
                                                           padding="max_length")
        self.model_keywords = self.tokenizer.batch_encode_plus(self.keywords, return_tensors="pt", max_length=20,
                                                           padding="max_length")

    def __getitem__(self, item):
        copy_tensor = self.model_keywords['input_ids'][item].clone()
        end_of_keywords = (copy_tensor == 1).nonzero(as_tuple=True)[0].item()
        perm = torch.randperm(end_of_keywords)
        copy_tensor[:end_of_keywords] = copy_tensor[perm]

        return copy_tensor, self.model_keywords["attention_mask"][item],\
               self.model_sent['input_ids'][item]

    def __len__(self):
        return len(self.sentences)


# if __name__ == '__main__':
#     df = pd.read_csv('datasets/keywords.csv')
#     df['labels_text'] = df['labels'].apply(lambda x: x.replace(',', '').replace("'", '')[1:-1])
#
#     sentences_lst = df['text'].to_list()
#     keywords_lst = df['labels_text'].to_list()
#     print(sentences_lst[:2])
#     tokenizer = T5TokenizerFast.from_pretrained("t5-small")
#     model_sent = tokenizer.batch_encode_plus(sentences_lst[:2], return_tensors="pt", max_length=40,
#                                                        padding="max_length")
#     print(model_sent)
#
