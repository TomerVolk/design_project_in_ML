import pickle

import torch.optim as optim
import torch.nn as nn
import torch
import warnings
import pandas as pd


from KeyWordsDataset import KeyWordsDataset
from t5model import KeyWordGeneration
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model_evaluation import init_evaluator, get_scores_single_sentence


def train(model, train_dataloader, test_dataloader, tokenizer, evaluation_model, evaluation_vectorizer,
          epochs=1000, lr=0.0001, ags=150):

    model.to('cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        i = 0
        for input_ids, attention_mask, labels in tqdm(train_dataloader, total=len(train_dataloader)):

            input_ids = input_ids.cuda()
            # copy_ids = input_ids.copy()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

            i += 1
            _, loss = model.forward(input_ids, attention_mask, labels)
            loss.backward()
            if i % ags == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f'Epoch: {epoch} \t loss: {loss.item()}')

        # with torch.no_grad():
        #     for j in [10, 50, 100]:
        #         input_ids, attention_mask, _ = train_dataloader.dataset.__getitem__(j)
        #         input_ids = input_ids.cuda()
        #         attention_mask = attention_mask.cuda()
        #         output_ids = model.eval_forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        #         print(f'Epoch: {epoch} \t Train: Keywords: "{tokenizer.decode(input_ids)}"'
        #               f' \t Sent: "{tokenizer.decode(output_ids[0])}"')
        # with torch.no_grad():
        #     for j in [10, 50, 100]:
        #         input_ids, attention_mask, _ = test_dataloader.dataset.__getitem__(j)
        #         print(input_ids.shape)
        #         input_ids = input_ids.cuda()
        #         attention_mask = attention_mask.cuda()
        #         output_ids = model.eval_forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        #         print(f'Epoch: {epoch} \t Test: Keywords: "{tokenizer.decode(input_ids)}"'
        #               f' \t Sent: "{tokenizer.decode(output_ids[0])}, score: {get_scores_single_sentence(tokenizer.decode(output_ids[0]))}"')
        if epoch > 8:
            model.train(False)
            with torch.no_grad():
                results_dict = {'w_score': [], 'keywords': [], 'sentence': [], 'epoch': []}
                for i in range(5):
                    results_dict[f'score_{i}'] = []
                for input_idx, (input_ids, attention_mask, _) in tqdm(enumerate(test_dataloader), 'test: ', total=len(test_dataloader)):
                    # input_ids, attention_mask, _ = test_dataloader.dataset.__getitem__(j)
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    # output_ids = model.eval_forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
                    output_ids = model.eval_forward(input_ids, attention_mask)
                    sent = tokenizer.decode(output_ids[0])
                    weighted_score, all_scores = get_scores_single_sentence(sent,
                                                       evaluation_model, evaluation_vectorizer)
                    keywords = tokenizer.decode(input_ids[0])
                    results_dict['w_score'].append(float(weighted_score))
                    print(all_scores[0])
                    for i, score in enumerate(all_scores[0]):
                        results_dict[f'score_{i}'].append(score)
                    results_dict['keywords'].append(keywords)
                    results_dict['sentence'].append(sent)
                    results_dict['epoch'].append(epoch)
                    if input_idx % 20 == 0:
                        print(f'Epoch: {epoch} \t w_score: {weighted_score} \t all_scores: {all_scores} \t keywords: "{keywords}"'
                              f' \t sent: "{sent}"')
                df = pd.DataFrame.from_dict(results_dict).sort_values(by='w_score', ascending=False)
                df.to_csv('./results/model_results_with_evaluation_another_try.csv', index=False)

            model.train(True)
        torch.save(model, 'trained_models/keyword_model_with_evaluation_4_scores.pt')




if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # bert, vec = None, None
    bert, vec = init_evaluator()
    print("created model")
    ds = KeyWordsDataset('datasets/keywords_30k.csv')
    model = KeyWordGeneration()
    train_lds, val_lds = random_split(ds, [len(ds) - int(0.01 * len(ds)), int(0.01 * len(ds))])
    train_dataloader = DataLoader(train_lds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(val_lds, batch_size=1, shuffle=False)

    with open('datasets/train_dataloader.pt', 'wb') as file:
        pickle.dump(train_dataloader, file)

    with open('datasets/test_dataloader.pt', 'wb') as file:
        pickle.dump(test_dataloader, file)

    train(model, train_dataloader, test_dataloader, ds.tokenizer, bert, vec)
