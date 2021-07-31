import torch.optim as optim
import torch.nn as nn
import torch


from KeyWordsDataset import KeyWordsDataset
from t5model import KeyWordGeneration
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train(model, train_dataloader, test_dataloader, tokenizer, epochs=1000, lr=0.0001, ags=8):
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
        with torch.no_grad():
            for j in [10, 50, 100]:
                input_ids, attention_mask, _ = train_dataloader.dataset.__getitem__(j)
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                output_ids = model.eval_forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
                print(f'Epoch: {epoch} \t Train: Keywords: "{tokenizer.decode(input_ids[0])}"'
                      f' \t Sent: "{tokenizer.decode(output_ids[0])}"')
        with torch.no_grad():
            for j in [10, 50, 100]:
                input_ids, attention_mask, _ = test_dataloader.dataset.__getitem__(j)
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                output_ids = model.eval_forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
                print(f'Epoch: {epoch} \t Test: Keywords: "{tokenizer.decode(input_ids[0])}"'
                      f' \t Sent: "{tokenizer.decode(output_ids[0])}"')
        torch.save(model, 'trained_models/keyword_model.pt')




if __name__ == '__main__':
    ds = KeyWordsDataset('datasets/keywords.csv')
    model = KeyWordGeneration()
    train_lds, val_lds = random_split(ds, [len(ds) - int(0.2 * len(ds)), int(0.2 * len(ds))])
    train_dataloader = DataLoader(train_lds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(val_lds, batch_size=1, shuffle=False)

    train(model, train_dataloader, test_dataloader, ds.tokenizer)
