import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from EncoderDecoderModels import EncoderDecoder, device, EncoderRNN, AttnDecoderRNN, EOS_token, SOS_token, DecoderRNN,\
    BertEncoderDecoder
import random
from argparse import ArgumentParser
from sentence_pairs_dataset import PairsDS
import matplotlib.ticker as ticker
from lm_dataset import LMDataset


def print_graphs(loss_list, test_loss_list=None):
    plt.plot(loss_list, label="loss", c="red")
    if test_loss_list is not None:
        plt.plot(test_loss_list, label="test loss", c="green")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


def eval_net(model, test_dataloader):
    # max_len = test_dataloader.max_len  #TODO
    loss = nn.NLLLoss()
    printable_loss = 0
    model.eval()
    with torch.no_grad():
        failed = 0
        acc = 0
        counter = 0
        for input_seq, _, target_seq in tqdm(test_dataloader, total=len(test_dataloader)):
            target_seq = target_seq.squeeze(0).squeeze(0)
            input_seq = input_seq.squeeze(0).squeeze(0)
            # target_seq = target_seq[target_seq != -100]
            # if torch.numel(target_seq) == 0:
            #     failed += 1
            #     continue
            if device == "cuda:0":
                target_seq = target_seq.cuda()
                input_seq = input_seq.cuda()
            pred_seq = model(input_seq, target_seq, force_learning=False)
            # if pred_seq.size(1) != len(target_seq):
            #     pad = torch.zeros(1, target_seq.size(0)-pred_seq.size(1), pred_seq.size(2), device=device)
            #     for i in range(len(pad[0])):
            #         pad[0][i][1] = 1
            #     pred_seq = torch.cat((pred_seq.to(device), pad), dim=1)
            pred_seq = F.log_softmax(pred_seq.squeeze(0))
            acc += ((torch.argmax(pred_seq.cpu(), dim=-1)[0] == target_seq.cpu()).numpy().sum())
            counter += len(target_seq)
            ls = loss(pred_seq, target_seq.to(device))
            printable_loss += ls.item()
    return printable_loss/(len(test_dataloader)-failed), acc / counter


def train_net(model, train_dataloader, test_dataloader=None, epochs=1000, lr=0.00001, force_training_prob=0.5,
              print_every=1, ags=8, choose_by_loss=True):
    model.to(device)
    loss = nn.NLLLoss()
    loss_list = []
    test_loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0
    best_test_loss = float('inf')
    for epoch in range(epochs):
        printable_loss = 0
        i = 0
        failed = 0
        for input_seq, _, target_seq in tqdm(train_dataloader, total=len(train_dataloader)):
            # print(f'ids, input :{input_seq}')
            # print(f'labels, target: {target_seq}')
            p = True if random.random() > force_training_prob else 0
            target_seq = target_seq.squeeze(0).squeeze(0)
            # target_seq = target_seq[target_seq != -100]
            # if torch.numel(target_seq) == 0:
            #     failed += 1
            #     continue
            if device == "cuda:0":
                target_seq = target_seq.cuda()
                input_seq = input_seq.cuda()
            pred_seq = model(input_seq.squeeze(0).squeeze(0), target_seq, force_learning=False)
            # print(f'Traind: {pred_seq.shape}', target_seq.shape)
            # if pred_seq.size(1) != len(target_seq):
            #     pad = torch.zeros(1, target_seq.size(0)-pred_seq.size(1), pred_seq.size(2), device=device)
            #     for j in range(len(pad[0])):
            #         pad[0][j][1] = 1
            #     pred_seq = torch.cat((pred_seq.to(device), pad), dim=1)
            # print(f'predicted :{torch.argmax(F.log_softmax(pred_seq.squeeze(0)),1)}')
            ls = loss(F.log_softmax(pred_seq.squeeze(0)), target_seq.to(device))
            printable_loss += ls.item()
            if i % ags == 0:
                ls.backward()
                optimizer.step()
                optimizer.zero_grad()
            i += 1
        printable_loss /= (len(train_dataloader) - failed)
        loss_list.append(printable_loss)
        if epoch % print_every == 0:
            print(epoch)
            model.train(False)
            with torch.no_grad():
                for j in range(5):
                    sen_to_print, _, target_sen_to_print = train_dataloader.dataset.__getitem__(j)
                    sen_to_print = sen_to_print.squeeze(0).squeeze(0)
                    target_sen_to_print = target_sen_to_print.squeeze(0).squeeze(0)
                    # target_sen_to_print = target_sen_to_print[target_sen_to_print != -100]
                    if torch.numel(target_sen_to_print) == 0:
                        continue
                    if device == "cuda:0":
                        sen_to_print = sen_to_print.cuda()
                        target_sen_to_print = target_sen_to_print.cuda()
                    print(f'sentence ids: {sen_to_print}')
                    print(f"Output before: {target_sen_to_print}")
                    pred = model(sen_to_print, target_sen_to_print, force_learning=False)
                    pred_ids = pred.argmax(dim=1)
                    # for word in pred[0]:
                    #     topv, topi = word.topk(1)
                    #     in_word = topi.squeeze().detach().item()
                    #     pred_ids.append(in_word)
                    print(f"Output after: {target_sen_to_print}")
                    print(f"Predicted: {pred_ids}")
                    with open("./results/lm_train.txt", "a") as f:
                        f.write(f"Output: {target_sen_to_print}\n")
                        f.write(f"Predicted: {pred_ids}\n")
                        f.write("\n")
                print(f"epoch: {epoch} \t train loss is {printable_loss}")
                with open("./results/lm_train.txt", "a") as f:
                    f.write(f"epoch: {epoch} \t train loss is {printable_loss}")
                    f.write("\n")
            model.train(True)
        if test_dataloader is not None:
            model.train(False)
            test_loss, test_acc = eval_net(model, test_dataloader)
            if choose_by_loss:
                if test_loss < best_test_loss:
                    torch.save(model, "lm_model.pt")
                    best_test_loss = test_loss
            else:
                if test_acc > best_test_acc:
                    torch.save(model, "lm_model.pt")
                    best_test_loss = test_acc
            # print(f'Test acc: {test_acc}')
            if epoch % print_every == 0:
                print(f"test loss is {test_loss}, Test acc: {test_acc}")
                with open("./results/lm_train.txt", "a") as f:
                    f.write(f"test loss is {test_loss}, Test acc: {test_acc}")
                    f.write("\n")
            test_loss_list.append(test_loss)
            model.train(True)
    if test_dataloader is not None:
        print_graphs(loss_list, test_loss_list)
    else:
        print_graphs(loss_list)
    return model, loss_list, test_loss_list

###############################################
###############################################

teacher_forcing_ratio = 0.5


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.savefig("loss.png")
    plt.plot(points)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = PairsDS.add_model_specific_args(parser)
    h_params = parser.parse_args()
    lds = LMDataset(h_params, "datasets/full_lm_data.csv")
    train_lds, val_lds = random_split(lds, [len(lds)-int(0.2*len(lds)), int(0.2*len(lds))])
    train_dataloader = DataLoader(train_lds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_lds, batch_size=1, shuffle=False)
    # full_dataloader = DataLoader(lds, batch_size=1, shuffle=True)
    # for ids, masks, labels in train_dataloader:
    #     print(f'ids: {ids}')
    #     print(f'masks: {masks}')
    #     print(f'labels: {labels}')
    # print(len(train_dataloader), len(val_dataloader))
    print(lds.vocab_size)
    print(lds.is_test)
    model = BertEncoderDecoder(vocab_size=lds.vocab_size, max_len=128)
    model = train_net(model, train_dataloader, test_dataloader=val_dataloader)
