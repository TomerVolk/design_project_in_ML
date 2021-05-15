import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
from EncoderDecoderModels import EncoderDecoder, device
import random
from sentence_pairs_dataset import PairsDS


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
    max_len = test_dataloader.max_len  #TODO
    loss = nn.NLLLoss()
    printable_loss = 0
    model.eval()
    with torch.no_grad():
        for input_seq, target_seq in tqdm(test_dataloader, total=len(test_dataloader)):
            pred = model(input_seq, target_seq, len=max_len, force_learning=False)
            ls = loss(pred, target_seq)
            printable_loss += ls.item()
    return printable_loss/len(test_dataloader)


def train_net(model, train_dataloader, test_dataloader=None, epochs=1000, lr=0.005, force_training_prob=0.5,
              print_every=50):
    model.to(device)
    loss = nn.NLLLoss()
    loss_list = []
    test_loss_list = []
    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=lr)
    for epoch in range(epochs):
        printable_loss = 0
        for input_seq, target_seq in tqdm(train_dataloader, total=len(train_dataloader)):
            p = True if random.random() > force_training_prob else 0
            pred_seq = model(input_seq, target_seq, force_learning=p)
            ls = loss(pred_seq, target_seq)
            printable_loss += ls.item()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        printable_loss /= len(train_dataloader)
        loss_list.append(printable_loss)
        if epoch % print_every == 0:
            print(epoch)
            print(f"train loss is {printable_loss}")
        if test_dataloader is not None:
            test_loss = eval_net(model, test_dataloader)
            if epoch % print_every == 0:
                print(f"test loss is {test_loss}")
            test_loss_list.append(test_loss)
    if test_dataloader is not None:
        print_graphs(loss_list, test_loss_list)
    else:
        print_graphs(loss_list)
    return model, loss_list, test_loss_list


if __name__ == '__main__':
    pds = PairsDS()


