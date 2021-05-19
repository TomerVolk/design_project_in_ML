import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from EncoderDecoderModels import EncoderDecoder, device, EncoderRNN, AttnDecoderRNN, EOS_token, SOS_token, DecoderRNN
import random
from argparse import ArgumentParser
from sentence_pairs_dataset import PairsDS
import matplotlib.ticker as ticker


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
              print_every=2, ags=50):
    model.to(device)
    loss = nn.NLLLoss()
    loss_list = []
    test_loss_list = []
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        printable_loss = 0
        i = 0
        for input_seq, _, target_seq, _ in tqdm(train_dataloader, total=len(train_dataloader)):
            p = True if random.random() > force_training_prob else 0
            target_seq = target_seq.squeeze(0).squeeze(0)
            pred_seq = model(input_seq.squeeze(0).squeeze(0), target_seq, force_learning=p)
            if pred_seq.size(1) != len(target_seq):
                pad = torch.zeros(1, target_seq.size(0)-pred_seq.size(1), pred_seq.size(2), device=device)
                for i in range(len(pad[0])):
                    pad[0][i][1] = 1
                pred_seq = torch.cat((pred_seq.to(device), pad), dim=1)
            ls = loss(F.log_softmax(pred_seq.squeeze(0)), target_seq.to(device))
            printable_loss += ls.item()
            if i % ags == 0:
                ls.backward()
                optimizer.step()
            i += 1
        printable_loss /= len(train_dataloader)
        loss_list.append(printable_loss)
        if epoch % print_every == 0:
            print(epoch)
            torch.save(model, "first_model.pt")
            i = random.randint(1, 200)
            with torch.no_grad():
                sen_to_print, _, target_sen_to_print, _ = train_dataloader.dataset.__getitem__(i)
                sen_to_print = sen_to_print.squeeze(0).squeeze(0)
                target_sen_to_print = target_sen_to_print.squeeze(0).squeeze(0)
                pred = model(sen_to_print, target_sen_to_print, force_learning=False)
                pred_ids = []
                for word in pred[0]:
                    topv, topi = word.topk(1)
                    in_word = topi.squeeze().detach().item()
                    pred_ids.append(in_word)
            print(train_dataloader.dataset.tokenizer.decode(target_sen_to_print))
            print(train_dataloader.dataset.tokenizer.decode(sen_to_print))
            print(train_dataloader.dataset.tokenizer.decode(pred_ids))
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

###############################################
###############################################

teacher_forcing_ratio = 0.5


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, do_step=False,max_length=128):
    encoder_hidden = encoder.initHidden()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, torch.tensor([target_tensor[di].item()], device=device))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, torch.tensor([target_tensor[di].item()], device=device))
            if decoder_input.item() == EOS_token:
                break
    if do_step:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    return loss.item() / (target_length*50)


def trainIters(train_dataloader, encoder, decoder, epochs, print_every=1, plot_every=1, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        i = 0
        for input_seq, _, target_seq, _ in tqdm(train_dataloader, total=len(train_dataloader)):
            input_tensor = input_seq.squeeze(0).squeeze(0).to(device)
            target_tensor = target_seq.squeeze(0).squeeze(0).to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, do_step=(i % 50 == 0))
            print_loss_total += loss
            plot_loss_total += loss
            i += 1

        if epoch % print_every == 0:
            print(epoch)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(print_loss_avg)
            with torch.no_grad():
                print_in_tensor, _, print_tar_tensor, _ = train_dataloader.dataset.__getitem__(5)
                input_tensor = print_in_tensor.squeeze(0).squeeze(0)
                input_length = print_in_tensor.squeeze(0).squeeze(0).size(0)
                target_length = print_tar_tensor.squeeze(0).squeeze(0).size(0)
                encoder_outputs = torch.zeros(128, encoder.hidden_size, device=device)
                encoder_hidden = encoder.initHidden()
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_tensor[ei].to(device), encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]
                decoder_input = torch.tensor([[SOS_token]], device=device)
                decoder_hidden = encoder_hidden
                pred_ids = []
                for di in range(target_length):
                    # decoder_output, decoder_hidden, decoder_attention = decoder(
                    #     decoder_input.to(device), decoder_hidden, encoder_outputs)
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
                    pred_ids.append(decoder_input)
                    if decoder_input.item() == EOS_token:
                        break
                print(train_dataloader.dataset.tokenizer.decode(input_tensor))
                print(train_dataloader.dataset.tokenizer.decode(print_tar_tensor.squeeze(0).squeeze(0)))
                print(train_dataloader.dataset.tokenizer.decode(pred_ids))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = PairsDS.add_model_specific_args(parser)
    # parser = Trainer.add_argparse_args(parser)
    h_params = parser.parse_args()
    # pds = PairsDS(h_params, "datasets/clean_dataset.csv")
    # torch.save(pds, "pairs_dataset.pt")
    pds = torch.load("pairs_dataset.pt")
    # pds.tokenizer.decode()
    pairs_dataloader = DataLoader(pds, batch_size=1, shuffle=True)
    # print(device)
    # model = EncoderDecoder(vocab_size=len(pds.tokenizer.get_vocab()), max_len=128)
    # model = train_net(model, pairs_dataloader)
    # torch.save(model, "first_model.pt")
    encoder = EncoderRNN(input_size=len(pds.tokenizer.get_vocab()), hidden_size=256)
    # decoder = AttnDecoderRNN(hidden_size=256, output_size=len(pds.tokenizer.get_vocab()))
    decoder = DecoderRNN(hidden_size=256, output_size=len(pds.tokenizer.get_vocab()))
    trainIters(pairs_dataloader, encoder.to(device), decoder.to(device), 1000)



