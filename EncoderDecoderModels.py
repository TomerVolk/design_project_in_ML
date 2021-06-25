import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.utils
from transformers import DistilBertModel

SOS_token = 32100
EOS_token = 1
device = "cuda:0"
# device = "cpu"
print(device)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=500, dropout=0.2):
        super(Encoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        # self.dropout_layer = nn.Dropout(p=self.dropout_p)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=1,
                            dropout=self.dropout_p)

    def forward(self, sen):
        embedded_sen = self.embedding_layer(sen.to(device).unsqueeze(0))
        # drop_sen = self.dropout_layer(sen)
        encoded_sen, inner_state = self.lstm(embedded_sen)
        return encoded_sen, inner_state


class DecoderCell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecoderCell, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.output_dim, num_layers=1,
                            batch_first=True)

    def forward(self, x, prev_hidden):
        out, inner_state = self.lstm(x, prev_hidden)
        return out, inner_state


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim=300, hidden_size=500, dropout=0.2, linear_dim=1500):
        super(Decoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decoder_call = DecoderCell(input_dim=self.hidden_size, output_dim=self.hidden_size)
        self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size + embedding_dim, self.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.linear_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.linear_dim, out_features=self.vocab_size)
        )

    def cell_forward(self, output, encoder_outputs, inner_state):
        embedded = self.embedding_layer(output.to(device)).view(1, 1, -1)
        embedded = self.dropout_layer(embedded)
        hidden = inner_state[0]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, inner_state = self.decoder_call(output, inner_state)
        return output, inner_state

    def forward(self, encoder_outputs, len, inner_state, target_tensor, force_learning=True):
        outputs = None
        in_word = torch.tensor([[SOS_token]], device=self.device)
        for i in range(len):
            output, inner_state = self.cell_forward(output=in_word, encoder_outputs=encoder_outputs,
                                                    inner_state=inner_state)
            output = self.head(output)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=1)
            if force_learning:
                in_word = target_tensor[i]
            else:
                topv, topi = output.topk(1)
                in_word = topi.squeeze().detach()
                if in_word.item() == EOS_token:
                    break
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim=300, hidden_dim=500, dropout=0.2, linear_dim=1500):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=vocab_size, max_len=max_len, embedding_dim=embedding_dim,
                               hidden_size=hidden_dim, dropout=dropout, linear_dim=linear_dim)

    def forward(self, sen, target_tensor, len=None, force_learning=True):
        encoded_sen, inner_state = self.encoder(sen)
        pad = torch.zeros((1, self.max_len-encoded_sen.size(1), self.hidden_dim), device=device)
        encoded_sen = torch.cat((encoded_sen, pad), dim=1)
        if len is None:
            len = target_tensor.size(0)
        output = self.decoder(encoded_sen, len, inner_state, target_tensor, force_learning)
        return output


class BertEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim=300, dropout=0.2, linear_dim=1500):
        super(BertEncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.encoder.to(device)
        self.encoder.resize_token_embeddings(vocab_size)
        self.hidden_dim = self.encoder.config.dim
        global SOS_token
        SOS_token = 101
        global EOS_token
        EOS_token = 102
        self.decoder = Decoder(vocab_size=vocab_size, max_len=max_len, embedding_dim=embedding_dim,
                               hidden_size=self.hidden_dim, dropout=dropout, linear_dim=linear_dim)

    def forward(self, sen, target_tensor, len=None, force_learning=True):
        encoded_sen = self.encoder(sen.unsqueeze(0))[0]
        # pad = torch.zeros((1, self.max_len-encoded_sen.size(1), self.hidden_dim), device=device)
        # encoded_sen = torch.cat((encoded_sen, pad), dim=1)
        if len is None:
            len = target_tensor.size(0)
        inner_state = self.init_inner_state()
        output = self.decoder(encoded_sen, len, inner_state, target_tensor, force_learning)
        return output

    def init_inner_state(self):
        return torch.ones(1, 1, self.hidden_dim, device=device), torch.ones(1, 1, self.hidden_dim, device=device)


# model = BertEncoderDecoder(vocab_size=10000, max_len=50)
# model.to(device)
# in_tensor = torch.tensor([0, 2, 3, 4, 5, 1])
# target_tensor = torch.tensor([0, 4, 5, 6, 7, 1])
# # padding = torch.tensor([99]*(50-len(in_tensor)))
# # in_tensor = torch.cat((in_tensor, padding))
# out = model(in_tensor, target_tensor, force_learning=True)
# print(out)
# print(out.shape)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# exit(0)


####################################################################
####################################################################
####################################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=128):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

###