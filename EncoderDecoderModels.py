import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.utils

SOS_token = 0
EOS_token = 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=500, dropout=0.2):
        super(Encoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.dropout_p = dropout
        # self.dropout_layer = nn.Dropout(p=self.dropout_p)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=1)

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
        embedded = self.embedding_layer(output).view(1, 1, -1)
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
        in_word = torch.tensor([[EOS_token]], device=self.device)
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
        print(encoded_sen.shape)
        pad = torch.zeros((1, self.max_len-encoded_sen.size(1), self.hidden_dim))
        print(pad.shape)
        encoded_sen = torch.cat((encoded_sen, pad), dim=1)
        print(encoded_sen.shape)
        if len is None:
            len = target_tensor.size(0)
        print(len)
        output = self.decoder(encoded_sen, len, inner_state, target_tensor, force_learning)
        return output


model = EncoderDecoder(vocab_size=100, max_len=50)
model.to(device)
in_tensor = torch.tensor([0, 2, 3, 4, 5, 1])
target_tensor = torch.tensor([0, 4, 5, 6, 7, 1])
# padding = torch.tensor([99]*(50-len(in_tensor)))
# in_tensor = torch.cat((in_tensor, padding))
out = model(in_tensor, target_tensor, force_learning=True)
print(out)
print(out.shape)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
