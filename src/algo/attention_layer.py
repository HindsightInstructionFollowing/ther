import torch
from torch import nn
import torch.nn.functional as F

class DummyAttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.rnn_decoder = nn.GRU(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=1,
                                  batch_first=True)

    def forward(self, next_input, input_hidden, encoder_hts):
        return self.rnn_decoder(next_input, input_hidden)



class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_length, device):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = device

        self.attn = nn.Linear(input_size, self.max_length)
        self.attn_combine = nn.Linear(input_size, self.hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, next_input, input_hidden, encoder_hts):

        attn_weights = torch.softmax(
            self.attn(torch.cat((next_input, input_hidden), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_hts)

        output = torch.cat((next_input, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.gru(output, input_hidden.unsqueeze(0))

        return output, hidden