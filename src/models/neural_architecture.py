import torch
from torch import nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity='relu')


class MlpNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        n_hidden = 20

        self.fc1 = nn.Linear(obs_space.spaces["image"].shape[0], out_features=n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_features=n_hidden)
        self.fc3 = nn.Linear(n_hidden, action_space.n)

        self.apply(init_weights)

    def forward(self, state):
        state = state["image"]
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        return self.fc3(out)



class MinigridConv(nn.Module):
    def __init__(self, obs_space, action_space, use_lstm_after_conv):
        super().__init__()

        if len(obs_space["image"].shape) == 4:
            f, c, h, w = obs_space["image"].shape
        else:
            raise NotImplementedError("Image shape should be 4, is {}, try using frame stacker wrapper".format(
                len(obs_space["image"].shape)))

        self.num_token = int(obs_space["mission"].high.max())

        self.n_actions = action_space.n

        self.fc_text_embedding_size = 32
        self.lstm_after_conv = use_lstm_after_conv
        self.frames = f
        self.c = c
        self.h = h
        self.w = w

        output_conv_h = ((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = ((w - 1) // 2 - 2)  # w-3 without maxpooling

        self.size_after_conv = 64 * output_conv_h * output_conv_w

        if self.lstm_after_conv:
            self.memory_rnn = nn.LSTM(self.size_after_conv, self.size_after_conv, batch_first=True)

        frames_conv_net = 1 if use_lstm_after_conv else self.frames

        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames_conv_net, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.word_embedding_size = 32
        self.word_embedding = nn.Embedding(self.num_token, self.word_embedding_size)
        self.rnn_text_embedding_size = 128
        self.text_rnn = nn.GRU(self.word_embedding_size, self.rnn_text_embedding_size, batch_first=True)
        self.fc_language = nn.Linear(in_features=self.rnn_text_embedding_size, out_features=self.fc_text_embedding_size)

        self.fc_out = nn.Linear(in_features=self.fc_text_embedding_size + self.size_after_conv, out_features=self.n_actions)

        # Initialize network
        self.apply(init_weights)

    def forward(self, state):

        if self.lstm_after_conv:
            batch_dim = state["image"].shape[0]

            out_conv = self.conv_net(state["image"].reshape(-1, self.c, self.h, self.w))
            out_conv = out_conv.reshape(batch_dim, self.frames, -1)
            (outputs, (h_t, c_t)) = self.memory_rnn(out_conv)
            flatten = h_t[0] # get last ht
        else:
            out_conv = self.conv_net(state["image"])
            flatten = out_conv.view(out_conv.shape[0], -1)

        # state["mission"] contains list of indices
        embedded = self.word_embedding(state["mission"])
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(input=embedded,
                                                   lengths=state["mission_length"],
                                                   batch_first=True, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.text_rnn(packed)
        out_language = self.fc_language(hidden[0])

        out_language = F.relu(out_language)

        concat = torch.cat((flatten, out_language), dim=1)
        return self.fc_out(concat)