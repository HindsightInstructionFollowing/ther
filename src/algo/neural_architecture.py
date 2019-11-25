import torch
from torch import nn
import torch.nn.functional as F
import contextlib

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=.1)
        m.bias.data.fill_(0.01)


class MlpNet(nn.Module):
    def __init__(self, obs_space, action_space, config):
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
    def __init__(self, obs_space, action_space, config):
        super().__init__()

        # ================= VISION ====================
        # =============================================
        # If frame stacker
        if len(obs_space["image"].shape) == 4:
            f, c, h, w = obs_space["image"].shape
        else:
            c, h, w = obs_space["image"].shape
            f = 1

        self.frames = f # Number of stacked frames
        self.channel = c
        self.height = h
        self.width = w

        self.lstm_after_conv = config["use_lstm_after_conv"]
        frames_conv_net = 1 if self.lstm_after_conv else self.frames
        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames_conv_net, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        output_conv_h = ((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = ((w - 1) // 2 - 2)  # w-3 without maxpooling
        self.size_after_conv = 64 * output_conv_h * output_conv_w

        # Encode each frame and then pass them through a rnn
        if self.lstm_after_conv:
            self.memory_rnn = nn.LSTM(self.size_after_conv, self.size_after_conv, batch_first=True)

        # ====================== TEXT ======================
        # ==================================================
        self.fc_text_embedding_size = config["fc_text_embedding_hidden"]
        self.num_token =              int(obs_space["mission"].high.max())
        self.last_hidden_fc_size =    config["last_hidden_fc_size"]

        # Mission : word embedding => gru => linear => concat with vision
        self.ignore_text = config["ignore_text"]
        if not self.ignore_text:
            self.word_embedding_size =       32
            self.rnn_text_hidden_size =      config["rnn_text_hidden_size"]
            self.size_after_text_viz_merge = self.fc_text_embedding_size + self.size_after_conv

            self.word_embedding = nn.Embedding(self.num_token, self.word_embedding_size)
            self.text_rnn =       nn.GRU(self.word_embedding_size, self.rnn_text_hidden_size, batch_first=True)
            self.fc_language =    nn.Linear(in_features=self.rnn_text_hidden_size,
                                            out_features=self.fc_text_embedding_size)


            self.text_context = contextlib.suppress() if config["learn_text"] else torch.no_grad()

        else:
            self.size_after_text_viz_merge = self.size_after_conv



        # ================ Q-values, V, Advantages =========
        # ==================================================
        self.n_actions =           action_space.n
        self.dueling =             config["dueling_architecture"]
        self.last_hidden_fc_size = config["last_hidden_fc_size"]

        if self.dueling:
            self.value_graph = nn.Sequential(
                nn.Linear(self.size_after_text_viz_merge, self.last_hidden_fc_size),
                nn.ReLU(),
                nn.Linear(self.last_hidden_fc_size, 1)
            )

        # If dueling is not used, this is normal Q-values
        self.advantages_graph = nn.Sequential(
                nn.Linear(self.size_after_text_viz_merge, self.last_hidden_fc_size),
                nn.ReLU(),
                nn.Linear(self.last_hidden_fc_size, self.n_actions)
            )

        # Initialize network
        self.apply(init_weights)

    def forward(self, state):

        if self.lstm_after_conv:
            batch_dim = state["image"].shape[0]

            out_conv = self.conv_net(state["image"].reshape(-1, self.channel, self.height, self.width))
            out_conv = out_conv.reshape(batch_dim, self.frames, -1)
            (outputs, (h_t, c_t)) = self.memory_rnn(out_conv)
            flatten = h_t[0] # get last ht
        else:
            out_conv = self.conv_net(state["image"])
            flatten = out_conv.view(out_conv.shape[0], -1)

        if not self.ignore_text:

            # Stop gradient or not
            with self.text_context:
                # state["mission"] contains list of indices
                embedded = self.word_embedding(state["mission"])
                # Pack padded batch of sequences for RNN module
                packed = nn.utils.rnn.pack_padded_sequence(input=embedded,
                                                           lengths=state["mission_length"],
                                                           batch_first=True, enforce_sorted=False)
                # Forward pass through GRU
                outputs, hidden = self.text_rnn(packed)
                out_language =    self.fc_language(hidden[0])
                out_language =    F.relu(out_language)
                flatten =         torch.cat((flatten, out_language), dim=1)

        # hidden_out = F.relu(self.fc_hidden(flatten))
        qvals = self.advantages_graph(flatten)

        if self.dueling:
            values = self.value_graph(flatten)
            # qvals =  values + qvals - qvals.mean(dim=1, keepdim=True)
            qvals =  values + qvals - qvals.mean()

        return qvals