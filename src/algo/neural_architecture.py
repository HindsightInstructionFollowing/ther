import torch
from torch import nn
import torch.nn.functional as F
import contextlib

from algo.pg_base import RecurrentACModel, ACModel
from torch.distributions.categorical import Categorical


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


class MinigridConv(nn.Module, RecurrentACModel):
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

        else:
            self.size_after_text_viz_merge = self.size_after_conv


        # ================ Q-values, V, Advantages =========
        # ==================================================
        self.n_actions =               action_space.n
        self.dueling_dqn =             config["dueling_architecture"]
        self.last_hidden_fc_size =     config["last_hidden_fc_size"]
        self.use_memory =              config["use_memory"]
        self.memory_lstm_size =        self.size_after_text_viz_merge

        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.size_after_text_viz_merge, self.memory_lstm_size)

        self.critic = nn.Sequential(
            nn.Linear(self.size_after_text_viz_merge, self.last_hidden_fc_size),
            nn.ReLU(), #todo : tanh ?
            nn.Linear(self.last_hidden_fc_size, 1)
        )

        # If dueling is not used, this is normal Q-values
        self.actor = nn.Sequential(
                nn.Linear(self.size_after_text_viz_merge, self.last_hidden_fc_size),
                nn.ReLU(),
                nn.Linear(self.last_hidden_fc_size, self.n_actions)
            )

        # Initialize network
        self.apply(init_weights)

    def _text_embedding(self, mission, mission_length):

        # state["mission"] contains list of indices
        embedded = self.word_embedding(mission)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(input=embedded,
                                                   lengths=mission_length,
                                                   batch_first=True, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.text_rnn(packed)
        out_language = self.fc_language(hidden[0])
        out_language = F.relu(out_language)
        return out_language

    def forward(self, state, memory=None):

        if self.lstm_after_conv:
            batch_dim = state["image"].shape[0]

            out_conv = self.conv_net(state["image"].reshape(-1, self.channel, self.height, self.width))
            out_conv = out_conv.reshape(batch_dim, self.frames, -1)
            (outputs, (h_t, c_t)) = self.memory_rnn(out_conv)
            flatten_vision_and_text = h_t[0] # get last ht
        else:
            out_conv = self.conv_net(state["image"])
            flatten_vision_and_text = out_conv.view(out_conv.shape[0], -1)

        if not self.ignore_text:
            out_text = self._text_embedding(mission=state["mission"],
                                            mission_length=state["mission_length"])
            flatten_vision_and_text = torch.cat((flatten_vision_and_text, out_text), dim=1)

        if self.use_memory:
            hidden_memory = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden_memory = self.memory_rnn(flatten_vision_and_text, hidden_memory)
            flatten_vision_and_text = hidden_memory[0]
            memory = torch.cat(hidden_memory, dim=1)

        q_values = self.actor(flatten_vision_and_text)
        next_state_values = self.critic(flatten_vision_and_text)

        if self.dueling_dqn:
            # q_values =  next_state_values + q_values - q_values.mean(dim=1, keepdim=True)
            q_values =  next_state_values + q_values - q_values.mean()
            policy_dist = q_values
        else:
            policy_dist = Categorical(logits=F.log_softmax(q_values, dim=1))
            next_state_values = next_state_values.view(-1)

        return policy_dist, next_state_values, memory

    @property
    def memory_size(self):
        return 2*self.memory_lstm_size

    @property
    def semi_memory_size(self):
        return self.memory_lstm_size
