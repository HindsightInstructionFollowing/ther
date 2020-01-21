import torch
from torch import nn
import torch.nn.functional as F
import contextlib

from algo.pg_base import RecurrentACModel, ACModel
from torch.distributions.categorical import Categorical
import collections


import numpy as np

basic_transition = collections.namedtuple("Transition",
                                          ["current_state", "action", "reward", "next_state", "terminal",
                                           "mission", "mission_length", "gamma"])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=.1)
        m.bias.data.fill_(0.01)

def compute_accuracy(logits, labels):

    predicted_token = logits.argmax(dim=1)
    accuracy = (predicted_token == labels).sum()
    accuracy = float(accuracy.item())
    return accuracy / labels.size(0)


class MlpNet(nn.Module):
    def __init__(self, obs_space, action_space, config, device):
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

def conv_factory(input_shape, channels, kernels, strides, max_pool):
    assert len(input_shape) == 3, "shape should be 3 dimensionnal"
    conv_net = nn.Sequential()
    last_layer_channel = input_shape[0]
    for layer in range(len(channels)):
        conv_net.add_module(name='conv{}'.format(layer),
                                 module=nn.Conv2d(in_channels=last_layer_channel,
                                                  out_channels=channels[layer],
                                                  kernel_size=kernels[layer],
                                                  stride=strides[layer])
                                 )
        conv_net.add_module(name='relu{}'.format(layer),
                                 module=nn.ReLU()
                                 )

        # For next iter
        last_layer_channel = channels[layer]

        if max_pool[layer]:
            conv_net.add_module(name='max_pool{}'.format(layer),
                                module=nn.MaxPool2d(kernel_size=max_pool[layer])
                                )

    size_after_conv = int(np.prod(conv_net(torch.zeros(1, *input_shape)).shape))
    return conv_net, size_after_conv

class MinigridRecurrentPolicy(nn.Module):
    def __init__(self, obs_space, action_space, config, device):
        super().__init__()
        self.device = device

        # ================= VISION ====================
        # =============================================
        # If frame stacker
        c, h, w = obs_space["image"].shape

        self.channel = c
        self.height = h
        self.width = w

        # xxxxx_list[0] correspond to the first conv layer, xxxxx_list[1] to the second etc ...
        channel_list = config["conv_layers_channel"] if "conv_layers_channel" in config else [16, 32, 64]
        kernel_list = config["conv_layers_size"] if "conv_layers_size" in config else [2, 2, 2]
        stride_list = config["conv_layers_stride"] if "conv_layers_stride" in config else [1, 1, 1]
        max_pool_list = config["max_pool_layers"] if "max_pool_layers" in config else [2, 0, 0]
        projection_after_conv = config["projection_after_conv"]

        self.conv_net, self.size_after_conv = conv_factory(input_shape=obs_space["image"].shape,
                                                           channels=channel_list,
                                                           kernels=kernel_list,
                                                           strides=stride_list,
                                                           max_pool=max_pool_list)

        # ====================== TEXT ======================
        # ==================================================
        # Mission : word embedding => gru => linear => concat with vision
        self.ignore_text = config["ignore_text"]
        if not self.ignore_text:

            self.use_gated_attention = config["use_gated_attention"]
            num_token = int(obs_space["mission"].high.max())
            word_embedding_size =    config["word_embedding_size"]
            rnn_text_hidden_size =   config["rnn_text_hidden_size"]
            fc_text_embedding_size = config["fc_text_embedding_hidden"]

            self.word_embedding = nn.Embedding(num_token, word_embedding_size)
            self.text_rnn = nn.GRU(word_embedding_size, rnn_text_hidden_size, batch_first=True)

            # Project text using a linear layer or not
            if fc_text_embedding_size:
                self.fc_language = nn.Sequential(nn.Linear(in_features=rnn_text_hidden_size,
                                                            out_features=fc_text_embedding_size),
                                                  nn.ReLU())
                self.text_size = fc_text_embedding_size
            else:
                self.fc_language = lambda x: x
                self.text_size = rnn_text_hidden_size

            if self.use_gated_attention:
                num_features_last_cnn = channel_list[-1]
                self.att_linear = nn.Linear(self.text_size, num_features_last_cnn)
                self.text_size = 0

            self.size_after_text_viz_merge = self.size_after_conv + self.text_size

        else:
            self.size_after_text_viz_merge = self.size_after_conv

        # ============== PROJECT TEXT AND VISION ? ===============
        # ========================================================
        projection_size = config["projection_after_conv"]
        if projection_size:
            self.projection_after_merge = nn.Sequential(nn.Linear(self.size_after_text_viz_merge, projection_size),
                                                         nn.ReLU())
            self.size_after_text_viz_merge = projection_size
        else:
            self.projection_after_merge = lambda x:x


        # ================ Q-values, V, Advantages =========
        # ==================================================
        self.n_actions =           action_space.n
        self.memory_size =         config["rnn_state_hidden_size"]
        self.last_hidden_fc_size = config["last_hidden_fc_size"]
        self.memory_lstm_size =    self.size_after_text_viz_merge

        # Adding memory to the agent
        self.memory_rnn = nn.LSTM(input_size=self.size_after_text_viz_merge, hidden_size=self.memory_size, batch_first=True)

        self.critic = nn.Sequential(
            nn.Linear(self.memory_size, self.last_hidden_fc_size),
            nn.ReLU(),
            nn.Linear(self.last_hidden_fc_size, 1)
        )

        self.actor = nn.Sequential(
                nn.Linear(self.memory_size, self.last_hidden_fc_size),
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
        return out_language

    def forward(self, state, ht=None):

        max_seq_len = state["max_sequence_length"]
        sequences_length = state["state_sequence_lengths"]
        padding_mask = state["padding_mask"]
        n_sequence = sequences_length.size(0)

        # ======= VISION ========
        out_conv = self.conv_net(state["image"])
        flatten_vision_and_text = out_conv.view(-1, self.size_after_conv)

        # ======= TEXT AND MERGE ========
        if not self.ignore_text:
            out_text = self._text_embedding(mission=state["mission"],
                                            mission_length=state["mission_length"])

            if self.use_gated_attention:
                batch_size, last_conv_size, h, w = out_conv.size()

                attention_weights = torch.sigmoid(self.att_linear(out_text))
                attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)
                attention_weights = attention_weights.expand(batch_size, last_conv_size, h, w)
                assert attention_weights.size() == out_conv.size()
                flatten_vision_and_text = attention_weights * out_conv
                flatten_vision_and_text = flatten_vision_and_text.view(batch_size, -1)

            else:
                flatten_vision_and_text = torch.cat((flatten_vision_and_text, out_text), dim=1)


        # ======= PROJECTION ========
        flatten_vision_and_text = self.projection_after_merge(flatten_vision_and_text)

        vision_and_text_sequence_format = flatten_vision_and_text.view(n_sequence, max_seq_len, -1)
        vision_and_text_sequence_format = torch.nn.utils.rnn.pack_padded_sequence(input=vision_and_text_sequence_format,
                                                                                  lengths=sequences_length,
                                                                                  batch_first=True,
                                                                                  enforce_sorted=False)

        # ===== RECURRENT MODULE ======
        if ht is None:
            ht = torch.zeros(1, n_sequence, self.memory_size*2).to(self.device)

        hidden_memory = (ht[:, :, :self.memory_size], ht[:, :, self.memory_size:])
        all_ht, hidden_memory = self.memory_rnn(vision_and_text_sequence_format, hidden_memory)

        all_ht, size = torch.nn.utils.rnn.pad_packed_sequence(all_ht, batch_first=True)

        memory = torch.cat(hidden_memory, dim=2)

        flatten_vision_and_text = all_ht.view(-1, self.memory_size)

        # Delete all useless padding in state sequences, to avoid computing the q-values for them
        flatten_vision_and_text_without_padding = flatten_vision_and_text[padding_mask == 1]

        # ===== Dueling architecture ======
        q_values = self.actor(flatten_vision_and_text_without_padding)
        next_state_values = self.critic(flatten_vision_and_text_without_padding)

        q_values =  next_state_values + q_values - q_values.mean()
        return q_values, memory


class MinigridConvPolicy(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, config, device):
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

        # xxxxx_list[0] correspond to the first conv layer, xxxxx_list[1] to the second etc ...
        channel_list = config["conv_layers_channel"] if "conv_layers_channel" in config else [16, 32, 64]
        kernel_list = config["conv_layers_size"] if "conv_layers_size" in config else [2, 2, 2]
        stride_list = config["conv_layers_stride"] if "conv_layers_stride" in config else [1, 1, 1]
        max_pool_list = config["max_pool_layers"] if "max_pool_layers" in config else [2, 0, 0]

        self.conv_net, self.size_after_conv = conv_factory(input_shape=obs_space["image"].shape,
                                                           channels=channel_list,
                                                           kernels=kernel_list,
                                                           strides=stride_list,
                                                           max_pool=max_pool_list)

        # ====================== TEXT ======================
        # ==================================================
        # Mission : word embedding => gru => linear => concat with vision
        self.ignore_text = config["ignore_text"]
        if not self.ignore_text:

            self.use_gated_attention = config["use_gated_attention"]
            num_token = int(obs_space["mission"].high.max())
            word_embedding_size =    config["word_embedding_size"]
            rnn_text_hidden_size =   config["rnn_text_hidden_size"]
            fc_text_embedding_size = config["fc_text_embedding_hidden"]

            self.word_embedding = nn.Embedding(num_token, word_embedding_size)
            self.text_rnn = nn.GRU(word_embedding_size, rnn_text_hidden_size, batch_first=True)

            # Project text using a linear layer or not
            if fc_text_embedding_size:
                self.fc_language = nn.Sequential(nn.Linear(in_features=rnn_text_hidden_size,
                                                           out_features=fc_text_embedding_size),
                                                  nn.ReLU())
                self.text_size = fc_text_embedding_size
            else:
                self.fc_language = lambda x: x
                self.text_size = rnn_text_hidden_size

            if self.use_gated_attention:
                num_features_last_cnn = channel_list[-1]
                self.att_linear = nn.Linear(self.text_size, num_features_last_cnn)
                self.text_size = 0

            self.size_after_text_viz_merge = self.size_after_conv + self.text_size

        else:
            self.size_after_text_viz_merge = self.size_after_conv

        # ============== PROJECT TEXT AND VISION ? ===============
        # ========================================================
        projection_size = config["projection_after_conv"]
        if projection_size:
            self.projection_after_merge = nn.Sequential(nn.Linear(self.size_after_text_viz_merge, projection_size),
                                                         nn.ReLU())
            self.size_after_text_viz_merge = projection_size
        else:
            self.projection_after_merge = lambda x: x


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
        return out_language

    def forward(self, state, memory=None):

        out_conv = self.conv_net(state["image"])
        flatten_vision_and_text = out_conv.view(out_conv.shape[0], -1)

        if not self.ignore_text:
            out_text = self._text_embedding(mission=state["mission"],
                                            mission_length=state["mission_length"])

            if self.use_gated_attention:
                batch_size, last_conv_size, h, w = out_conv.size()

                attention_weights = torch.sigmoid(self.att_linear(out_text))
                attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)
                attention_weights = attention_weights.expand(batch_size, last_conv_size, h, w)
                assert attention_weights.size() == out_conv.size()
                flatten_vision_and_text = attention_weights * out_conv
                flatten_vision_and_text = flatten_vision_and_text.view(batch_size, -1)

            else:
                flatten_vision_and_text = torch.cat((flatten_vision_and_text, out_text), dim=1)


        # ===== PROJECTION =======
        flatten_vision_and_text = self.projection_after_merge(flatten_vision_and_text)

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


class InstructionGenerator(nn.Module):
    def __init__(self, input_shape, n_output, config, device):
        """
        Basic instruction generator
        Convert state (or trajectory) to an instruction in approximate english (not very sophisticated litterature, sorry)

        todo : trajectory
        """
        super().__init__()
        self.device = device

        # This is a convention, veryfied by the environment tokenizer, might be ugly
        self.BEGIN_TOKEN = 0
        self.END_TOKEN = 1

        self.vocabulary_size = n_output
        self.generator_max_len = config["generator_max_len"]

        channel_list = config["conv_layers_channel"]
        kernel_list = config["conv_layers_size"]
        stride_list = config["conv_layers_stride"]
        max_pool_list = config["max_pool_layers"]
        projection_size = config["projection_after_conv"]

        # Not being used at the moment
        # self.teacher_forcing_ratio = config["teacher_forcing"]
        self.conv_net, size_after_conv = conv_factory(input_shape=input_shape,
                                                      channels=channel_list,
                                                      kernels=kernel_list,
                                                      strides=stride_list,
                                                      max_pool=max_pool_list,
                                                      projection_size=projection_size)

        self.word_embedder = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=config["embedding_dim"])
        self.rnn_decoder = nn.GRU(input_size=config["embedding_dim"], hidden_size=int(size_after_conv), batch_first=True)
        self.mlp_decoder = nn.Linear(in_features=int(size_after_conv), out_features=n_output)

    def forward(self, states, teacher_sentence, lengths):
        """
        Takes states and instruction pairs
        :return: a tensor of instruction predicted by the network of size (batch_size * n_word_per_sequence, n_token)
        """

        import copy
        batch_size = teacher_sentence.size(0)
        max_length = lengths.max().item()

        conv_ht = self.conv_net(states)
        conv_ht = conv_ht.view(batch_size, -1)

        # Remove the last <pad> token as it's not useful to predict
        # teacher_sentence_cut = teacher_sentence[:, :-1]
        # lengths_reduced = lengths - 1

        embedded_sentences = self.word_embedder(teacher_sentence)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(input=embedded_sentences,
                                                                  lengths=lengths,
                                                                  batch_first=True, enforce_sorted=False)

        # hx size is (num_layers*num_directions, batch_size, hidden_size), 1 layer and 1 direction in this architecture
        conv_ht = conv_ht.unsqueeze(0) # Adding num_layer*directions dimension

        # Compute all ht in a forward pass, teacher_forcing 100% of the time
        ht, _ = self.rnn_decoder(input=packed_embedded, hx=conv_ht)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True) # Unpack sequence
        view_ht = ht.view(batch_size*max_length, -1)
        assert view_ht.size(1) == conv_ht.size(2)

        logits = self.mlp_decoder(input=view_ht)
        return logits

    def generate(self, input):

        batch_size = input.size(0)
        assert batch_size == 1, "While generating, input should have a batch size of 1 is {}".format(batch_size)

        out = self.conv_net(input)

        is_last_word = False
        last_ht = out.view(1, 1, -1)
        next_token = torch.empty(batch_size, 1).fill_(self.BEGIN_TOKEN).long().to(self.device)

        generated_token = []
        while not is_last_word:
            next_input = self.word_embedder(next_token)
            _, last_ht = self.rnn_decoder(input=next_input, hx=last_ht)
            softmax_token = F.softmax(self.mlp_decoder(last_ht.view(1, -1)), dim=1)
            _, next_token = torch.max(softmax_token, dim=1)

            if next_token == self.END_TOKEN or len(generated_token) > self.generator_max_len:
                is_last_word = True
            else:
                generated_token.append(next_token.item())

            next_token = next_token.unsqueeze(0)

        return torch.LongTensor(generated_token)




