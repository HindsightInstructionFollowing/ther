import torch
from torch import nn
import torch.nn.functional as F
import contextlib

from algo.pg_base import RecurrentACModel, ACModel
from torch.distributions.categorical import Categorical

from algo.attention_layer import AttnDecoderRNN, DummyAttnDecoder

import numpy as np

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

def conv_factory(input_shape, channels, kernels, strides, max_pool, batch_norm):
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

        if batch_norm[layer]:
            conv_net.add_module('batch_norm{}'.format(layer),
                                module=nn.BatchNorm2d(channels[layer]))

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

        self.n_actions = action_space.n
        self.device =    device

        # ================= VISION ====================
        # =============================================

        channel_list =  config["conv_layers_channel"]
        self.conv_net, self.size_after_conv = conv_factory(input_shape=obs_space["image"].shape,
                                                           channels=channel_list,
                                                           kernels=config["conv_layers_size"],
                                                           strides=config["conv_layers_stride"],
                                                           max_pool=config["max_pool_layers"],
                                                           batch_norm=config["batch_norm_layers"])

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

        # ======== ADD ACTION ======
        self.use_action_rnn = config["use_action_rnn"]
        if self.use_action_rnn:
            action_embedding_size = config["action_embedding_size"]
            self.action_embedding = nn.Embedding(self.n_actions + 1, action_embedding_size) # + 1 is for padding action
            self.size_after_text_viz_merge += action_embedding_size


        # ================ Q-values, V, Advantages =========
        # ==================================================
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

        # ======== ADD ACTION =====
        if self.use_action_rnn:
            action_embedding = self.action_embedding(state["last_action"]).view(n_sequence, max_seq_len, -1)
            vision_and_text_sequence_format = torch.cat([vision_and_text_sequence_format, action_embedding], dim=2)

        # ===== RECURRENT MODULE ======
        if ht is None:
            ht = torch.zeros(1, n_sequence, self.memory_size*2).to(self.device)

        vision_and_text_sequence_format = torch.nn.utils.rnn.pack_padded_sequence(input=vision_and_text_sequence_format,
                                                                                  lengths=sequences_length,
                                                                                  batch_first=True,
                                                                                  enforce_sorted=False)

        hidden_memory = (ht[:, :, :self.memory_size], ht[:, :, self.memory_size:])

        all_ht, hidden_memory = self.memory_rnn(vision_and_text_sequence_format, hidden_memory)
        all_ht, size = torch.nn.utils.rnn.pad_packed_sequence(all_ht, batch_first=True)

        memory = torch.cat(hidden_memory, dim=2)

        flatten_vision_and_text = all_ht.view(-1, self.memory_size)

        # Delete all useless padding in state sequences, to avoid computing the q-values for them
        # If state sequence is shorter, padding doesn't match lstm's output, so remove trailing padding
        padding_mask = padding_mask.view(all_ht.size(0), -1)[:, :all_ht.size(1)].contiguous().view(-1)
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

        channel_list = config["conv_layers_channel"]
        self.conv_net, self.size_after_conv = conv_factory(input_shape=obs_space["image"].shape,
                                                           channels=channel_list,
                                                           kernels=config["conv_layers_size"],
                                                           strides=config["conv_layers_stride"],
                                                           max_pool=config["max_pool_layers"],
                                                           batch_norm=config["batch_norm_layers"])

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
        """
        super().__init__()
        self.device = device

        # This is a convention, veryfied by the environment tokenizer, might be ugly
        self.BEGIN_TOKEN = 0
        self.END_TOKEN = 1
        self.PAD_TOKEN = 2


        self.vocabulary_size = n_output
        self.generator_max_len = config["generator_max_len"]
        self.input_shape = input_shape

        channel_list =                        config["conv_layers_channel"]
        projection_after_conv =               config["projection_after_conv"]
        self.n_state_to_predict_instruction = config["n_state_to_predict_instruction"]
        trajectory_encoding_rnn             = config["trajectory_encoding_rnn"]

        dropout =               config["dropout"]
        hidden_size =           config["decoder_hidden"]

        conv_input = self.input_shape[-3:]
        self.conv_net, self.size_after_conv = conv_factory(input_shape=conv_input,
                                                           channels=channel_list,
                                                           kernels=config["conv_layers_size"],
                                                           strides=config["conv_layers_stride"],
                                                           max_pool=config["max_pool_layers"],
                                                           batch_norm=config["batch_norm_layers"])



        embedding_size = config["embedding_dim"]
        self.word_embedder = nn.Embedding(num_embeddings=self.vocabulary_size,
                                          embedding_dim=embedding_size,
                                          padding_idx=2)

        if projection_after_conv:
            self.project_after_conv = nn.Linear(self.size_after_conv, projection_after_conv)
            self.size_after_conv = projection_after_conv
        else:
            self.project_after_conv = lambda x:x

        if self.n_state_to_predict_instruction > 1:
            self.trajectory_encoding = nn.GRU(input_size=self.size_after_conv,
                                              hidden_size=trajectory_encoding_rnn,
                                              num_layers=1,
                                              batch_first=True)

            self.rnn_decoder = AttnDecoderRNN(input_size=embedding_size+trajectory_encoding_rnn,
                                              hidden_size=self.size_after_conv,
                                              output_size=self.size_after_conv,
                                              max_length=self.n_state_to_predict_instruction,
                                              device=self.device)
        else:
            self.rnn_decoder = DummyAttnDecoder(input_size=config["embedding_dim"],
                                                hidden_size=self.size_after_conv)




        self.mlp_decoder_hidden = nn.Linear(in_features=int(self.size_after_conv), out_features=hidden_size)
        self.mlp_decoder = nn.Linear(in_features=hidden_size, out_features=n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, teacher_sentence, lengths):
        """
        Takes states and instruction pairs
        :return: a tensor of instruction predicted by the network of size (batch_size * n_word_per_sequence, n_token)
        """
        # Dim : (batch, states_sequence, channels, w, h)
        states = states[:, -self.n_state_to_predict_instruction:].contiguous()

        batch_size = states.size(0)
        states_seq_size = states.size(1)

        sentence_max_length = lengths.max().item()

        states = states.view(batch_size*states_seq_size, *states.size()[-3:])

        conv_ht = self.conv_net(states)
        conv_ht = conv_ht.view(batch_size * states_seq_size, -1)
        conv_ht = self.project_after_conv(conv_ht)

        # hx size is (num_layers*num_directions, batch_size, hidden_size), 1 layer and 1 direction in this architecture
        conv_ht = conv_ht.unsqueeze(0) # Adding num_layer*directions dimension

        embedded_sentences = F.relu(self.word_embedder(teacher_sentence))
        embedded_sentences = self.dropout(embedded_sentences)

        if self.n_state_to_predict_instruction > 1:
            conv_ht = conv_ht.view(batch_size, states_seq_size, -1)
            all_ht, next_ht = self.trajectory_encoding(conv_ht)
            next_ht = next_ht[0]

            outputs = []

            for word_step in range(sentence_max_length):
                next_input = embedded_sentences[:, word_step, :]
                output, next_ht = self.rnn_decoder(next_input, next_ht, all_ht)
                next_ht = next_ht[0]
                outputs.append(output)

            ht = torch.cat(outputs, dim=1)

        else:
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(input=embedded_sentences,
                                                                      lengths=lengths,
                                                                      batch_first=True, enforce_sorted=False)

            # Compute all ht in a forward pass, teacher_forcing 100% of the time
            ht, _ = self.rnn_decoder(input=packed_embedded, hx=conv_ht)
            ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True) # Unpack sequence

        view_ht = ht.view(batch_size*sentence_max_length, -1)
        assert view_ht.size(1) == conv_ht.size(2)

        logits = F.relu(self.mlp_decoder_hidden(input=view_ht))
        logits = self.dropout(logits)
        logits = self.mlp_decoder(logits)

        return logits

    def pad(self, tokens):
        len_seq = len(tokens)
        tokens += [self.PAD_TOKEN] * (self.generator_max_len - len_seq)
        return tokens, len_seq

    def _generate(self, states_seq):

        states_seq = states_seq[-self.n_state_to_predict_instruction:]

        self.eval()
        with torch.no_grad():

            seq_len = states_seq.size(0)

            out = self.conv_net(states_seq)
            out = self.project_after_conv(out.view(seq_len, -1))
            last_ht = out.unsqueeze(0)

            if self.n_state_to_predict_instruction > 1:
                all_ht, last_ht = self.trajectory_encoding(last_ht.view(1, seq_len, -1))

            is_last_word = False
            next_token = torch.empty(1, 1).fill_(self.BEGIN_TOKEN).long().to(self.device)

            generated_token = []
            while not is_last_word:
                next_input = F.relu(self.word_embedder(next_token))
                ht, last_ht = self.rnn_decoder(next_input[0], last_ht[0], all_ht)
                hidden_activation = F.relu(self.mlp_decoder_hidden(ht[0]))
                logits_token = self.mlp_decoder(hidden_activation)
                next_token = logits_token.argmax(dim=1)

                if next_token == self.END_TOKEN or len(generated_token) > self.generator_max_len:
                    is_last_word = True
                else:
                    generated_token.append(next_token.item())

                next_token = next_token.unsqueeze(0)


        padded_mission, length_mission = self.pad(generated_token)
        return torch.LongTensor(padded_mission), length_mission



