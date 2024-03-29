import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.ln3 = nn.LayerNorm(hidden_size)
        self.action_out = nn.Linear(hidden_size, output_size)
        # self.apply(weight_init)

        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                # m.weight.data.mul_(0.1)
                # m.bias.data.zero_()

    def forward(self, x):
        # x = F.softplus(self.ln1(self.fc1(x)))
        # x = F.softplus(self.ln2(self.fc2(x)))
        # x = F.softplus(self.ln3(self.fc3(x)))
        x = F.softplus(self.fc1(x))
        F.softplus
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        actions = torch.tanh(self.action_out(x))

        return actions


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_scores, value)
        return attended_values


class Critic(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_size + act_size, hidden_size)
        self.attention = SelfAttention(obs_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.ln3 = nn.LayerNorm(hidden_size)
        self.q_out = nn.Linear(hidden_size, 1)
        # self.apply(weight_init)

        # self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        state = self.attention(state)
        x = torch.cat([state, action], dim=1)
        # x = F.softplus(self.ln1(self.fc1(x)))
        # x = F.softplus(self.ln2(self.fc2(x)))
        # x = F.softplus(self.ln3(self.fc3(x)))
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        q_value = self.q_out(x)
        return q_value