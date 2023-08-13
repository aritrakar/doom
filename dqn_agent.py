# import copy
import random
from collections import deque

import numpy as np
import torch
from torch import nn


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(
                64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

        return x


class DQN_Agent:

    # def __init__(self, layer_sizes, learning_rate, sync_freq, exp_replay_size):
    def __init__(
            self,
            action_size,
            exp_replay_size,
            batch_size,  # number of experiences to sample from experience replay buffer
            discount_factor,  # discount factor for future rewards
            learning_rate,
            sync_freq,  # the number of experiences between every network synchronization
            epsilon=1):
        # Use GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Make the models
        self.q_net = DuelQNet(action_size).to(self.device)
        self.target_net = DuelQNet(action_size).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.q_net.parameters(), lr=learning_rate)

        self.action_size = action_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.discount_factor = discount_factor
        self.experience_replay = deque(maxlen=exp_replay_size)
        self.epsilon = epsilon

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def load_pretrained_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_net.state_dict(), model_path)

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().to(self.device))
        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(
            0, action_space_len, (1,))
        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)

        # Convert list of numpy ndarrays to a single numpy ndarray
        s = np.array([exp[0] for exp in sample], dtype=np.float32)
        a = np.array([exp[1] for exp in sample], dtype=np.float32)
        rn = np.array([exp[2] for exp in sample], dtype=np.float32)
        sn = np.array([exp[3] for exp in sample], dtype=np.float32)

        # Convert numpy ndarray to PyTorch tensor
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        rn = torch.tensor(rn).to(self.device)
        sn = torch.tensor(sn).to(self.device)
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.to(self.device))
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.to(self.device))
        target_return = rn.to(self.device) + self.discount_factor * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()
