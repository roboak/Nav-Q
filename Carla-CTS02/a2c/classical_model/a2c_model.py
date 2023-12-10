import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linetimer import CodeTimer

class SharedNetwork(nn.Module):
    def __init__(self, dim_red=False, hidden_dim=256):
        super(SharedNetwork, self).__init__()

        # input_shape = [None, 400, 400, 3]
        self.inp_norm = nn.LayerNorm([400, 400])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4)) #[1, 32, 99, 99]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)) #[1,64, 48, 48]
        self.layer_norm2 = nn.LayerNorm([64, 48, 48])
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)) #[1, 64, 46, 46]
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(9, 9), stride=(3, 3)) #[1, 128, 13, 13]
        self.layer_norm4 = nn.LayerNorm([128, 13, 13])
        if not dim_red:
            self.conv5 = nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 1)) #[1, 128, 5, 5]
            self.conv6 = nn.Conv2d(128, hidden_dim, kernel_size=(5, 5), stride=(1, 1)) #[1, hidden_dim, 1, 1]
        else:
            self.conv5 = nn.Identity()
            self.conv6 = nn.Conv2d(128, hidden_dim, kernel_size=(6, 1), stride=(1, 1))
        # self.pool = nn.AdaptiveMaxPool2d((1, 1, hidden_dim))
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv4.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv5.weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv6.weight, np.sqrt(2))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_norm1 = nn.LayerNorm([hidden_dim])
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(hidden_dim + 4, hidden_dim)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, obs, cat_tensor):
        obs, (hx, cx) = obs
        x = self.inp_norm(obs)
        # print(obs.size())
        x = self.relu(self.conv1(obs))
        x = self.relu(self.layer_norm2(self.conv2(x)))
        # x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.layer_norm4(self.conv4(x)))
        # x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        # print(x.size())
        x = self.relu(self.conv6(x))
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc_norm1(self.fc1(x)))
        # x = self.relu(self.fc1(x))
        x = torch.cat((x, cat_tensor), dim=1)
        hx, cx = self.lstm(x, (hx, cx))
        return hx, cx


class SharedNetworkScaled(nn.Module):
    def __init__(self, dim_red=False, hidden_dim=256):
        super(SharedNetworkScaled, self).__init__()

        # input_shape = [None, 400, 400, 3]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        # self.conv5 = nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 1))
        self.conv5 = nn.Conv2d(128, hidden_dim, kernel_size=(3, 3), stride=(1, 1))
        # self.pool = nn.AdaptiveMaxPool2d((1, 1, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(hidden_dim + 4, hidden_dim)

    def forward(self, obs, cat_tensor):
        obs, (hx, cx) = obs
        # print(obs.size())
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        # print(x.size())
        x = self.relu(self.conv6(x))
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = torch.cat((x, cat_tensor), dim=1)
        # print("debug__________________")
        # print("in device: ", x.device, hx.device, cx.device)
        # print("model device: ",self.lstm.weight_hh.device)
        hx, cx = self.lstm(x, (hx, cx))
        return hx, cx

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.layer_norm1 = nn.LayerNorm([64])
        self.fc2 = nn.Linear(64, 1)
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_norm1(self.fc1(x)))
        # x = self.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class ActionNetwork(nn.Module):
    def __init__(self, input_dim=256, num_actions=9):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.layer_norm1 = nn.LayerNorm([64])
        self.fc2 = nn.Linear(64, num_actions)
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_norm1(self.fc1(x)))
        # x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class A2C(nn.Module):
    def __init__(self, dim_red=False, hidden_dim=256, num_actions=3, inp_scaling=False, model_eval=False):
        super(A2C, self).__init__()
        self.model_eval = model_eval
        # if self.scaling:
        #     self.shared_network = SharedNetwork_for_downscaled_image(hidden_dim=hidden_dim, dim_red=dim_red)
        # else:
        if inp_scaling:
            self.shared_network = SharedNetworkScaled(hidden_dim=hidden_dim, dim_red=dim_red)
        else:
            self.shared_network = SharedNetwork(hidden_dim=hidden_dim, dim_red=dim_red)
        # self.value_network = ValueNetwork(hidden_dim)
        self.action_policy = ActionNetwork(hidden_dim, num_actions)
        if not model_eval:
            self.value_network = ValueNetwork(hidden_dim)


    def forward(self, x, lstm_state, cat_tensor):
        x =x.permute(2, 0, 1)[None, :]
            # torch.reshape(x, (-1, 3, 400, 400))
        value = None
        cat_tensor = torch.reshape(cat_tensor, (-1, 4))
        obs = (x, lstm_state)
        # with CodeTimer("Shared Network"):
        features, cx = self.shared_network(obs, cat_tensor)
        if not self.model_eval:
            # with CodeTimer("Value Network"):
            value = self.value_network(features)
        # with CodeTimer("Actor Network"):
        action = self.action_policy(features)
        return action, value, (features, cx)


# class A2CGym(nn.Module):
#     def __init__(self, input_dim, num_actions, hidden_dim):
#         super(A2CGym, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.relu = nn.ReLU()
#         self.value_network = nn.Linear(128, 1)
#         self.action_policy = nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = torch.reshape(x, (-1, 4))
#         x = F.relu(self.fc1(x))
#         value = self.value_network(x)
#         action = self.action_policy(x)
#         return action, value
