import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
import os
from helper import *
import SL_train

# define image transform
transform = transforms.Resize(size=(40, 40))
PUNISHMENT = [1, -1, 0]  # list of punishment according to SL model prediction class
# PUNISHMENT = [0,0,0] # using this line to play with only DQN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
DQN model
'''


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


def Res_QNet(out_features):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, out_features)
    return model_ft


class QTrainer:
    def __init__(self, model, lr, gamma, slmodel):
        self.lr = lr
        self.gamma = gamma
        self.slmodel = slmodel.to(device)
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, game, n_game):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:

            # (1, x)
            state = torch.unsqueeze(state, 0).to(device)
            next_state = torch.unsqueeze(next_state, 0).to(device)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

            pred = self.model(state)

            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx].to(device)
                if not done[idx]:
                    image = game.get_step_map(action[idx])  # get game playground with current action
                    image = transform(image)  # compress map for sl model
                    batch = torch.unsqueeze(image, 0).type(torch.double).to(device)
                    out_sl = self.slmodel(batch).to('cpu')  # get prediction from sl model
                    _, target_sl = torch.max(out_sl, 1)  # convert to class 0: rewards, 1:punish, 2:nothing
                    punish = PUNISHMENT[target_sl]  # find correct punishment value
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) + punish # update Q score

                target[idx][torch.argmax(action[idx]).item()] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
