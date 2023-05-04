import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
import pickle
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

root_path = 'C:/Users/panxi/PycharmProjects/NeuroScience/650/'
save_path = 'data/'
path = 'data/train_data.pkl'
epochs = 50
criterion = nn.CrossEntropyLoss()
output_feature = 3
duplicate_rate = 2

'''
SL model
'''


class SLnet(nn.Module):
    def __init__(self, input_channel, output_feature):
        super(SLnet, self).__init__()
        self.input_channel = input_channel
        self.output_feature = output_feature
        self.conv1 = nn.Conv2d(self.input_channel, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3200, 256)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, self.output_feature)  # 4 possible actions: move left, right, up, or down

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

    def get_name(self):
        return 'SLMODEL'


def combine_pkl(path):
    res = []
    for file in os.listdir(path):
        # check only text files
        if file.endswith('.pkl'):
            res.append(file)
    print(res)
    data_list = []
    for r in res:
        with open(os.path.join(path,r),'rb') as f:
            data = pickle.load(f)
            data_list = data_list + data
            f.close()
    return data_list


def collect_data(root_path, path):
    data = combine_pkl(os.path.join(root_path,path))
    df = pd.DataFrame.from_records(data)
    print(df['ifend'].value_counts())
    temp_df = []
    for row in df.itertuples(index=False):
        if row.ifend == 1:
            temp_df.extend([list(row)] * duplicate_rate)
        else:
            temp_df.append(list(row))

    df = pd.DataFrame(temp_df, columns=df.columns)
    print(df.describe())
    print(df['ifend'].value_counts())
    data = df.to_dict('records')

    return data


class DATASET():
    def __init__(self, data_list):
        self.data_list = data_list
        self.transform = transforms.Compose([
            transforms.Resize(50),
            transforms.CenterCrop(40),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = copy.deepcopy(self.data_list[idx])
        d['image'] = self.transform(d['image'])
        return d


def load_dataset(dataset, valid_ratio, test_ratio, bs):
    valid_size = int(valid_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - valid_size - test_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=bs, shuffle=True,
                             num_workers=0) if valid_dataset is not None else []
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=True,
                            num_workers=0) if test_dataset is not None else None

    return trainloader, validloader, testloader


def define_dataset(trial):
    batch_size = trial.suggest_int("batch_size", 3, 10)
    data_list = collect_data(root_path, save_path)
    dataset = DATASET(data_list)
    train_iter, valid_iter, test_iter = load_dataset(dataset, 0.2, 0.1, batch_size)

    return train_iter, valid_iter, test_iter


def objective(trial):
    # Generate the model.
    train_iter, valid_iter, test_iter = define_dataset(trial)
    model = SLnet(3, output_feature).type(torch.double).to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []

    for epoch in range(epochs):

        t_loss, t_acc, v_loss, v_acc = train_val_per_epoch(model, optimizer, train_iter, valid_iter)

        train_loss.append(t_loss)
        train_acc.append(t_acc)
        validation_loss.append(v_loss)
        validation_acc.append(v_acc)

        print(f'Epoch: {epoch + 1},  Training Loss: {t_loss:.4f}, Training Accuracy: {100 * t_acc: .2f}%')

        print(
            f'Validation Loss: {v_loss:.4f}, Validation Accuracy: {100 * v_acc: .2f}%')
        report = v_loss  # objective function for optimizing tuning
        trial.report(report, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return report


def train_val_per_epoch(model, optimizer, train_iter, valid_iter):
    # train
    model.train()
    running_loss = 0.
    correct, total = 0, 0
    train_loss, validation_loss = 0, 0
    train_acc, validation_acc = 0, 0
    steps = 0

    for idx, data in enumerate(train_iter, 0):
        text = data['image'].type(torch.double)
        target = data['ifend']
        target = torch.autograd.Variable(target).type(torch.int64)
        text, target = text.to(device), target.to(device)

        # add micro for coding training loop
        optimizer.zero_grad()
        output = model(text)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        steps += 1
        running_loss += loss.item()

        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_loss = running_loss / len(train_iter)
    train_acc = correct / total

    # evaluate on validation data
    model.eval()
    running_loss = 0.
    correct, total = 0, 0

    with torch.no_grad():
        for idx, data in enumerate(valid_iter, 0):
            text = data['image'].type(torch.double)
            target = data['ifend']
            target = torch.autograd.Variable(target).type(torch.int64)
            text, target = text.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(text)

            loss = criterion(output, target)
            running_loss += loss.item()

            # get accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        validation_loss = running_loss / len(valid_iter)
        validation_acc = correct / total

    return train_loss, train_acc, validation_loss, validation_acc


def detailed_objective(trial):
    # Generate the model.
    model = SLnet(3, 4).type(torch.double).to(device)
    train_iter, valid_iter, test_iter = define_dataset(trial)
    best_model = None

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    valid_loss_min = torch.inf

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.
        correct, total = 0, 0
        steps = 0

        for idx, data in enumerate(train_iter, 0):
            text = data['image'].type(torch.double)
            target = data['ifend']
            target = torch.autograd.Variable(target).type(torch.int64)
            text, target = text.to(device), target.to(device)

            # add micro for coding training loop
            optimizer.zero_grad()
            output = model(text)
            # print(output.shape, target.shape)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            steps += 1
            running_loss += loss.item()

            # get accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss.append(running_loss / len(train_iter))
        train_acc.append(correct / total)
        t_loss = running_loss / len(train_iter)

        print(
            f'Epoch: {epoch + 1},  Training Loss: {running_loss / len(train_iter):.4f}, Training Accuracy: {100 * correct / total: .2f}%')

        # evaluate on validation data
        model.eval()
        running_loss = 0.
        correct, total = 0, 0

        with torch.no_grad():
            for idx, data in enumerate(valid_iter, 0):
                text = data['image'].type(torch.double)
                target = data['ifend']
                target = torch.autograd.Variable(target).type(torch.int64)
                text, target = text.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(text)

                loss = criterion(output, target)
                running_loss += loss.item()

                # get accuracy
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(
            f'Validation Loss: {running_loss / len(valid_iter):.4f}, Validation Accuracy: {100 * correct / total: .2f}%')

        validation_loss.append(running_loss / len(valid_iter))
        validation_acc.append(correct / total)

        valid_loss = running_loss / len(valid_iter)

        report = t_loss  # objective function for optimizing tuning
        trial.report(report, epoch)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss
            best_model = model

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_model, train_acc, validation_acc


if __name__ == "__main__":
    import pandas as pd

    data_list = collect_data(root_path, save_path)

    study = optuna.create_study(direction="minimize")  # maximize valid accuracy
    study.optimize(objective, n_trials=50, timeout=3000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    df = study.trials_dataframe()
    os.makedirs('data', exist_ok=True)

    model, train_acc, valid_acc = detailed_objective(study.best_trial)
    torch.save(model, save_path + "best_model.pt")
    model_name = model.get_name()

    with open('data/' + model_name + '_' + 'param.txt', 'w') as f:
        f.write(f"best model for {model_name}: \n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write("    {}: {}\n".format(key, value))
        f.write(f'Train Accuracy: {train_acc[-1]} %\n')
        f.write(f'Validation Accuracy: {valid_acc[-1]} %\n')
