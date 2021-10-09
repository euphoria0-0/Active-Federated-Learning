import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.client_optimizer = args.client_optimizer
        self.lr = args.lr_local
        self.wdecay = args.wdecay
        self.num_epoch = args.num_epoch


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)

    def train(self, data, tracking=True):
        model = self.model
        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        for epoch in range(self.num_epoch):
            train_loss, correct, total = 0., 0, 0
            for input, labels in data:
                print(input.size(), labels.size())
            for input, labels in data:
                print(input.size())
                input, labels = input.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(input)
                #loss = criterion(output, labels.unsqueeze(1).type_as(output))
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                train_loss += torch.sum(loss)
                correct += torch.sum(preds == labels.data)
                total += input.size(0)

                loss.backward()
                optimizer.step()

            train_acc = correct / total
            train_loss = train_loss / np.sqrt(total)
            sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch+1, self.num_epoch,
                                                                                     train_loss, train_acc))
        if tracking:
            print()
        self.model = model

        return model, train_acc, train_loss.cpu().detach()

    def test(self, data):
        model = self.model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            y_true, y_score = np.empty((0)), np.empty((0))
            for input, labels in data:
                input, labels = input.to(self.device), labels.to(self.device)
                output = model(input)
                #loss = criterion(output, labels.unsqueeze(1).type_as(output))
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += loss.item()
                correct += preds.eq(labels).sum()
                total += input.size(0)

                y_true = np.append(y_true, labels.cpu().numpy(), axis=0)
                y_score = np.append(y_score, preds.cpu().numpy(), axis=0)

        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0

        if total > 0:
            test_loss /= total
            test_acc = correct / total
        else:
            test_acc = correct
        return test_acc, test_loss, auc
