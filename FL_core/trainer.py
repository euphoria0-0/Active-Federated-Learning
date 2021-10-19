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
        self.num_epoch = args.num_epoch   # num of local epoch E
        self.batch_size = args.batch_size # local batch size B


    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model.load_state_dict(model.cpu().state_dict())

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
                input, labels = input.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                train_loss += loss.item()
                correct += torch.sum(preds == labels.data)
                total += input.size(0)

                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()

            train_acc = correct / total
            train_loss = train_loss / total
            sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch+1, self.num_epoch,
                                                                                     train_loss, train_acc))

        if tracking:
            print()
        self.model = model

        return train_acc, train_loss

    def train_E0(self, data, tracking=True):
        model = self.model
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        correct, total = 0, 0
        batch_loss = []
        for input, labels in data:
            input, labels = input.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            batch_loss.append(loss) ##### loss sum
            total += input.size(0)
            correct += torch.sum(preds == labels.data)

            torch.cuda.empty_cache()

        train_acc = correct / total
        avg_loss = sum(batch_loss) / total

        avg_loss.backward()
        optimizer.step()

        sys.stdout.write('\rTrainLoss {:.6f} TrainAcc {:.4f}'.format(avg_loss, train_acc))

        if tracking:
            print()
        self.model = model

        return train_acc, avg_loss.cpu().detach()

    def test(self, model, data):
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            y_true, y_score = np.empty((0)), np.empty((0))
            for input, labels in data:
                input, labels = input.to(self.device), labels.to(self.device)
                output = model(input)
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += loss.item()
                correct += preds.eq(labels).sum()
                total += input.size(0)

                y_true = np.append(y_true, labels.cpu().numpy(), axis=0)
                y_score = np.append(y_score, preds.cpu().numpy(), axis=0)

                torch.cuda.empty_cache()

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
