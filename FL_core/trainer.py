import torch.nn as nn
import torch.optim as optim
import torch


class Trainer:
    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.client_optimizer = args.client_optimizer
        self.lr = args.lr
        self.wdecay = args.wdecay
        self.num_epoch = args.num_epoch

    def get_model(self):
        return self.model.cpu().state_dict()

    def set_model(self, model):
        self.model = model

    def train(self, data):
        model = self.model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay, amsgrad=True)

        for epoch in range(self.num_epoch):
            train_loss, correct, total = 0, 0, 0
            for input, labels in data:
                input, labels = input.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                train_loss += torch.sum(loss)
                correct += torch.sum(preds == labels.data)
                total += input.size(0)

                loss.backward()
                optimizer.step()

        return correct / total, train_loss / total

    def test(self, data):
        model = self.model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for epoch in range(self.num_epoch):
                test_loss, correct, total = 0, 0, 0
                for input, labels in data:
                    input, labels = input.to(self.device), labels.to(self.device)
                    output = model(input)
                    loss = criterion(output, labels.long())
                    _, preds = torch.max(output.data, 1)

                    test_loss += torch.sum(loss)
                    correct += preds.eq(labels).sum()
                    total += input.size(0)

        return correct / total, test_loss / total
