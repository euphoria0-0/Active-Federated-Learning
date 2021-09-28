import sys
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.client_optimizer = args.client_optimizer
        self.lr = args.lr
        self.wdecay = args.wdecay
        self.num_epoch = args.num_epoch

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_params):
        self.model.load_state_dict(model_params)

    def train(self, data):
        model = self.model
        model = model.to(self.device)
        model.train()

        criterion = nn.BCEWithLogitsLoss().to(self.device)
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
                loss = criterion(output, labels.unsqueeze(1).type_as(output))
                _, preds = torch.max(output.data, 1)

                train_loss += torch.sum(loss)
                correct += torch.sum(preds == labels.data)
                total += input.size(0)

                loss.backward()
                optimizer.step()

            train_acc = correct / total
            train_loss = train_loss / total
            sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch+1, self.num_epoch,
                                                                                     train_loss, train_acc))
        print()

        return model, train_acc, train_loss

    def test(self, data):
        model = self.model.to(self.device)
        model.eval()

        criterion = nn.BCEWithLogitsLoss().to(self.device)

        with torch.no_grad():
            for epoch in range(self.num_epoch):
                test_loss, correct, total = 0, 0, 0
                for input, labels in data:
                    input, labels = input.to(self.device), labels.to(self.device)
                    output = model(input)
                    loss = criterion(output, labels.unsqueeze(1).type_as(output))
                    _, preds = torch.max(output.data, 1)

                    test_loss += torch.sum(loss)
                    correct += preds.eq(labels).sum()
                    total += input.size(0)

        test_loss = test_loss / total
        test_acc = correct / total
        print('TestLoss {:.6f} TestAcc {:.4f}'.format(test_loss, test_acc))
        return test_acc, test_loss
