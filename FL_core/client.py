
from FL_core.trainer import Trainer


class Client(object):
    def __init__(self, client_idx, local_train_data, local_test_data, device):
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data
        self.device = device
        self.trainer = Trainer(self.device)

    def train(self, global_model):
        self.trainer.set_model(global_model)
        model, loss = self.trainer.train()
        self.model, self.loss = model, loss
        return model, loss

    def test(self, mode='test'):
        if mode == 'train':
            loss, acc = self.trainer.test(self.local_train_data)
        else:
            loss, acc = self.trainer.test(self.local_test_data)
        return {'loss': loss, 'acc': acc}
