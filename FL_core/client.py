from FL_core.trainer import Trainer


class Client(object):
    def __init__(self, client_idx, local_train_data, local_test_data, model, args):
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)

    def train(self, global_model, tracking=True):
        self.trainer.set_model_params(global_model)
        model, acc, loss = self.trainer.train(self.local_train_data, tracking)
        return model, acc, loss

    def test(self, mode='test'):
        if mode == 'train':
            acc, loss = self.trainer.test(self.local_train_data)
        else:
            acc, loss = self.trainer.test(self.local_test_data)
        return {'loss': loss, 'acc': acc}
