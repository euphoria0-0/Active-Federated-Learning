from FL_core.client_trainer import ClientTrainer


class Client(object):
    def __init__(self, client_idx, local_train_data, local_test_data, model, args):
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data
        self.device = args.device
        self.trainer = ClientTrainer(model, args)
        self.num_epoch = args.num_epoch # local epoch

    def train(self, global_model, tracking=True):
        self.trainer.set_model_params(global_model)
        if self.num_epoch == 0:
            acc, loss = self.trainer.train_E0(self.local_train_data, tracking)
        else:
            acc, loss = self.trainer.train(self.local_train_data, tracking)
        model = self.trainer.get_model_params()
        return model, acc, loss

    def test(self, model, mode='test'):
        if mode == 'train':
            acc, loss, auc = self.trainer.test(model, self.local_train_data)
        else:
            acc, loss, auc = self.trainer.test(model, self.local_test_data)
        return {'loss': loss, 'acc': acc, 'auc': auc}

    def get_client_idx(self):
        return self.client_idx
