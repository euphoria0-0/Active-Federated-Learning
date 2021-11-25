from FL_core.trainer import Trainer
from FL_core.active_learner import *
import numpy as np
from torch.utils.data import Subset


class Client(object):
    def __init__(self, client_idx, nTrain, local_train_data, local_test_data, model, args):
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)
        self.num_epoch = args.num_epoch # local epochs E

        nLabeled = int(args.labeled_ratio * nTrain)
        init_indices = np.random.choice(nTrain, nLabeled, replace=False).tolist()
        self.labeled_indices, self.unlabeled_indices = [], [*range(nTrain)]
        self.update(init_indices)

        self.nQuery = args.nQuery
        self.kwargs = {'unlabeled_indices': self.unlabeled_indices, 'unlabeled_dataset': self.local_unlabeled_data}
        if args.AL_method == 'MaxEntropy':
            self.active_learner = MaxEntropy(args, **self.kwargs)
        elif args.AL_method == 'Random':
            self.active_learner = RandomSelection(args, **self.kwargs)
        else:
            self.active_learner = None


    def train(self, global_model, tracking=True):
        self.trainer.set_model(global_model)
        if self.num_epoch == 0:
            acc, loss = self.trainer.train_E0(self.local_labeled_data, tracking)
        else:
            acc, loss = self.trainer.train(self.local_labeled_data, tracking)
        model = self.trainer.get_model()
        # local AL
        if self.active_learner is not None:
            self.active_learning(model)
        return model, acc, loss

    def test(self, model, mode='test'):
        if mode == 'train':
            acc, loss, auc = self.trainer.test(model, self.local_labeled_data)
        else:
            acc, loss, auc = self.trainer.test(model, self.local_test_data)
        return {'loss': loss, 'acc': acc, 'auc': auc}

    def get_client_idx(self):
        return self.client_idx

    def update(self, query_indices):
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))
        self.local_labeled_data = Subset(self.local_train_data, self.labeled_indices)
        self.local_unlabeled_data = Subset(self.local_train_data, self.unlabeled_indices)

    def active_learning(self, model):
        query_indices = self.active_learner.query(self.nQuery, model)
        self.update(query_indices)
        self.active_learner.update(**self.kwargs)
