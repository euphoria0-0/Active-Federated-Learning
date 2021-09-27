import torch
import wandb

from FL_core.client import Client
from FL_core.trainer import Trainer
from FL_core.client_selection import ClientSelection, ActiveFederatedLearning


class Server(object):
    def __init__(self, data, device, method='random'):
        train_data, test_data, train_sizes, test_sizes = data
        self.train_data = train_data
        self.train_sizes = train_sizes
        self.test_data = test_data
        self.test_sizes = test_sizes
        self.method = method
        self.device = device

        self.client_list = []
        self.total_num_clients = 7527
        self.num_clients_per_round = 200
        self.total_round = 20

        self.trainer = Trainer()

        self._init_clients()

    def _init_clients(self):
        for client_idx in range(self.total_num_clients):
            c = Client(client_idx, self.train_data[client_idx], self.test_data[client_idx], self.device)
            self.client_list.append(c)

    def _client_selection(self, method='AFL'):
        if method == 'AFL':
            selection_method = ActiveFederatedLearning
        return selection_method

    def _aggregate_local_models(self, local_models, client_indices, method='FedAvg'):
        if method == 'FedAvg':
            global_model = local_models[0].keys()
            for k in local_models[0].keys():
                for idx in range(1,len(local_models)):
                    local_model = local_models[idx]
                    num_local_data = self.train_sizes[client_indices]
                    weight = num_local_data / sum(self.train_sizes)
                    global_model[k] += weight * local_model[k]

        return global_model

    def train(self):
        # get global model
        global_model = self.trainer.get_model()
        for round_idx in range(self.total_round):
            # set clients
            if self.method == 'random':
                client_indices = torch.randint(self.total_num_clients, (self.num_clients_per_round,)).tolist()
            else:
                client_indices = [*range(self.total_num_clients)]

            # local training
            local_models, local_losses = [], []
            for client_idx in client_indices:
                client = self.client_list[client_idx]
                local_model, local_loss = client.train(global_model)
                local_models.append(local_model)
                local_losses.append(local_loss)

            # client selection
            if self.method != 'random':
                selection_method = self._client_selection()
                selection_method = selection_method(local_models, local_losses,
                                                    self.total_num_clients, self.num_clients_per_round, self.device)
                selected_client_indices = selection_method.select(round_idx)
                local_models = local_models[selected_client_indices]
                client_indices = client_indices[selected_client_indices]

            # update global model
            global_model = self._aggregate_local_models(local_models, client_indices)
            self.trainer.set_model(global_model)

            # test
            self._test(round_idx)


    def _test(self, round_idx):
        for mode in ['Train', 'Test']:
            datasize = sum(self.train_sizes) if mode == 'Train' else sum(self.test_sizes)
            metrics = {'loss': [], 'acc': []}
            for client_idx in range(self.total_num_clients):
                client = self.client_list[client_idx]
                result = client.test(mode.lower())
                metrics['loss'].append(result['loss'].item())
                metrics['acc'].append(result['acc'].item())

            '''wandb.log({
                mode+'/Loss': sum(metrics['loss']) / datasize,
                mode+'/Acc': sum(metrics['acc']) / datasize
            })'''