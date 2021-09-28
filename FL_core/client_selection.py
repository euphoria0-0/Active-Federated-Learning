import numpy as np


class ClientSelection:
    def __init__(self, model, metric, total, budget, device):
        self.model = model
        self.metric = metric
        self.total = total
        self.budget = budget
        self.device = device

    def select(self, n):
        pass

    def update(self, selected_indices):
        self.model = self.model[selected_indices]



class ActiveFederatedLearning(ClientSelection):
    def __init__(self, model, metric, total, budget, device):
        super().__init__(model, metric, total, budget, device)
        self.alpha1 = 0.75
        self.alpha2 = 0.01
        self.alpha3 = 0.1

    def select(self, seed=0):
        '''print(f'total users: {self.total}')
        # 1) select 75% of K(total) users
        num_selecting_users = int(self.alpha1 * self.total)
        argsorted_value_list = np.argsort(np.array(self.metric))
        selected_client_idxs = argsorted_value_list[-num_selecting_users:]
        value_list = np.array(self.metric)[selected_client_idxs]
        print(f'75% of K users: {value_list.shape}')
        # 2) select 99% of m users using prob.
        num_selecting_users = int((1 - self.alpha3) * self.budget)
        argsorted_value_list = np.argsort(value_list)
        selected_client_idxs1 = argsorted_value_list[-num_selecting_users:]
        # 3) select 1% of m users randomly
        num_selecting_users = int(self.alpha3 * self.budget)
        selected_client_idxs2 = argsorted_value_list[:num_selecting_users]
        np.random.seed(seed)  # make sure for each comparison, we are selecting the same clients each round
        selected_client_idxs3 = np.random.choice(selected_client_idxs2,
                                                 num_selecting_users, replace=False)
        # selected_client_idxs = selected_client_idxs1 + selected_client_idxs3
        selected_client_idxs = np.append(selected_client_idxs1, selected_client_idxs3)
        print(f'selected users: {selected_client_idxs.shape}')'''
        # set sampling distribution
        probs = np.array([np.exp(self.alpha2 * v) for v in self.metric])
        # 1) select 75% of K(total) users
        num_select = int(self.alpha1 * self.total)
        argsorted_value_list = np.argsort(np.array(self.metric))
        client_idxs = argsorted_value_list[-num_select:]
        probs[client_idxs] = 0
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * self.budget)
        selected = np.random.choice(self.total, num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(set(np.arange(self.total)) - set(selected))
        selected2 = np.random.choice(not_selected, self.budget - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'selected users: {selected_client_idxs.shape}')
        return selected_client_idxs
