import numpy as np


class ClientSelection:
    def __init__(self, total, device):
        self.total = total
        self.device = device

    def select(self, n, metric):
        pass



class ActiveFederatedLearning(ClientSelection):
    def __init__(self, total, device, args):
        super().__init__(total, device)
        self.alpha1 = args.alpha1 #0.75
        self.alpha2 = args.alpha2 #0.01
        self.alpha3 = args.alpha3 #0.1

    def select(self, n, metric, seed=0):
        # set sampling distribution
        probs = np.array(metric) * np.exp(self.alpha2)
        # 1) select 75% of K(total) users
        num_select = int(self.alpha1 * self.total)
        argsorted_value_list = np.argsort(metric)
        drop_client_idxs = argsorted_value_list[:self.total - num_select]
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        #probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        np.random.seed(seed)
        selected = np.random.choice(self.total, num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(self.total)) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')
        return selected_client_idxs.astype(int)


def modified_exp(x, SAFETY=2.0):
    mrn = np.finfo(x.dtype).max
    threshold = np.log(mrn / x.size) - SAFETY
    xmax = x.max()
    if xmax > threshold:
        return np.exp(x - (xmax - threshold))
    else:
        return np.exp(x)