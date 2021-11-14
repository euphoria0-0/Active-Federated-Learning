'''
Client Selection

Reference:
    ClusteredSampling
        https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
'''

import numpy as np


class ClientSelection:
    def __init__(self, total, device):
        self.total = total
        self.device = device

    def select(self, n, metric):
        pass



class RandomSelection(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, metric=None):
        selected_client_idxs = np.random.choice(self.total, size=n, replace=False)
        return selected_client_idxs




class ActiveFederatedLearning(ClientSelection):
    '''
    Active Federated Learning [1]
    '''
    def __init__(self, total, device, args):
        super().__init__(total, device)
        self.alpha1 = args.alpha1 #0.75
        self.alpha2 = args.alpha2 #0.01
        self.alpha3 = args.alpha3 #0.1

    def select(self, n, metric, seed=0):
        # set sampling distribution
        probs = np.exp(np.array(metric) * self.alpha2)
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





class ClusteredSampling1(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def setup(self, n, n_samples):
        '''
        Since clustering is performed according to the clients sample size n_i,
        unless n_i changes during the learning process,
        Algo 1 needs to be run only once at the beginning of the learning process.
        '''
        epsilon = int(10 ** 10)
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        weights = n_samples / np.sum(n_samples)
        # associate each client to a cluster
        augmented_weights = np.array([w * n * epsilon for w in weights])
        ordered_client_idx = np.flip(np.argsort(augmented_weights))

        distri_clusters = np.zeros((n, self.total)).astype(int)
        k = 0
        for client_idx in ordered_client_idx:
            while augmented_weights[client_idx] > 0:
                sum_proba_in_k = np.sum(distri_clusters[k])
                u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])
                distri_clusters[k, client_idx] = u_i
                augmented_weights[client_idx] += -u_i
                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

        distri_clusters = distri_clusters.astype(float)
        for l in range(n):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        self.distri_clusters = distri_clusters


    def select(self, n, metric=None):
        # sample clients
        selected_client_idxs = np.zeros(n, dtype=int)
        for k in range(n):
            selected_client_idxs[k] = int(np.random.choice(self.total, 1, p=self.distri_clusters[k]))
        return selected_client_idxs






'''def modified_exp(x, SAFETY=2.0):
    mrn = np.finfo(x.dtype).max
    threshold = np.log(mrn / x.size) - SAFETY
    xmax = x.max()
    if xmax > threshold:
        return np.exp(x - (xmax - threshold))
    else:
        return np.exp(x)'''