


class Client(object):
    def __init__(self, client_idx, local_train_data, local_test_data):
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data

