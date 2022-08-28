

import paddle.fluid as fluid


class Model(object):
    def __init__(self, save_path, data_class, lr, seed=2022):
        self.save_path = save_path
        self.cost = None
        self.data_class = data_class
        self.seed = seed
        self.metrics = dict()
        self.lr = lr

    def get_inputs(self):
        return self.data_class.get_data_var()

    def get_label(self):
        return self.data_class.get_label()

    def get_hash_size(self):
        return self.data_class.get_hash_size()

    def get_metrics(self):
        return self.metrics

    def get_cost(self):
        return self.cost

    def build_optimizer(self, name):
        name = name.upper()
        optimizers = ["SGD", "ADAM", "ADAGRAD"]
        if name not in optimizers:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")
        if name == "SGD":
            optimizer = fluid.optimizer.SGD(self.lr)
        elif name == "ADAM":
            optimizer = fluid.optimizer.Adam(self.lr, lazy_mode=True)
        elif name == "ADAGRAD":
            optimizer = fluid.optimizer.Adagrad(self.lr)
        else:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")
        return optimizer

    def net(self, is_infer=False):
        return None






