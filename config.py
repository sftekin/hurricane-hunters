import itertools
from random import shuffle


model_params_pool = {
    "lstm": {
        "batch_size": [32, 128],
        "shuffle": [True],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "num_epochs": [100],
        "loss_type": ["l2"],
        "optimizer_type": ["adam"],
        "grad_clip": [1],
        "l2_reg": [0, 1e-4],
        "dropout_rate": [0, 0.1],
        "early_stop_tolerance": [5],
        "final_act_type": ["leaky_relu"],
        "relu_alpha": [1],
        "window_len_input": [10],
        "window_len_output": [10],
        "stride": [1],
        "hidden_dim_list": [[8, 8], [8, 8, 8], [32, 32]],
        "norm_method": ["standard"],
    },
    "trajgru": {}
}


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_params = model_params_pool[self.model_name]

        self.conf_list = self.create_params_list(dict(**self.model_params))
        self.num_confs = len(self.conf_list)

        self.experiment_params = {
            "num_works": 1,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "input_dim": list(range(7)),
            "output_dim": list(range(2)),
        }

        self.data_params = {
            "season_range": (1994, 2020)
        }

    def next(self):
        for conf in self.conf_list:
            yield conf

    @staticmethod
    def create_params_list(pool):
        params_list = []
        keys = pool.keys()
        lists = [l for l in pool.values()]
        all_lists = list(itertools.product(*lists))
        for i in range(len(all_lists)):
            params_list.append(dict(zip(keys, all_lists[i])))
        shuffle(params_list)

        return params_list



