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
        "weather_info": [False]
    },
    "trajgru": {
        "input_size": [(25, 25)],
        "input_dim": [15],
        "output_dim": [1],
        "window_in": [10],
        "window_out": [10],
        "regression": ["linear"],
        "loss_type": ["MSE"],
        "encoder_conf": [{
            "en_num_layers": 2,
            "en_conv_dims": [16, 64],
            "en_conv_kernel": 3,
            "en_conv_stride": 1,
            "en_pool_kernel": 3,
            "en_pool_stride": 2,
            "en_pool_padding": 0,
            "en_gru_dims": [32, 96],
            "en_gru_kernels": [5, 3],
            "en_connection": 5,
            "en_bias": True
        }],
        "decoder_conf": [{
            "de_input_dim": 96,
            "de_num_layers": 2,
            "de_conv_dims": [64, 16],
            "de_conv_kernel": 3,
            "de_conv_stride": 2,
            "de_conv_padding": 0,
            "de_gru_dims": [96, 32],
            "de_gru_kernels": [3, 3],
            "de_connection": 5,
            "de_bias": True
        }],
        "output_conv_dims": [[16, 16]],
        "output_conv_kernels": [[5, 1]],

        # finetune params
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
        "weather_info": [False]
    }
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
            "hur_input_dim": list(range(7)),
            "weather_input_dim": list(range(5)),
            "hur_output_dim": list(range(2)),
            "return_mode": 'weather'
        }

        self.data_params = {
            "season_range": (2015, 2020),
            "weather_im_size": (25, 25),
            "weather_freq": 3,
            "weather_spatial_range": [[0, 65], [-110, 10]],
            "weather_raw_dir": 'data/weather_raw',
            "rebuild": False
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



