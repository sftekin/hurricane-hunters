import itertools
from random import shuffle


model_params_pool = {
    "lstm": {
        "learning_rate": [2e-3],
        "num_epochs": [30],
        "loss_type": ["l2"],
        "optimizer_type": ["adam"],
        "grad_clip": [10],
        "l2_reg": [0],
        "dropout_rate": [0],
        "early_stop_tolerance": [5],
        "final_act_type": ["leaky_relu"],
        "relu_alpha": [0.01],
        "hidden_dim_list": [[8, 8]],
        "input_norm_method": ["minmax"],
        "output_norm_method": ["minmax"],
        "batch_size": [16],
        "stride": [1],
        "window_len": [3],
        "phase_shift": [1],
        "return_mode": ["hurricane"],
        "cut_start": [True],
        "shuffle": [True]
    },
    "trajgru": {
        "input_size": [(25, 25)],
        "en_dec_output_dim": [1],
        "window_in": [10],
        "window_out": [10],
        "regression": ["linear"],
        "loss_type": ["MSE"],
        "encoder_conf": [{
            "en_num_layers": 2,
            "en_conv_dims": [16, 32],
            "en_conv_kernel": 3,
            "en_conv_stride": 1,
            "en_pool_kernel": 3,
            "en_pool_stride": 2,
            "en_pool_padding": 0,
            "en_gru_dims": [32, 64],
            "en_gru_kernels": [5, 3],
            "en_connection": 5,
            "en_bias": True
        }],
        "decoder_conf": [{
            "de_input_dim": 64,
            "de_num_layers": 2,
            "de_conv_dims": [32, 16],
            "de_conv_kernel": 3,
            "de_conv_stride": 2,
            "de_conv_padding": 0,
            "de_gru_dims": [64, 32],
            "de_gru_kernels": [3, 3],
            "de_connection": 5,
            "de_bias": True
        }],
        "output_conv_dims": [[16, 16]],
        "output_conv_kernels": [[5, 1]],
        "relu_alpha": [1],
        "stateful": [False],
        "clip": [5],
        # finetune params
        "learning_rate": [1e-3],
        "num_epochs": [200],
        "loss_type": ["l2"],
        "optimizer_type": ["adam"],
        "grad_clip": [1],
        "l2_reg": [0],
        "dropout_rate": [0],
        "early_stop_tolerance": [5],
        "final_act_type": ["leaky_relu"],
        "norm_method": ["minmax"],
        # batch gen params
        "batch_size": [8],
        "stride": [0],
        "window_len": [10],
        "return_mode": ['weather'],
        "phase_shift": [1],
        "cut_start": [True],
        "shuffle": [True],
        "early_side_info_flag": [True],
        "early_side_info_dims": [[16, 8]],
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
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "hur_input_dim": list(range(7)),
            "weather_input_dim": list(range(5)),
            "hur_output_dim": list(range(2)),
            "side_info_dim": list(range(2, 7))
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



