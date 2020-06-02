import itertools
from random import shuffle


model_params_pool = {
    "lstm": {
        "batch_size": [32],
        "shuffle": [True],
        "learning_rate": [3e-3],
        "num_epochs": [1],
        "loss_type": ["l2"],
        "optimizer_type": ["adam"],
        "grad_clip": [10],
        "l2_reg": [0],
        "dropout_rate": [0],
        "early_stop_tolerance": [5],
        "final_act_type": ["tanh"],
        "relu_alpha": [1],
        "window_len_input": [10],
        "window_len_output": [10],
        "stride": [1],
        "hidden_dim_list": [[8, 8], [8, 8, 8], [32, 32]],
        "norm_method": ["standard"],
    },
    "trajgru": {
        "input_size": [(25, 25)],
        "en_dec_output_dim": [1],
        "window_in": [10],
        "window_out": [10],
        "encoder_conf": [{
            "en_num_layers": 1,
            "en_conv_dims": [16],
            "en_conv_kernel": 3,
            "en_conv_stride": 1,
            "en_pool_kernel": 3,
            "en_pool_stride": 2,
            "en_pool_padding": 0,
            "en_gru_dims": [32],
            "en_gru_kernels": [5],
            "en_connection": 5,
            "en_bias": True
        }],
        "decoder_conf": [{
            "de_input_dim": 32,
            "de_num_layers": 1,
            "de_conv_dims": [16],
            "de_conv_kernel": 3,
            "de_conv_stride": 2,
            "de_conv_padding": 0,
            "de_gru_dims": [32],
            "de_gru_kernels": [3],
            "de_connection": 5,
            "de_bias": True
        }],
        "output_conv_dims": [[16, 16]],
        "output_conv_kernels": [[5, 1]],
        "relu_alpha": [1],
        "early_side_info_flag": [True],
        "early_side_info_dims": [[16, 8]]
    }
}

trainer_params = {
    # finetune params
    "learning_rate": [1e-3],
    "num_epochs": [50],
    "loss_type": ["l2"],
    "l2_reg": [1e-3],
    "early_stop_tolerance": [5],
    "norm_method": ["standard"],
    "clip": [5],
    # batch gen params
    "batch_size": [1],
    "shuffle": [True],
    "window_len": [8],
    "return_mode": ['weather'],
    "phase_shift": [8],
    "cut_start": [False],
}


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_params = model_params_pool[self.model_name]
        self.trainer_params = trainer_params

        self.conf_list = self.create_params_list(dict(**self.trainer_params, **self.model_params))
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



