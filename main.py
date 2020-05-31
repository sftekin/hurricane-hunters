import argparse
import os
import pickle as pkl

from config import Config
from data_creator import DataCreator
from batch_generator import BatchGenerator
from train import train, predict


def select_best_model(results_dir):
    print("Selecting best model...")
    experiments = os.listdir(results_dir)

    best_model = None
    best_conf = {}
    best_loss = 1e6
    for exp in experiments:
        if "experiment" not in exp:
            continue
        exp_path = os.path.join(results_dir, exp)
        conf_path = os.path.join(exp_path, 'config.pkl')
        model_path = os.path.join(exp_path, 'model.pkl')

        with open(conf_path, 'rb') as f:
            config = pkl.load(f)
        eval_loss = config['evaluation_val_loss']
        if eval_loss < best_loss:
            best_loss = eval_loss
            with open(model_path, 'rb') as f:
                best_model = pkl.load(f)
            best_conf = config

    return best_model, best_conf


def main(overwrite_flag):
    model_name = 'trajgru'
    data_folder = 'data'
    hurricane_path = os.path.join(data_folder, 'ibtracs.NA.list.v04r00.csv')
    results_folder = 'results'

    config_obj = Config(model_name)
    data = DataCreator(hurricane_path, **config_obj.data_params)
    hurricane_list, weather_list = data.hurricane_list, data.weather_list

    print("Starting experiments")
    for exp_count, conf in enumerate(config_obj.conf_list):
        print('\nExperiment {}'.format(exp_count))
        print('-*-' * 10)
        # print(conf)

        batch_generator = BatchGenerator(hurricane_list=hurricane_list,
                                         weather_list=weather_list,
                                         batch_size=conf["batch_size"],
                                         shuffle=conf['shuffle'],
                                         window_len_input=conf["window_len_input"],
                                         window_len_output=conf["window_len_output"],
                                         stride=conf["stride"],

                                         **config_obj.experiment_params)

        train(model_name, batch_generator, exp_count, overwrite_flag, **conf)

    best_model, best_conf = select_best_model(results_folder)

    batch_generator = BatchGenerator(hurricane_list=hurricane_list,
                                     batch_size=best_conf["batch_size"],
                                     shuffle=best_conf['shuffle'],
                                     window_len_input=best_conf["window_len_input"],
                                     window_len_output=best_conf["window_len_output"],
                                     stride=best_conf["stride"],
                                     **config_obj.experiment_params)

    print("Testing with best model...")
    predict(best_model, batch_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', type=int, default=0)  # overwrite previous results

    args = parser.parse_args()

    main(args.overwrite)
