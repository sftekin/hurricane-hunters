import argparse
import os
import pickle as pkl

from config import Config
from dataset import Dataset
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
    model_name = 'rnn'
    data_folder = 'data'
    results_folder = 'results'

    config_obj = Config(model_name)

    print("Starting experiments")
    for exp_count, conf in enumerate(config_obj.conf_list):
        print('\nExperiment {}'.format(exp_count))
        print('-*-' * 10)

        # TODO (sft) Kanka bu argümanları abstract yazdım buraya lazım olcak muhtemelen diye
        dataset = Dataset(data_dir=data_folder,
                          batch_size=conf["batch_size"],
                          shuffle=conf['shuffle'],
                          window_len_input=conf["window_len_input"],
                          window_len_output=conf["window_len_output"],
                          **config_obj.data_params)

        train(dataset.train_ds, dataset.val_ds, exp_count, overwrite_flag, **conf)

    best_model, best_conf = select_best_model(results_folder)

    dataset = Dataset(data_dir=data_folder,
                      batch_size=best_conf["batch_size"],
                      shuffle=False,
                      window_len_input=best_conf["window_len_input"],
                      window_len_output=best_conf["window_len_output"],
                      **config_obj.data_params)

    print("Testing with best model...")
    predict(best_model, dataset.test_ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', type=int, default=0)  # overwrite previous results

    args = parser.parse_args()

    main(args.overwrite)
