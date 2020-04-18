import os
import pickle as pkl
import matplotlib.pyplot as plt
from models.lstm import LSTM


def train(train_ds, val_ds, exp_count, overwrite_flag, **params):

    if overwrite_flag:
        tag = exp_count
    else:
        tag_list = [int(path.split("_")[-1]) for path in os.listdir("results") if "exp" in path]
        if tag_list:
            tag = max(tag_list) + 1
        else:
            tag = 0

    save_dir = 'experiment_' + str(tag)
    save_dir = os.path.join('results', save_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    input_dim = train_ds.input_dim
    output_dim = train_ds.output_dim

    model = LSTM(input_dim, output_dim, **params)

    train_loss, val_loss, evaluation_val_loss = model.fit(train_ds, val_ds)

    # plot and save the loss curve
    plot_loss_curve(train_loss, val_loss, save_dir)

    save_dict = params
    save_dict['train_loss'] = train_loss
    save_dict['val_loss'] = val_loss
    save_dict['evaluation_val_loss'] = evaluation_val_loss
    print('Saving...')
    conf_save_path = os.path.join(save_dir, 'config.pkl')
    model_save_path = os.path.join(save_dir, 'model.pkl')
    for path, obj in zip([conf_save_path, model_save_path], [save_dict, model]):
        with open(path, 'wb') as file:
            pkl.dump(obj, file)


def predict(model, test_ds):
    test_loss = model.step_loop(test_ds, model.eval_step)
    print("Test Rounded MAE loss: {:.3f}".format(test_loss))


def plot_loss_curve(train_loss, eval_loss, save_dir):
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, '-o')
    plt.plot(range(len(eval_loss)), eval_loss, '-o')
    plt.title('Learning Curve')
    plt.xlabel('num_epoch')
    plt.ylabel('MSE loss')
    plt.legend(['train', 'validation'])
    plt.grid(True)
    plt.savefig(save_path, dpi=400)
    plt.close()
