import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from normalizer import Normalizer
from static_helpers import haversine_dist


class LSTM(nn.Module):

    optimizer_dispatcher = {"adam": torch.optim.Adam,
                            "sgd": torch.optim.SGD}

    loss_dispatcher = {"l2": nn.MSELoss}

    activation_dispatcher = {"leaky_relu": nn.LeakyReLU,
                             "tanh": nn.Tanh,
                             "sigmoid": nn.Sigmoid}

    def __init__(self, input_dim, output_dim, **params):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_type = params["loss_type"]
        self.learning_rate = params["learning_rate"]
        self.optimizer_type = params["optimizer_type"]
        self.grad_clip = params["grad_clip"]
        self.l2_reg = params["l2_reg"]
        self.dropout_rate = params["dropout_rate"]

        self.num_epochs = params["num_epochs"]
        self.early_stop_tolerance = params["early_stop_tolerance"]

        self.hidden_dim_list = params["hidden_dim_list"]
        self.num_layers = len(self.hidden_dim_list)
        self.final_act_type = params["final_act_type"]
        self.relu_alpha = params["relu_alpha"]

        self.input_norm_method = params["input_norm_method"]
        self.output_norm_method = params["output_norm_method"]

        self.__create_rnn_cell_list()
        self.__create_dropout_layer()
        self.__create_dense_layer()
        self.__create_final_act_layer()

        self.optimizer = self.optimizer_dispatcher[self.optimizer_type](self.parameters(), lr=self.learning_rate,
                                                                        weight_decay=self.l2_reg)

        self.input_normalizer = Normalizer(self.input_norm_method)
        self.output_normalizer = Normalizer(self.output_norm_method)

    def __create_rnn_cell_list(self):
        cell_list = []
        input_dim = self.input_dim

        for i in range(self.num_layers):
            hidden_dim = self.hidden_dim_list[i]
            cell_list.append(nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim))
            input_dim = self.hidden_dim_list[i]

        self.cell_list = nn.ModuleList(cell_list)

    def __create_dropout_layer(self):
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def __create_dense_layer(self):
        self.dense_layer = nn.Linear(in_features=self.hidden_dim_list[-1], out_features=self.output_dim)

    def __pass(self, x):
        return x

    def __create_final_act_layer(self):
        if self.final_act_type == "none":
            self.final_act_layer = self.__pass
        elif self.final_act_type == "leaky_relu":
            self.final_act_layer = self.activation_dispatcher[self.final_act_type](self.relu_alpha)
        else:
            self.final_act_layer = self.activation_dispatcher[self.final_act_type]()

    def __init_hidden_states(self, batch_size):
        self.h_list = []
        self.c_list = []

        for cell in self.cell_list:
            self.h_list.append(Variable(torch.zeros(batch_size, cell.hidden_size)))
            self.c_list.append(Variable(torch.zeros(batch_size, cell.hidden_size)))

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(LSTM.repackage_hidden(v) for v in h)

    def forward(self, input_tensor):

        batch_size = input_tensor.shape[0]
        num_steps = input_tensor.shape[1]
        self.__init_hidden_states(batch_size)

        for step in range(num_steps):
            x = input_tensor[:, step]
            for layer_idx, cell in enumerate(self.cell_list):
                h, c = cell(x, (self.h_list[layer_idx], self.c_list[layer_idx]))
                self.h_list[layer_idx] = h
                self.c_list[layer_idx] = c
                x = h

        x = self.dropout_layer(x)
        x = self.dense_layer(x)
        x = self.final_act_layer(x)

        return x

    def fit(self, batch_generator):
        print('Training starts...')
        train_loss = []
        val_loss = []

        tolerance = 0
        best_epoch = 0
        best_val_loss = 1e6
        evaluation_val_loss = best_val_loss
        best_dict = self.state_dict()

        data_list = []
        label_list = []
        for x, y, _ in batch_generator.generate('train'):
            data_list.append(x.reshape(-1, *x.shape[2:]))
            label_list.append(y.reshape(-1, *y.shape[2:]))

        self.input_normalizer.fit(torch.cat(data_list))
        self.output_normalizer.fit(torch.cat(label_list))

        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()
            running_train_loss = self.step_loop(batch_generator, self.train_step, self.loss_fun, 'train', denormalize=False)
            running_val_loss = self.step_loop(batch_generator, self.eval_step, self.loss_fun, 'validation', denormalize=False)
            epoch_time = time.time() - start_time

            message_str = "Epoch: {}, Train_loss: {:.8f}, Validation_loss: {:.8f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))
            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(self.state_dict())  # brutal
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.early_stop_tolerance or epoch == self.num_epochs - 1:
                self.load_state_dict(best_dict)
                evaluation_val_loss = self.step_loop(batch_generator, self.eval_step, self.loss_fun_evaluation,
                                                     'validation', denormalize=True)
                message_str = "Early exiting from epoch: {}, Validation error: {:.5f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

        print('Training finished')
        return train_loss, val_loss, evaluation_val_loss

    def step_loop(self, batch_generator, step_fun, loss_fun, dataset_type, denormalize):
        count = 0
        running_loss = 0.0

        for count, (input_data, output_data, _) in enumerate(batch_generator.generate(dataset_type)):
            b, t, d = input_data.shape
            input_data = self.input_normalizer.transform(input_data.reshape(-1, d)).reshape(b, t, d)
            b, t, d = output_data.shape
            output_data = self.output_normalizer.transform(output_data.reshape(-1, d)).reshape(b, t, d)
            loss = step_fun(input_data, output_data[:, -1], loss_fun, denormalize)  # many-to-one
            try:
                running_loss += loss.detach().numpy()
            except:
                running_loss += loss

        running_loss /= (count + 1)

        return running_loss

    def train_step(self, input_tensor, output_tensor, loss_fun, denormalize):

        def closure():
            self.optimizer.zero_grad()
            predictions = self.forward(input_tensor)
            loss = loss_fun(predictions, output_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            return loss

        loss = self.optimizer.step(closure)

        return loss

    def eval_step(self, input_tensor, output_tensor, loss_fun, denormalize):

        predictions = self.forward(input_tensor)
        if denormalize:
            predictions = self.output_normalizer.inverse_transform(predictions)
            output_tensor = self.output_normalizer.inverse_transform(output_tensor)
        loss = loss_fun(predictions, output_tensor)

        return loss

    def loss_fun(self, predictions, labels):
        """
        :param predictions: BxD_out
        :param labels: BxD_out
        :return:
        """
        loss_obj = self.loss_dispatcher[self.loss_type]()
        loss = loss_obj(predictions, labels)

        return loss

    def loss_fun_evaluation(self, predictions, labels):
        """
        :param labels:
        :param preds:
        :return:
        """
        predictions = predictions.detach().numpy()
        labels = labels.detach().numpy()

        loss = haversine_dist(predictions, labels).mean()

        return loss
