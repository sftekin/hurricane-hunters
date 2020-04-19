import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):

    optimizer_dispatcher = {"adam": torch.optim.Adam,
                            "sgd": torch.optim.SGD}

    loss_dispatcher = {"l2": nn.MSELoss}

    activation_dispatcher = {"relu": nn.ReLU,
                             "tanh": nn.Tanh,
                             "sigmoid": nn.Sigmoid}

    def __init__(self, input_dim, output_dim, **params):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_type = params["loss_type"]
        self.learning_rate = params["learning_rate"]
        self.optimizer_type = params["optimizer_type"]
        self.l2_reg = params["l2_reg"]
        self.optimizer = self.optimizer_dispatcher[self.optimizer_type](lr=self.learning_rate, weight_decay=self.l2_reg)
        self.dropout_rate = params["dropout_rate"]

        self.num_epochs = params["num_epochs"]
        self.early_stop_tolerance = params["early_stop_tolerance"]

        self.num_layers = params["num_layers"]
        self.hidden_dim_list = params["hidden_dim_list"]
        self.final_act_type = params["final_act_type"]

        self.__create_cells()
        self.__create_dropout_layer()
        self.__create_dense()

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

    def __create_final_act_layer(self):
        self.final_act_layer = self.activation_dispatcher[self.final_act_type]()

    def __init_hidden_states(self, batch_size):
        h_list = []
        for cell in self.cell_list:
            h_list.append(Variable(torch.zeros(batch_size, cell.hidden_size)))
        return h_list

    def forward(self, input_tensor):

        batch_size = input_tensor.shape[0]
        num_steps = input_tensor.shape[1]
        h_list = self.__init_hidden_states(batch_size)

        for step in num_steps:
            x = input_tensor[:, step]
            for layer_idx, cell in enumerate(self.cell_list):
                h = cell(x, h_list[layer_idx])
                h_list[layer_idx] = h
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

        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()
            running_train_loss = self.step_loop(batch_generator, self.train_step, 'train')
            running_val_loss = self.step_loop(batch_generator, self.eval_step, 'validation')
            epoch_time = time.time() - start_time

            message_str = "Epoch: {}, Train_loss: {:.3f}, Validation_loss: {:.3f}, Took {:.3f} seconds."
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
                evaluation_val_loss = self.step_loop(batch_generator, self.eval_step, 'validation')
                message_str = "Early exiting from epoch: {}, Rounded MAE for validation set: {:.3f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

        print('Training finished')
        return train_loss, val_loss, evaluation_val_loss

    def step_loop(self, batch_generator, step_fun, dataset_type):
        count = 0
        running_loss = 0.0

        for count, (input_data, output_data) in batch_generator.generate(dataset_type):
            input_data = torch.Tensor(input_data)
            output_data = torch.Tensor(output_data)
            loss = step_fun(input_data, output_data, self.loss_fun)
            running_loss += loss.numpy()

        running_loss /= (count + 1)

        return running_loss

    def train_step(self, input_tensor, output_tensor):

        def closure():
            self.optimizer.zero_grad()
            predictions = self.forward(input_tensor)
            loss = self.loss_fun(output_tensor, predictions)
            loss.backward()
            return loss

        loss = self.optimizer.step(closure)

        return loss

    def eval_step(self, input_tensor, output_tensor):

        predictions = self.forward(input_tensor)
        loss = self.loss_fun(predictions, output_tensor)

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
