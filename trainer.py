import time
import torch
import torch.nn as nn
import torch.optim as optim


from copy import deepcopy
from normalizer import Normalizer
from static_helpers import haversine_dist


class Trainer:

    loss_dispatcher = {"l2": nn.MSELoss}

    def __init__(self, **params):
        self.num_epochs = params["num_epochs"]
        self.early_stop_tolerance = params["early_stop_tolerance"]
        self.norm_method = params["norm_method"]
        self.loss_type = params["loss_type"]
        self.learning_rate = params["learning_rate"]
        self.l2_reg = params["l2_reg"]
        self.clip = params['clip']
        self.device = params['device']

        self.input_normalizer = Normalizer(self.norm_method)
        self.output_normalizer = Normalizer(self.norm_method)

    def fit(self, model, batch_generator):
        """
        :param TrajGRU model:
        :param BatchGenerator batch_generator:
        :return:
        """
        print('Training starts...')
        model.to(self.device)

        train_loss = []
        val_loss = []

        tolerance = 0
        best_epoch = 0
        best_val_loss = 1e6
        evaluation_val_loss = best_val_loss
        best_dict = model.state_dict()

        data_list = []
        label_list = []
        for x, y in batch_generator.generate('train'):
            data_list.append(x)
            label_list.append(y)

        self.input_normalizer.fit(torch.cat(data_list))
        self.output_normalizer.fit(torch.cat(label_list))

        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.l2_reg)

        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()

            # train
            running_train_loss = self.step_loop(model=model,
                                                batch_generator=batch_generator,
                                                step_fun=self.train_step,
                                                loss_fun=self.loss_fun,
                                                dataset_type='train',
                                                optimizer=optimizer,
                                                denormalize=False)

            # validate
            running_val_loss = self.step_loop(model=model,
                                              batch_generator=batch_generator,
                                              step_fun=self.eval_step,
                                              loss_fun=self.loss_fun,
                                              dataset_type='validation',
                                              optimizer=None,
                                              denormalize=False)

            epoch_time = time.time() - start_time

            message_str = "\nEpoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))
            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(model.state_dict())  # brutal
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.early_stop_tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)

                evaluation_val_loss = self.step_loop(model=model,
                                                     batch_generator=batch_generator,
                                                     step_fun=self.eval_step,
                                                     loss_fun=self.loss_fun_evaluation,
                                                     dataset_type='validation',
                                                     optimizer=None,
                                                     denormalize=True)

                message_str = "Early exiting from epoch: {}, Validation error: {:.5f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break
        print('Training finished')

        return train_loss, val_loss, evaluation_val_loss

    def step_loop(self, model, batch_generator, step_fun, loss_fun, dataset_type, optimizer, denormalize):
        count = 0
        running_loss = 0.0
        dataset = batch_generator.dataset_dict[dataset_type]
        hidden = self.reset_per_epoch(model=model, batch_size=batch_generator.batch_size)
        for count, (input_data, output_data) in enumerate(batch_generator.generate(dataset_type)):
            print("\r{:.2f}%".format(dataset.count * 100 / len(dataset)), flush=True, end='')

            input_data = self.input_normalizer.transform(input_data).to(self.device)
            output_data = self.output_normalizer.transform(output_data).to(self.device)

            loss = step_fun(model=model,
                            input_tensor=input_data,
                            output_tensor=output_data,
                            hidden=hidden,
                            loss_fun=loss_fun,
                            optimizer=optimizer,
                            denormalize=denormalize)  # many-to-one

            hidden = self.repackage_hidden(hidden)

            running_loss += loss.item()

        running_loss /= (count + 1)

        return running_loss

    def train_step(self, model, input_tensor, output_tensor, hidden, loss_fun, optimizer, denormalize):
        def closure():
            optimizer.zero_grad()
            predictions = model.forward(input_tensor, hidden)
            loss = loss_fun(predictions, output_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            return loss

        loss = optimizer.step(closure)

        return loss

    def eval_step(self, model, input_tensor, output_tensor, hidden, loss_fun, optimizer, denormalize):
        predictions = model.forward(input_tensor, hidden)
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

    def reset_per_epoch(self, model, batch_size):
        """
        This will be called at beginning of every epoch
        :param model:
        :param batch_size:
        :return:
        """
        hidden_list = model.init_hidden(batch_size=batch_size, device=self.device)
        return hidden_list

    def repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        :param h: list of states, e.g [state, state, ...]
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
