import re
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from normalizer import Normalizer
from copy import deepcopy
from static_helpers import haversine_dist


class TrajGRU(nn.Module):

    loss_dispatcher = {"l2": nn.MSELoss}

    class TrajGRUCell(nn.Module):
        def __init__(self, input_size, input_dim, hidden_dim,
                     kernel_size, bias, connection):
            """
            :param input_size: (int, int) width(M) and height(N) of input grid
            :param input_dim: int, number of channels (D) of input grid
            :param hidden_dim: int, number of channels of hidden state
            :param kernel_size: (int, int) size of the convolution kernel
            :param bias: bool weather or not to add the bias
            """
            super(TrajGRU.TrajGRUCell, self).__init__()

            self.height, self.width = input_size
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.connection = connection

            self.kernel_size = kernel_size
            self.bias = bias
            self.padding = self.kernel_size // 2

            self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                        out_channels=3 * self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

            self.projecting_channels = []
            for _ in range(self.connection):
                self.projecting_channels.append(nn.Conv2d(in_channels=self.hidden_dim,
                                                          out_channels=3 * self.hidden_dim,
                                                          kernel_size=1))

            self.projecting_channels = nn.ModuleList(self.projecting_channels)

            self.sub_net = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                     out_channels=2 * self.connection,
                                     kernel_size=5,
                                     padding=2)

        def forward(self, x, h_prev):
            """
            :param x: (b, d, m, n)
            :type x: tensor
            :param h_prev: (b, d, m, n)
            :type h_prev: tensor
            :return: (b, d, m, n)
            """
            input_conv = self.conv_input(x)

            x_z, x_r, x_h = torch.split(input_conv, self.hidden_dim, dim=1)

            traj_tensor = None
            for local_link, warped in enumerate(self.__warp(x=x, h=h_prev)):
                if local_link == 0:
                    traj_tensor = self.projecting_channels[local_link](warped)
                else:
                    traj_tensor += self.projecting_channels[local_link](warped)

            h_z, h_r, h_h = torch.split(traj_tensor, self.hidden_dim, dim=1)

            z = torch.sigmoid(x_z + h_z)
            r = torch.sigmoid(x_r + h_r)
            h = nn.functional.leaky_relu(x_h + r * h_h, negative_slope=0.2)

            h_next = (1 - z) * h + z * h_prev

            return h_next

        def __warp(self, x, h):
            """
            :param x: (b, d, m, n)
            :type x: tensor
            :param h: (b, d, m, n)
            :type h: tensor
            :return: yields warped tensor
            """
            combined = torch.cat([x, h], dim=1)
            combined_conv = self.sub_net(combined)

            # (b, 2L, m, n) --> (b, m, n, 2L)
            combined_conv = combined_conv.permute(0, 2, 3, 1)

            # scale to [0, 1]
            combined_conv = (combined_conv - combined_conv.min()) / \
                            (combined_conv.max() - combined_conv.min())
            # scale to [-1, 1]
            combined_conv = 2 * combined_conv - 1

            for l in range(0, self.connection, 2):
                # (b, m, n, 2)
                grid = combined_conv[..., l:l + 2]
                warped = nn.functional.grid_sample(h, grid, mode='bilinear')

                yield warped

        def init_hidden(self, batch_size, device):
            """
            # Create new tensor with sizes n_layers x batch_size x n_hidden,
            # initialized to zero, for hidden state of GRU
            :param batch_size: int
            :return:(b, d, m, n) tensor
            """
            hidden = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))

            return hidden

    class EncoderBlock(nn.Module):
        def __init__(self, **kwargs):
            super(TrajGRU.EncoderBlock, self).__init__()
            # encoder conf
            self.input_size = kwargs['input_size']
            self.input_dim = kwargs['input_dim']
            self.num_layers = kwargs['num_layers']
            self.window_in = kwargs['window_in']

            # down-sample conf
            self.conv_dims = kwargs['conv_dims']
            self.conv_kernel = kwargs['conv_kernel']
            self.conv_stride = kwargs['conv_stride']
            self.conv_padding = self.conv_kernel // 2

            self.pool_kernel = kwargs['pool_kernel']
            self.pool_stride = kwargs['pool_stride']
            self.pool_padding = kwargs['pool_padding']

            # traj-gru conf
            self.gru_input_sizes = self.__calc_input_size()
            self.gru_dims = kwargs['gru_dims']
            self.gru_kernels = kwargs['gru_kernels']
            self.connection = kwargs['connection']
            self.bias = kwargs['bias']

            # This attribute will be transfered to decoder
            self.list_of_sizes = [self.input_size] + self.gru_input_sizes

            self.cell_list = []
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.gru_dims[i - 1]
                self.cell_list += [
                    nn.Conv2d(in_channels=cur_input_dim,
                              out_channels=self.conv_dims[i],
                              kernel_size=self.conv_kernel,
                              stride=self.conv_stride,
                              padding=self.conv_padding),

                    nn.MaxPool2d(kernel_size=self.pool_kernel,
                                 stride=self.pool_stride,
                                 padding=self.pool_padding),

                    TrajGRU.TrajGRUCell(input_size=self.gru_input_sizes[i],
                                        input_dim=self.conv_dims[i],
                                        hidden_dim=self.gru_dims[i],
                                        kernel_size=self.gru_kernels[i],
                                        connection=self.connection,
                                        bias=self.bias)
                ]
            self.cell_list = nn.ModuleList(self.cell_list)

        def init_memory(self, batch_size, device):
            """
            Initialise every memory element hidden state
            :param batch_size: int
            :return: list of tensors (b, d, m, n)
            """
            init_states = []
            # Only iterate odd indexes
            for i in range(2, 3 * self.num_layers, 3):
                init_states.append(self.cell_list[i].init_hidden(batch_size, device))

            return init_states

        def forward(self, input_tensor, hidden_states):
            """
            :param input_tensor: (B, T, D, M, N)
            :param hidden_states: [(B, D, M, N), ..., (B, D, M, N)]
            :return:[(B, D', M', N'), ..., (B, D', M', N')] list of down-sampled tensors
            """
            layer_state_list = []
            cur_layer_input = input_tensor
            for layer_idx in range(self.num_layers):
                h = hidden_states[layer_idx]
                output_inner = []
                for t in range(self.window_in):
                    # Down-sample
                    conv_output = self.cell_list[3 * layer_idx](cur_layer_input[:, t])
                    max_output = self.cell_list[3 * layer_idx + 1](conv_output)

                    # Memory element
                    h = self.cell_list[3 * layer_idx + 2](max_output, h)
                    # Store states
                    output_inner.append(h)
                cur_layer_input = torch.stack(output_inner, dim=1)
                layer_state_list.append(h)

            return layer_state_list

        def __calc_input_size(self):
            input_sizes = []
            cur_dim = self.input_size
            for _ in range(self.num_layers):
                h, w = cur_dim
                f = self.pool_kernel
                s = self.pool_stride
                p = self.pool_padding
                # floor the kernel sizes
                cur_dim = (int((h - f + 2 * p) / s + 1),
                           int((w - f + 2 * p) / s + 1))
                input_sizes.append(cur_dim)

            return input_sizes

    class DecoderBlock(nn.Module):
        def __init__(self, **kwargs):
            super(TrajGRU.DecoderBlock, self).__init__()

            # This param should be determined by encoder
            self.encoder_layer_sizes = kwargs['encoder_layer_sizes']

            # decoder conf
            self.input_size = self.encoder_layer_sizes[-1]
            self.input_dim = kwargs['input_dim']
            self.num_layers = kwargs['num_layers']
            self.window_out = kwargs['window_out']

            # up-sample conf
            self.output_padding = self.__calc_output_pad()

            self.conv_dims = kwargs['conv_dims']
            self.conv_kernel = kwargs['conv_kernel']
            self.conv_stride = kwargs['conv_stride']
            self.conv_padding = kwargs['conv_padding']

            # traj-gru conf
            self.gru_input_sizes = self.__calc_input_size()
            self.gru_dims = kwargs['gru_dims']
            self.gru_kernels = kwargs['gru_kernels']
            self.connection = kwargs['connection']
            self.bias = kwargs['bias']

            self.cell_list = []
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.conv_dims[i - 1]
                cur_input_size = self.input_size if i == 0 else self.gru_input_sizes[i - 1]
                self.cell_list += [
                    TrajGRU.TrajGRUCell(input_size=cur_input_size,
                                        input_dim=cur_input_dim,
                                        hidden_dim=self.gru_dims[i],
                                        kernel_size=self.gru_kernels[i],
                                        connection=self.connection,
                                        bias=self.bias),

                    nn.ConvTranspose2d(in_channels=self.gru_dims[i],
                                       out_channels=self.conv_dims[i],
                                       kernel_size=self.conv_kernel,
                                       stride=self.conv_stride,
                                       padding=self.conv_padding,
                                       output_padding=self.output_padding[i]),
                ]
            self.cell_list = nn.ModuleList(self.cell_list)

        def init_memory(self, batch_size):
            """
            Initialise every memory element hidden state
            :param batch_size: int
            :return: list of tensors (b, d, m, n)
            """
            init_states = []
            # Only iterate even indexes
            for i in range(0, 2 * self.num_layers, 2):
                init_states.append(self.cell_list[i].init_hidden(batch_size=batch_size))

            return init_states

        def forward(self, input_tensor, hidden_states):
            """
            :param input_tensor: (B, T, D', M', N')
            :type input_tensor: tensor
            :param hidden_states: [(B, D', M', N'), ..., (B, D', M', N')]
            :type hidden_states: list
            :return:(B, D', M, N), [(B, D', M', N'), ..., (B, D', M', N')]
            """
            layer_state_list = []
            # Since decoder has reverse order
            len_states = len(hidden_states) - 1
            cur_layer_input = input_tensor
            for layer_idx in range(self.num_layers):
                h = hidden_states[len_states - layer_idx]
                output_inner = []
                for t in range(self.window_out):
                    # Memory element
                    h = self.cell_list[2 * layer_idx](cur_layer_input[:, t], h)

                    # Up-sample
                    conv_output = self.cell_list[2 * layer_idx + 1](h)
                    output_inner.append(conv_output)

                cur_layer_input = torch.stack(output_inner, dim=1)
                # store hidden states and decoder has reverse order
                layer_state_list.insert(0, h)

            return cur_layer_input, layer_state_list

        def __calc_output_pad(self):
            """
            calculates padding for every layer
            :return: list
            """
            output_pads = []
            for size in self.encoder_layer_sizes:
                output_pads.append((int(not size[0] % 2), int(not size[1] % 2)))

            # drop the last since we only consider the transitions
            output_pads = output_pads[:-1]

            # reverse the list since we are decoding
            output_pads = [output_pads[i - 1] for i in range(len(output_pads), 0, -1)]

            return output_pads

        def __calc_input_size(self):
            """
            calculates input sizes for every layer
            :return: list
            """
            input_sizes = []
            cur_dim = self.input_size
            for _ in range(self.num_layers):
                h, w = cur_dim
                f = self.conv_kernel
                s = self.conv_stride
                p = self.conv_padding
                cur_dim = (int((h - 1) * s - 2 * p + f), int((w - 1) * s - 2 * p + f))
                input_sizes.append(cur_dim)

            return input_sizes

    def __init__(self, input_dim, fc_output_dim, **params):
        super(TrajGRU, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim * 3
        self.fc_output_dim = fc_output_dim

        self.loss_type = params["loss_type"]
        self.learning_rate = params["learning_rate"]
        self.optimizer_type = params["optimizer_type"]
        self.l2_reg = params["l2_reg"]

        self.num_epochs = params["num_epochs"]
        self.early_stop_tolerance = params["early_stop_tolerance"]
        self.norm_method = params["norm_method"]

        self.window_in = params['window_in']
        self.window_out = params['window_out']
        self.input_size = params['input_size']
        self.en_dec_output_dim = params['en_dec_output_dim']

        self.encoder_conf = self.__remove_prefix(params['encoder_conf'])
        self.decoder_conf = self.__remove_prefix(params['decoder_conf'])

        self.regression = params.get("regression", "logistic")
        self.loss_type = params.get("loss_type", "BCE")
        self.is_stateful = params['stateful']
        self.relu_alpha = params["relu_alpha"]
        self.clip = params['clip']

        # output conv conf
        self.conv_dims = params['output_conv_dims']
        self.conv_kernels = params['output_conv_kernels']

        # define model blocks
        self.encoder = TrajGRU.EncoderBlock(input_size=self.input_size,
                                            input_dim=self.input_dim,
                                            window_in=self.window_in,
                                            **self.encoder_conf)

        encoder_layer_sizes = self.encoder.list_of_sizes

        self.decoder = TrajGRU.DecoderBlock(encoder_layer_sizes=encoder_layer_sizes,
                                            window_out=self.window_out,
                                            **self.decoder_conf)
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_dims[0],
                      out_channels=self.conv_dims[0],
                      kernel_size=self.conv_kernels[0],
                      padding=self.conv_kernels[0] // 2),
            nn.Conv2d(in_channels=self.conv_dims[1],
                      out_channels=self.en_dec_output_dim,
                      kernel_size=self.conv_kernels[1],
                      padding=self.conv_kernels[1] // 2)
        )

        self.fc = nn.Linear(in_features=self.en_dec_output_dim * self.input_size[0] * self.input_size[1],
                            out_features=self.fc_output_dim)
        self.final_act = nn.LeakyReLU(self.relu_alpha)

        self.hidden_state = None

        self.input_normalizer = Normalizer(self.norm_method)
        self.output_normalizer = Normalizer(self.norm_method)

        # set optimizer
        self._set_optimizer()

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
        for x, y in batch_generator.generate('train'):
            data_list.append(x)
            label_list.append(y)

        self.input_normalizer.fit(torch.cat(data_list))
        self.output_normalizer.fit(torch.cat(label_list))

        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()
            running_train_loss = self.step_loop(batch_generator, self.train_step, self.loss_fun, 'train', denormalize=False)
            running_val_loss = self.step_loop(batch_generator, self.eval_step, self.loss_fun, 'validation', denormalize=False)
            epoch_time = time.time() - start_time

            message_str = "\nEpoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
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
        dataset = batch_generator.dataset_dict[dataset_type]
        total_len = len(dataset)
        for count, (input_data, output_data) in enumerate(batch_generator.generate(dataset_type)):
            print("\r{:.2f}%".format(dataset.count * 100 / total_len), flush=True, end='')
            input_data = self.input_normalizer.transform(input_data).to(self.device)
            output_data = self.output_normalizer.transform(output_data).to(self.device)
            loss = step_fun(input_data, output_data, loss_fun, denormalize)  # many-to-one
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
            nn.utils.clip_grad_norm_(self.parameters(), self.clip)
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

    def forward(self, input_tensor):
        """
        :param input_tensor: (B, T, M, N, D)
        :type input_tensor: tensor
        :return: (B, T', M, N, D')
        """
        batch_size = input_tensor.shape[0]
        self.hidden_state = self.__init_hidden(batch_size=batch_size)

        # (b, t, m, n, d) -> (b, t, d, m, n)
        input_tensor = input_tensor.permute(0, 1, 4, 2, 3)

        # forward encoder block
        cur_states = self.hidden_state
        cur_states = self.encoder(input_tensor, cur_states)

        self.set_hidden_state(cur_states)

        # forward decoder block
        decoder_input = self._create_decoder_input()
        dec_output, cur_states = self.decoder(decoder_input, cur_states)

        # forward convolution
        block_output_list = []
        for t in range(self.window_out):
            conv_output = self.output_conv(dec_output[:, t])
            block_output_list.append(conv_output)

        block_output = torch.stack(block_output_list, dim=1)

        # (b, t, d, m, n) -> (b, t, m*n, d)
        block_output = block_output.permute(0, 1, 3, 4, 2)
        block_output = block_output.reshape(batch_size, self.window_out, -1)

        final_output_list = []
        for t in range(self.window_out):
            fc_out = self.final_act(self.fc(block_output[:, t]))
            final_output_list.append(fc_out)

        final_output = torch.stack(final_output_list, dim=1)

        return final_output

    def _create_decoder_input(self):
        """
        :return: zero tensor
        """
        decoder_input = torch.zeros_like(self.hidden_state[-1]).to(self.device)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, self.window_out, -1, -1, -1)
        return decoder_input

    def reset_per_epoch(self, **kwargs):
        """
        This will be called at beginning of every epoch
        :param kwargs: dict
        :return:
        """
        batch_size = kwargs['batch_size']
        self.hidden_state = self.__init_hidden(batch_size=batch_size)

    def __init_hidden(self, batch_size):
        """
        Initializes hidden states of blocks
        :param batch_size: int
        :return: list of states, e.g [state, state, ...]
        """
        # only the first block hidden is needed
        hidden_list = self.encoder.init_memory(batch_size=batch_size, device=self.device)

        return hidden_list

    def __repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        :param h: list of states, e.g [state, state, ...]
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)

    @staticmethod
    def __remove_prefix(in_dict):
        """
        removes 'de' and 'en' from input dictionary keys
        :param in_dict: input dictionary
        :type in_dict: dict
        :return: dict
        """
        new_keys = [re.sub(r'\b(en|de)_', '', old_key) for old_key in in_dict.keys()]
        return_dict = {key: value for key, value in zip(new_keys, in_dict.values())}

        return return_dict

    def _set_optimizer(self):  # TODO (fi) enable different optimizers !
        """
        Sets the optimizer
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)

    def set_hidden_state(self, state):
        """
        stores the hidden states
        :param state: list of states, e.g [state, state, ...]
        :type state: list
        """
        self.hidden_state = state
