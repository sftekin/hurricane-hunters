import torch
import torch.nn as nn
from torch.autograd import Variable


class TrajGRU(nn.Module):
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

    def __init__(self, params):

        self.input_size = params['input_size']
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']

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
                                            output_dim=self.output_dim,
                                            window_out=self.window_out,
                                            **self.decoder_conf)
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_dims[0],
                      out_channels=self.conv_dims[0],
                      kernel_size=self.conv_kernels[0],
                      padding=self.conv_kernels[0] // 2),
            nn.Conv2d(in_channels=self.conv_dims[1],
                      out_channels=self.output_dim,
                      kernel_size=self.conv_kernels[1],
                      padding=self.conv_kernels[1] // 2)
        )

        # set optimizer
        self._set_optimizer()

    def forward(self, input_tensor, early_side_info_dict, late_side_info_dict):
        """
        :param input_tensor: (B, T, M, N, D)
        :type input_tensor: tensor
        :param early_side_info_dict: first output of self.organize_side_info_dict
        :type early_side_info_dict: dict
        :param late_side_info_dict: second output of self.organize_side_info_dict
        :type late_side_info_dict: dict
        :return: (B, T', M, N, D')
        """
        # (b, t, m, n, d) -> (b, t, d, m, n)
        input_tensor = input_tensor.permute(0, 1, 4, 2, 3)

        if self.early_side_info_block:
            input_tensor = self.forward_early_late_block(input_tensor=input_tensor,
                                                         side_info_dict=early_side_info_dict,
                                                         block_type='early')

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

        if self.late_side_info_block:
            block_output = self.forward_early_late_block(input_tensor=block_output,
                                                         side_info_dict=late_side_info_dict,
                                                         block_type='late')
        # (b, t, d, m, n) -> (b, t, m, n, d)
        final_output = block_output.permute(0, 1, 3, 4, 2)

        if self.regression == 'logistic':
            final_output = torch.sigmoid(final_output)

        return final_output

    def _create_decoder_input(self):
        """
        :return: zero tensor
        """
        decoder_input = torch.zeros_like(self.hidden_state[-1]).to(self.device)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, self.window_out, -1, -1, -1)
        return decoder_input
