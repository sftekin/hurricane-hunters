import torch
import numpy as np


class HurrDataset:
    def __init__(self, hurricane_list, weather_list, **params):
        self.hurricane_list = hurricane_list
        self.weather_list = weather_list

        self.data_len = len(self.hurricane_list)
        # I've assumed input and output window_len are equal
        self.batch_size = params['batch_size']
        self.window_len_input = params['window_len_input']
        self.window_len_output = params['window_len_output']
        self.stride = params['stride']
        self.hur_input_dim = params['hur_input_dim']
        self.weather_input_dim = params['weather_input_dim']
        self.hur_output_dim = params['hur_output_dim']
        self.return_mode = params['return_mode']

    def _create_buffer(self, data, label):
        x_buffer = []
        y_buffer = []
        for n in range(0, data.shape[0] - (self.window_len_input + self.window_len_output), self.stride):
            x = data[n:n+self.window_len_input, :]
            y = label[n+self.window_len_input:n+self.window_len_output+self.window_len_output, :]

            x_buffer.append(x)
            y_buffer.append(y)

        # target and data are in shape of (N, window_len, D)
        x_buffer = np.stack(x_buffer, axis=0)
        y_buffer = np.stack(y_buffer, axis=0)

        return x_buffer, y_buffer

    def __len__(self):
        return self.data_len

    def next(self):
        for idx in range(self.data_len):
            hur_path = self.hurricane_list[idx]
            weather_path = self.weather_list[idx]

            # load hurricane sample
            hur_data = np.load(hur_path, allow_pickle=True)

            if len(hur_data) < self.window_len_input + self.window_len_output:
                continue

            if self.return_mode == 'weather':
                # load weather sample and format
                weather_data = np.load(weather_path, allow_pickle=True)
                weather_data = weather_data[..., self.weather_input_dim]

                # reshape weather sample
                t, l, m, n, d = weather_data.shape
                weather_data = np.transpose(weather_data, (0, 2, 3, 1, 4))
                weather_data = weather_data.reshape((t, m, n, d*l))

                # generate batch
                x, y = self._create_buffer(data=weather_data, label=hur_data)

                # format label
                y = y[:, :, self.hur_output_dim]
                y = y[1:] - y[:-1]

            elif self.return_mode == 'hurricane':
                # generate batch
                x, y = self._create_buffer(data=hur_data, label=hur_data)

                # format data and label
                x = x[:, :, self.hur_input_dim]
                y = y[:, :, self.hur_output_dim]
            else:
                raise KeyError("return mode: {}".format(self.return_mode))

            # convert to tensor
            x = torch.Tensor(x)
            y = torch.Tensor(y)

            # return batches
            for i in range(0, len(x), self.batch_size):
                if i+self.batch_size <= len(x):
                    yield x[i:i+self.batch_size], y[i:i+self.batch_size]
