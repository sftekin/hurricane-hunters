import torch
import numpy as np

from torch.utils.data import Dataset


class HurrDataset(Dataset):
    def __init__(self, hurricane_list, weather_list=None, **params):
        self.hurricane_list = hurricane_list
        self.weather_list = weather_list
        # I've assumed input and output window_len are equal
        self.window_len_input = params['window_len_input']
        self.window_len_output = params['window_len_output']
        self.stride = params['stride']
        self.hur_input_dim = params['hur_input_dim']
        self.hur_output_dim = params['hur_output_dim']
        self.hur_data, self.hur_label = self._create_buffer(self.hurricane_list)
        if weather_list:
            self.weather_data, self.weather_label = self._create_buffer(self.weather_list)
            # reshape (b, t, l, m, n, d) --> (b, t, m, n, d*l)
            b, t, l, m, n, d = self.weather_data.shape
            self.weather_data = np.transpose(self.weather_data, (0, 1, 3, 4, 2, 5))
            self.weather_data = self.weather_data.reshape((b, t, m, n, d*l))

    def _create_buffer(self, in_datalist):
        x_buffer = []
        y_buffer = []
        for data in in_datalist:
            for n in range(0, data.shape[0] - (self.window_len_input + self.window_len_output), self.stride):
                x = data[n:n+self.window_len_input, :]
                y = np.zeros_like(x)
                try:
                    # targets shifted by one
                    y[:-1], y[-1] = x[1:], data[n+self.window_len_output]
                except IndexError:
                    continue

                x_buffer.append(x)
                y_buffer.append(y)

        # target and data are in shape of (N, window_len, D)
        x_buffer = np.stack(x_buffer, axis=0)
        y_buffer = np.stack(y_buffer, axis=0)

        return x_buffer, y_buffer

    def __len__(self):
        return self.hur_data.shape[0]

    def __getitem__(self, idx):

        if self.weather_data is None:
            x = self.hur_data[:, :, self.hur_input_dim]
            y = self.hur_label[:, :, self.hur_output_dim]
        else:
            x = self.weather_data
            y = self.hur_label[:, :, self.hur_output_dim]
            y = y[1:] - y[:-1]

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        return x[idx], y[idx]
