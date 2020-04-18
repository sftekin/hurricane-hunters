import numpy as np

from torch.utils.data import Dataset


class HurrDataset(Dataset):
    def __init__(self, hurr_list, **params):
        self.hurr_list = hurr_list
        # TODO: I've assumed input and output window_len are equal
        self.window_len = params['window_len']
        self.output_dim = params['output_dim']
        self.data, self.label = self._create_buffer()

    def _create_buffer(self):
        x_buffer = []
        y_buffer = []
        for hurr in self.hurr_list:
            for n in range(0, hurr.shape[0], self.window_len):
                x = hurr[n:n+self.window_len, :]
                y = np.zeros_like(x)
                try:
                    # targets shifted by one
                    y[:-1], y[-1] = x[1:], hurr[n+self.window_len]
                except IndexError:
                    pass

                x_buffer.append(x)
                y_buffer.append(y)

        # target and data are in shape of (N, self.window_len, D)
        x_buffer = np.stack(x_buffer, axis=0)
        y_buffer = np.stack(y_buffer, axis=0)

        return x_buffer, y_buffer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx, :, [self.output_dim]]
