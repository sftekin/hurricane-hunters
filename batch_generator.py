from dataset import HurrDataset
from torch.utils.data import DataLoader


class BatchGenerator:
    def __init__(self, hurricane_list, weather_list, **params):
        self.hurricane_list = hurricane_list
        self.weather_list = weather_list
        self.params = params

        self.batch_size = params['batch_size']
        self.test_ratio = params['test_ratio']
        self.val_ratio = params['val_ratio']
        self.num_works = params.get('num_works', 1)

        self.hurricane_dict = self._split_data(self.hurricane_list)
        self.weather_dict = self._split_data(self.weather_list)

        self.dataset_dict = self._create_sets()

    def _split_data(self, in_data):
        data_len = len(in_data)

        test_count = int(data_len * self.test_ratio)
        val_count = int(data_len * self.val_ratio)

        data_dict = {
            'test': in_data[:test_count],
            'validation': in_data[test_count:test_count+val_count],
            'train': in_data[test_count+val_count:]
        }
        return data_dict

    def _create_sets(self):
        hurricane_dataset = {}
        for i in ['train', 'validation', 'test']:
            dataset = HurrDataset(hurricane_list=self.hurricane_dict[i],
                                  weather_list=self.weather_dict[i],
                                  **self.params)
            hurricane_dataset[i] = dataset

        return hurricane_dataset

    def generate(self, dataset_type):
        selected_loader = self.dataset_dict[dataset_type]
        yield from selected_loader.next()


if __name__ == '__main__':
    from data_creator import DataCreator

    params = {
        'batch_size': 4,
        'test_ratio': 0.1,
        'val_ratio': 0.1,
        'shuffle': True,
        'num_works': 0,
        'window_len_input': 10,
        'window_len_output': 10,
        'hur_input_dim': list(range(7)),
        'hur_output_dim': [0, 1],
        'weather_input_dim': list(range(5)),
        'return_mode': 'weather',
        'stride': 1
    }

    data_params = {
        "season_range": (2015, 2020),
        "weather_im_size": (25, 25),
        "weather_freq": 3,
        "weather_spatial_range": [[0, 65], [-110, 10]],
        "weather_raw_dir": 'data/weather_raw',
        "rebuild": False
    }
    data_creator = DataCreator(hurricane_path='data/ibtracs.NA.list.v04r00.csv', **data_params)
    batch_generator = BatchGenerator(hurricane_list=data_creator.hurricane_list,
                                     weather_list=data_creator.weather_list,
                                     **params)

    for x, y in batch_generator.generate('train'):
        print(x.shape, y.shape)

