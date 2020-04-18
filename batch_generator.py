from dataset import HurrDataset
from torch.utils.data import DataLoader


class BatchGenerator:
    def __init__(self, hurr_list, **params):
        self.hurr_list = hurr_list
        self.params = params

        self.batch_size = params['batch_size']
        self.test_ratio = params['test_ratio']
        self.val_ratio = params['val_ratio']
        self.shuffle = params['shuffle']
        self.num_works = params.get('num_works', 1)

        self.data_dict = self._split_data()
        self.dataset_dict, self.data_loader_dict = self._create_sets()

    def _split_data(self):
        data_len = len(self.hurr_list)

        test_count = int(data_len * self.test_ratio)
        val_count = int(data_len * self.val_ratio)

        data_dict = {
            'test': self.hurr_list[:test_count],
            'validation': self.hurr_list[test_count:test_count+val_count],
            'train': self.hurr_list[test_count+val_count:]
        }

        return data_dict

    def _create_sets(self):
        hurr_dataset = {}
        for i in ['test', 'validation', 'train']:
            hurr_dataset[i] = HurrDataset(hurr_list=self.data_dict[i],
                                          **self.params)

        hurr_loader = {}
        for i in ['test', 'validation', 'train']:
            hurr_loader[i] = DataLoader(hurr_dataset[i],
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle,
                                        drop_last=False)

        return hurr_dataset, hurr_loader

    def generate(self, dataset_type):
        selected_loader = self.data_loader_dict[dataset_type]
        yield from selected_loader
