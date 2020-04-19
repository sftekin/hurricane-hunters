import torch


class Normalizer:

    def __init__(self, method):

        self.method = method  # minmax, standard, none

        self.min_tensor = None
        self.max_tensor = None
        self.mean_tensor = None
        self.std_tensor = None

        self.epsilon = 1e-9

    def fit(self, input_tensor):
        """
        :param input_tensor: ....xD
        :return:
        """
        fit_dispatcher = {"minmax": self.__fit_minmax,
                          "standard": self.__fit_standard}

        if self.method == "none":
            return
        
        fit_dispatcher[self.method](input_tensor)
        
    def __fit_minmax(self, input_tensor):
        self.min_tensor = torch.min(input_tensor, dim=-1)
        self.max_tensor = torch.max(input_tensor, dim=-1)
        
    def __fit_standard(self, input_tensor):
        self.std_tensor = torch.std(input_tensor, dim=-1)
        self.mean_tensor = torch.mean(input_tensor, dim=-1)
        
    def transform(self, input_tensor):
        """
        :param input_tensor: ....xD
        :return:
        """
        transform_dispatcher = {"minmax": self.__transform_minmax,
                                "standard": self.__transform_standard}

        if self.method == "none":
            return input_tensor

        return transform_dispatcher[self.method](input_tensor)

    def __transform_minmax(self, input_tensor):
        min_tensor = self.min_tensor.repeat(*input_tensor.shape[:-1], 1)
        max_tensor = self.max_tensor.repeat(*input_tensor.shape[:-1], 1)
        return (input_tensor - min_tensor) / (max_tensor - min_tensor + self.epsilon)

    def __transform_standard(self, input_tensor):
        std_tensor = self.std_tensor.repeat(*input_tensor.shape[:-1], 1)
        mean_tensor = self.mean_tensor.repeat(*input_tensor.shape[:-1], 1)
        return (input_tensor - mean_tensor) / (std_tensor + self.epsilon)

