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
        elif isinstance(self.method, str):
            fit_dispatcher[self.method](input_tensor)
        else:
            pass
        
    def __fit_minmax(self, input_tensor):
        shape = input_tensor.shape
        input_tensor = input_tensor.reshape((int(input_tensor.numel() / shape[-1]), shape[-1]))
        self.min_tensor = torch.min(input_tensor, dim=0)[0]
        self.max_tensor = torch.max(input_tensor, dim=0)[0]
        
    def __fit_standard(self, input_tensor):
        dim = list(range(len(input_tensor.shape) - 1))
        self.std_tensor = torch.std(input_tensor, dim=dim)
        self.mean_tensor = torch.mean(input_tensor, dim=dim)
        
    def transform(self, input_tensor):
        """
        :param input_tensor: ....xD
        :return:
        """
        transform_dispatcher = {"minmax": self.__transform_minmax,
                                "standard": self.__transform_standard,
                                "scale": self.__transform_scale}

        if self.method == "none":
            return input_tensor
        elif isinstance(self.method, str):
            return transform_dispatcher[self.method](input_tensor)
        else:
            return transform_dispatcher["scale"](input_tensor, self.method)

    def __transform_minmax(self, input_tensor):
        min_tensor = self.min_tensor.repeat(*input_tensor.shape[:-1], 1)
        max_tensor = self.max_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.div(input_tensor - min_tensor, max_tensor - min_tensor + self.epsilon)

    def __transform_standard(self, input_tensor):
        std_tensor = self.std_tensor.repeat(*input_tensor.shape[:-1], 1)
        mean_tensor = self.mean_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.div(input_tensor - mean_tensor, std_tensor + self.epsilon)

    def __transform_scale(self, input_tensor, scale_term):
        return input_tensor / scale_term

    def inverse_transform(self, input_tensor):
        """
        :param input_tensor: ....xD
        :return:
        """
        inverse_transform_dispatcher = {"minmax": self.__inverse_transform_minmax,
                                        "standard": self.__inverse_transform_standard,
                                        "scale": self.__inverse_transform_scale}

        if self.method == "none":
            return input_tensor
        elif isinstance(self.method, str):
            return inverse_transform_dispatcher[self.method](input_tensor)
        else:
            return inverse_transform_dispatcher["scale"](input_tensor, self.method)

    def __inverse_transform_minmax(self, input_tensor):
        min_tensor = self.min_tensor.repeat(*input_tensor.shape[:-1], 1)
        max_tensor = self.max_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.mul(input_tensor, max_tensor - min_tensor) + min_tensor

    def __inverse_transform_standard(self, input_tensor):
        std_tensor = self.std_tensor.repeat(*input_tensor.shape[:-1], 1)
        mean_tensor = self.mean_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.mul(input_tensor, std_tensor) + mean_tensor

    def __inverse_transform_scale(self, input_tensor, scale_term):
        return input_tensor * scale_term
