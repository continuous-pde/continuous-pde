import torch
import numpy as np


class Accumulator:
    def __init__(self):
        self.history = {}
        self.size = 0

    def add(self, values, batch_size=None):
        for key in values.keys():
            value = values[key]
            try:
                value = value.detach().cpu().numpy()
            except:
                pass

            if batch_size is not None:
                delta = batch_size
            else:
                delta = int(value.shape[0])
            self._add_to_history(key, value, delta)

        self.size += delta

    def _add_to_history(self, key, value, delta):
        if key not in self.history.keys():
            self.history[key] = [value] * delta
        else:
            self.history[key] += [value] * delta

    def mean(self, prefix="", with_stats=False):
        assert self.size > 0
        output = {prefix + key: np.sum(self.history[key]) / self.size for key in self.history.keys()}
        if with_stats:
            for key in self.history.keys():
                d = np.array(self.history[key])
                output[prefix + key + "_std"] = np.sqrt(np.sum(d ** 2) / self.size - output[prefix + key] ** 2)
                output[prefix + key + "_min"] = np.min(d)
                output[prefix + key + "_max"] = np.max(d)
        return output
