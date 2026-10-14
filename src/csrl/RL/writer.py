import torch
from robotic.tools import yaml_helper
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, tag):
        self.data = {}
        self.tag = tag
        self.writer = SummaryWriter(log_dir=f"tensorboard/{self.tag}")

    # tim: add() should .detach() tensor values to avoid that the accumulator retains the full autograd graph for every training step between flushes
    def add(self, key, value, count):
        if isinstance(value, torch.Tensor):
            value = value.detach()  # mt-tim: this ok?
        if key not in self.data:
            self.data[key] = [value, count]
        else:
            self.data[key][0] += value
            self.data[key][1] += count

    def get_value(self, key):  # without clearing
        if key in self.data:
            value = self.data[key]
            if value[1] > 0.0:
                val = value[0]
                if torch.is_tensor(val):
                    val = val.item()
                return val / value[1]
            else:
                return None
        else:
            return None

    def get_dict(self):  # also clears the data
        dat = {}
        for key, value in self.data.items():
            if value[1] > 0.0:
                val = value[0]
                if torch.is_tensor(val):
                    val = val.item()
                dat[key] = val / value[1]
            self.data[key][0] = 0.0
            self.data[key][1] = 0.0
        return dat

    def write(self, t, verbose=0):
        D = self.get_dict()
        for key, value in D.items():
            self.writer.add_scalar(f"{key}", value, t)
        self.writer.flush()

        if verbose > 0:
            print(f"-- Writer data at t: {t}, tag: {self.tag}\n{yaml_helper.dict_as_yaml(D)}", end="")
