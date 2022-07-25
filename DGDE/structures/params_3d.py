import torch

from data.datasets.kitti_utils import Calibration as C1


class ParamsList():
    """
    This class represents labels of specific object.
    """

    def __init__(self, image_size, is_train=True):
        self.size = image_size
        self.is_train = is_train
        self.extra_fields = {}

    def add_field(self, field, field_data):
        if isinstance(field_data, (C1,torch.Tensor,str)):
            field_data = field_data
        else:
            field_data = torch.as_tensor(field_data)

        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, target):
        for k, v in target.extra_fields.items():
            self.extra_fields[k] = v

    def to(self, device):
        target = ParamsList(self.size, self.is_train)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            target.add_field(k, v)
        return target

    def __len__(self):
        if self.is_train:
            reg_num = len(torch.nonzero(self.extra_fields["reg_mask"]))
        else:
            reg_num = 0
        return reg_num

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "regress_number={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
