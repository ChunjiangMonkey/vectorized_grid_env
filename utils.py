import torch


def intersection_1d(tensor_a, tensor_b):
    intersect_batch, index = torch.unique(torch.cat((tensor_a, tensor_b)), return_counts=True)
    return intersect_batch[index > 1]
