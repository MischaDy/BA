from math import pi, ceil, floor
import psutil
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import logging
from timeit import default_timer

from multiprocess_chunks import map_list_as_chunks

logging.basicConfig(level=logging.INFO)

proc = psutil.Process(os.getpid())
proc.nice(psutil.NORMAL_PRIORITY_CLASS)

# meaning??
LIM = int(1e2)


def main(data, mat_size):
    return compute_dist_matrix(data, compute_dist)



def process_data(data_point, extra_data=None):
    # dummy function
    return data_point


def compute_dist_matrix(data, compute_dist):
    dist_matrix = np.array(
        [[compute_dist(val_inner, val_outer) if ind_outer <= ind_inner else 0
          for ind_inner, val_inner in enumerate(data)]
         for ind_outer, val_outer in enumerate(data)
         ]
    )
    dist_matrix = dist_matrix + dist_matrix.T
    return dist_matrix


def join_matrix_parts(matrix_parts, dim=0):
    return torch.cat(matrix_parts, dim)


def show_dist_matrix(dist_matrix):
    plt.imshow(dist_matrix, cmap=cm.YlOrRd)
    plt.colorbar()
    plt.show()


def compute_dist(val1, val2):
    return abs(val2 - val1)


# class GeneratorFactory:
#     def __init__(self, data, processing_func, extra_data=None, data_size=None, num_chunks=None):
#         self.data = tuple(data)  # data = img file names
#         self.chunks = map_list_as_chunks(data, (lambda val, ed: val), extra_data, num_chunks)
#         self.data_size = data_size
#         self.num_chunks = num_chunks
#         self._processing_func = processing_func
#
#     def get_next_chunked_generator(self):
#         # process_data: function to apply to each data point (file name)
#         for chunk in self.chunks:
#             yield (self._processing_func(data_point)
#                    for data_point in chunk)
#
#     def get_data_generator(self):
#         return (data_point for data_point in self.data)


if __name__ == '__main__':
    logging.info("START")

    data = [i for i in range(LIM)]

    t1 = default_timer()
    dist_matrix = main(data, LIM)
    t2 = default_timer()
    logging.info(f"Time: {round(t2-t1, 3)}s")

    show_dist_matrix(dist_matrix)
