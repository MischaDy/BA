# from multiprocessing import Pool, Lock, Manager, cpu_count

import dill
from multiprocess import Pool, Lock, Manager, cpu_count
from multiprocess_chunks import map_list_as_chunks

from math import pi, ceil, floor
import psutil
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import logging
from timeit import default_timer


logging.basicConfig(level=logging.INFO)

proc = psutil.Process(os.getpid())
proc.nice(psutil.NORMAL_PRIORITY_CLASS)

# meaning??
LIM = int(1e1)


def init(l):
    global lock
    lock = l


def main(data, mat_size):
    # lock solution: https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes#25558333
    l = Lock()

    gen_factory = GeneratorFactory(data, process_data)
    # chunked_gens = tuple(gen_factory.get_chunked_generators())

    # nums_list = list(data)
    # nums_chunks = chunk(nums_list, mat_size)
    dist_matrix_parts = [None for _ in range(len(data))]  # np.zeros(num_matrix_parts)

    # TODO: use result_queue to later distribute results to list
    result_queue = ...
    # TODO: Let workers read chunked_gens from a queue

    range_mat_parts = range(len(dist_matrix_parts))
    with Pool(initializer=init, initargs=(l)) as pool:
        for i in range_mat_parts:
            dist_matrix_parts[i] = pool.apply_async(compute_dist_matrix_part,
                                                    (gen_factory, data, compute_dist))

        for i in range_mat_parts:
            dist_matrix_parts[i].wait()
            dist_matrix_parts[i] = dist_matrix_parts[i].get()
        pool.close()
        pool.join()

    dist_matrix = join_matrix_parts(dist_matrix_parts)
    return dist_matrix


def process_data(data_point, extra_data):
    # dummy function
    return data_point


def compute_dist_matrix_part(gen_factory, compute_dist):
    # TODO: Rename!
    first_loop_values, first_loop_value_range = gen_factory.get_next_chunked_generator()
    second_loop_values, second_loop_value_range = gen_factory.get_data_generator()

    dist_matrix_part = torch.zeros(first_loop_value_range[1], second_loop_value_range[1])

    for ind1, val1 in enumerate(first_loop_values):
        for ind2, val2 in enumerate(second_loop_values):
            if ind1 % 1 == 0:
                lock.acquire()
                logging.info(f"{ind1}, {ind2}")
                lock.release()
            dist_matrix_part[ind1][ind2] = compute_dist(val1, val2)
    return dist_matrix_part



# def chunk(list_, mat_size):
#     num_chunks = cpu_count()  # = number of 'workers'
#
#     # size of matrix part = number of (complete, horizontal) lines of matrix
#     small_matrix_parts_size = mat_size // num_chunks
#     big_matrix_parts_size = small_matrix_parts_size + 1
#     num_high_load_workers = mat_size % num_chunks
#     num_low_load_workers = num_chunks - num_high_load_workers
#
#     chunked_iterable = []
#
#     for ind1 in range(num_high_load_workers):
#         chunked_iterable.append(
#             list_[ind1 * big_matrix_parts_size : (ind1+1) * big_matrix_parts_size]
#         )
#     ibm = (ind1+1) * big_matrix_parts_size
#     for ind2 in range(ibm, ibm + num_low_load_workers):
#         chunked_iterable.append(
#             list_[ind2 * small_matrix_parts_size : (ind2+1) * small_matrix_parts_size]
#         )
#
#     return chunked_iterable


def join_matrix_parts(matrix_parts, dim=0):
    return torch.cat(matrix_parts, dim)


def show_dist_matrix(dist_matrix):
    plt.imshow(dist_matrix.detach().numpy(), cmap=cm.YlOrRd)
    plt.colorbar()
    plt.show()


def compute_dist(val1, val2):
    return abs(val2 - val1)


# def compute_dist_matrix_from_nums(ns):
#     return compute_dist_matrix_part(ns, ns, mat_size, lock),


class GeneratorFactory:
    def __init__(self, data, processing_func, extra_data=None, data_size=None, num_chunks=None):
        self.data = tuple(data)  # data = img file names
        self.chunks = map_list_as_chunks(data, (lambda val, ed: val), extra_data, num_chunks)
        self.data_size = data_size
        self.num_chunks = num_chunks
        self._processing_func = processing_func

    def get_next_chunked_generator(self):
        # process_data: function to apply to each data point (file name)
        for chunk in self.chunks:
            yield (self._processing_func(data_point)
                   for data_point in chunk)

    def get_data_generator(self):
        return (data for data_point in self.data)


if __name__ == '__main__':
    logging.info("START")

    data = [i for i in range(LIM)]

    t1 = default_timer()
    dist_matrix = main(data, LIM)
    t2 = default_timer()
    logging.info(f"Time: {round(t2-t1, 3)}s")

    show_dist_matrix(dist_matrix)
