# from multiprocessing import Pool, Lock, Manager, cpu_count

import dill
from multiprocess import Pool, Lock, Manager, cpu_count

from math import pi, ceil
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



# Idea: pass a generator factory, so generator can be recreated!
def main(nums_gen, mat_size):  # , queue):
    nums_list = list(nums_gen)
    nums_chunks = chunk(nums_list, mat_size)
    dist_matrix_parts = [0 for _ in range(len(nums_chunks))]  # np.zeros(num_matrix_parts)

    with Pool() as pool:
        for i, num_part in enumerate(nums_chunks):
            dist_matrix_parts[i] = pool.apply_async(compute_dist_matrix_part,
                                                    (num_part, nums_gen, compute_dist))  # , queue))

        for i in range(len(dist_matrix_parts)):
            dist_matrix_parts[i].wait()
            dist_matrix_parts[i] = dist_matrix_parts[i].get()
        pool.close()

    dist_matrix = join_matrix_parts(dist_matrix_parts)
    return dist_matrix


def compute_dist_matrix_part(first_loop_values, second_loop_values, compute_dist):  # , queue):
    # TODO: Rename!
    dist_matrix_part = torch.zeros(len(first_loop_values), len(second_loop_values))

    for ind1, val1 in enumerate(first_loop_values):
        for ind2, val2 in enumerate(second_loop_values):
            # if ind2 % 50 == 1:
            #     lock = queue.get(block=True)
            #     lock.acquire()
            #     logging.info(f"{ind1}, {ind2}")
            #     lock.release()
            #     queue.put(lock)
            dist_matrix_part[ind1][ind2] = compute_dist(val1, val2)
    return dist_matrix_part


def chunk(list_, mat_size):
    num_chunks = cpu_count()  # = number of 'workers'

    # size of matrix part = number of (complete, horizontal) lines of matrix
    small_matrix_parts_size = mat_size // num_chunks
    big_matrix_parts_size = small_matrix_parts_size + 1
    num_high_load_workers = mat_size % num_chunks
    num_low_load_workers = num_chunks - num_high_load_workers

    chunked_iterable = []

    for ind1 in range(num_high_load_workers):
        chunked_iterable.append(
            list_[ind1 * big_matrix_parts_size : (ind1+1) * big_matrix_parts_size]
        )
    ibm = (ind1+1) * big_matrix_parts_size
    for ind2 in range(ibm, ibm + num_low_load_workers):
        chunked_iterable.append(
            list_[ind2 * small_matrix_parts_size : (ind2+1) * small_matrix_parts_size]
        )

    return chunked_iterable


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


if __name__ == '__main__':
    # lock = Lock()
    #
    # manager = Manager()
    # q = manager.Queue()
    # q.put(lock)

    logging.info("START")


    x = (i for i in range(LIM))

    def f(gen):
        def g():
            for elem in gen:
                yield elem
        return g

    g = f(x)

    t1 = default_timer()
    dist_matrix = main(g, LIM)  #, q)
    t2 = default_timer()
    logging.info(f"Time: {round(t2-t1, 3)}s")


    show_dist_matrix(dist_matrix)
