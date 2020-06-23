# from multiprocessing import Pool, Lock, Manager, cpu_count

import dill
from multiprocess import (JoinableQueue, Queue, Process, Lock,  # Pool, Manager,
                          cpu_count, freeze_support)

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

# Set priority of current process
psutil.Process(os.getpid()).nice(psutil.NORMAL_PRIORITY_CLASS)

LIM = int(1e1)


class Consumer(Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        next_task = self.task_queue.get()
        self.task_queue.task_done()

        first_loop_vals, second_loop_vals, dist_func, lock, id_ = next_task()

        result = compute_dist_matrix_part(first_loop_vals, second_loop_vals, dist_func)
        self.task_queue.task_done()
        self.result_queue.put((id_, result))

        # lock for printing
        lock.acquire()
        logging.info(f'Task {id_} finished.')
        lock.release()


class Task:
    def __init__(self, *args):
        self.args = args

    def __call__(self):
        return self.args

    def __str__(self):
        return f'{self.args}'


# TODO: idea - make generator chunky!
# TODO: Use Queue to get results of workers (and to pass tasks to them?)
def main(nums, mat_size):
    # Credit for idea of Tasks + Results Queues: https://pymotw.com/3/multiprocessing/communication.html
    # Adaptations to classes
    nums_list = list(nums)
    nums_chunks = chunk(nums_list, mat_size)
    dist_matrix_parts = [None for _ in nums_chunks]

    results = Queue()
    tasks = JoinableQueue()

    consumers = [Consumer(tasks, results) for _ in nums_chunks]
    for consumer in consumers:
        consumer.start()

    # Enqueue tasks
    lock = Lock()
    for id_, num_chunk in enumerate(nums_chunks):
        tasks.put(Task(num_chunk, nums, compute_dist, lock, results, id_))

    tasks.join()
    # for proc in consumers:
    #     proc.join()

    dist_matrix_parts = [results.get() for _ in nums_chunks]
    return join_matrix_parts(sorted(dist_matrix_parts, key=lambda tup: tup[0]))


    # nums_list = list(nums)
    # nums_chunks = chunk(nums_list, mat_size)
    # dist_matrix_parts = [0 for _ in range(len(nums_chunks))]
    #
    # with Pool() as pool:
    #     for i, num_part in enumerate(nums_chunks):
    #         dist_matrix_parts[i] = pool.apply_async(compute_dist_matrix_part,
    #                                                 (num_part, nums, compute_dist))  # , queue))
    #
    #     for i in range(len(dist_matrix_parts)):
    #         dist_matrix_parts[i].wait()
    #         dist_matrix_parts[i] = dist_matrix_parts[i].get()
    #     pool.close()
    #
    # dist_matrix = join_matrix_parts(dist_matrix_parts)
    # return dist_matrix


def compute_dist_matrix_part(first_loop_values, second_loop_values, compute_dist, lock):  # , queue):
    # TODO: Rename!
    dist_matrix_part = torch.zeros(len(first_loop_values), len(second_loop_values))

    for ind1, val1 in enumerate(first_loop_values):
        for ind2, val2 in enumerate(second_loop_values):
            # lock + logging test
            if ind1 % 1 == 0:
                lock.acquire()
                logging.info(f"{ind1}, {ind2}")
                lock.release()
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
    freeze_support()

    logging.info("START")

    x = [i for i in range(LIM)]  # TODO: to generator!

    t1 = default_timer()
    dist_matrix = main(x, LIM)  # , q)
    t2 = default_timer()
    logging.info(f"Time: {round(t2-t1, 3)}s")

    show_dist_matrix(dist_matrix)


# from multiprocessing import Process, Pipe
#
# def f(conn):
#     conn.send([42, None, 'hello'])
#     conn.close()
#
# if __name__ == '__main__':
#     parent_conn, child_conn = Pipe()
#     p = Process(target=f, args=(child_conn,))
#     p.start()
#     print(parent_conn.recv())   # prints "[42, None, 'hello']"
#     p.join()
