import os
from collections import defaultdict
from itertools import groupby

import numpy as np
from matplotlib import pyplot as plt

from Logic.ProperLogic.misc_helpers import get_every_nth_item, group_pairs, map_dict_vals

RESULTS_DIR_PATH = '../results'


def main(results_dir_path):
    result_files = os.listdir(results_dir_path)
    result_file_paths = [os.path.join(results_dir_path, result_file) for result_file in result_files]
    processed_file_names = []
    for result_file in result_files:
        file_name_no_ext = os.path.splitext(result_file)[0]
        processed_file_name = file_name_no_ext.split('___')[1].replace('_point_', '.')
        processed_file_names.append(processed_file_name)

    file_contents = dict()
    for file_name, result_path in zip(processed_file_names, result_file_paths):
        with open(result_path, 'r') as file:
            lines = file.readlines()
            file_contents[file_name] = process_lines(lines)

    thres_and_met_to_f_measure = dict()
    for thres_and_met, results_dict in file_contents.items():
        thres_and_met_to_f_measure[thres_and_met] = results_dict['f-measure']

    plot_f_measure_vs_metric(thres_and_met_to_f_measure)
    plot_f_measure_vs_threshold(thres_and_met_to_f_measure)
    plot_f_measure_vs_threshold_and_metric(thres_and_met_to_f_measure)


def plot_f_measure_vs_metric(thres_and_met_to_f_measure):
    print('hi')
    met_f_measure = [
        (thres_met.split('__')[0], f_measure)
        for thres_met, f_measure in thres_and_met_to_f_measure.items()
    ]
    met_f_measure = [
        (met.lstrip('L'), float(f_measure))
        for met, f_measure in met_f_measure
    ]
    # TODO: Incorporate variance, too?

    thres_met_f_measures_groups = group_pairs(met_f_measure, ret_dict=True)
    map_dict_vals(thres_met_f_measures_groups, func=np.mean)
    thres_met_f_measures_groups = sorted(thres_met_f_measures_groups.items())

    # grouped_thres_met_f_measures = defaultdict(list)
    # for met, f_measures in groupby(met_f_measure, key=lambda mf: mf[0]):
    #     grouped_thres_met_f_measures[met].append(np.mean(f_measures))
    # grouped_thres_met_f_measures = sorted(grouped_thres_met_f_measures.items())

    xs = get_every_nth_item(thres_met_f_measures_groups, n=0)
    ys = get_every_nth_item(thres_met_f_measures_groups, n=1)
    plt.scatter(list(xs), list(ys))
    plt.show()


def plot_f_measure_vs_threshold(thres_and_met_to_f_measure):
    pass


def plot_f_measure_vs_threshold_and_metric(thres_and_met_to_f_measure):
    pass


def process_lines(lines, sep=': '):
    processed_lines = dict(line.split(sep) for line in lines)
    return processed_lines


if __name__ == '__main__':
    main(RESULTS_DIR_PATH)