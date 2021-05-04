import os
from matplotlib import pyplot as plt

from Logic.ProperLogic.misc_helpers import get_every_nth_item, log_error, get_ext

MULTIPLOT = True
RESULTS_DIR_PATH = '../eval_lfw/results_thresholds_lfw4'  # '../eval_caltech/results_caltech4_rand'
WRITE = True
OUTPUT_PLOTS_DIR_PATH = '../eval_lfw/plots_lfw'
POSTFIX = 'lfw4'
YVALUES = ['f-measure', 'precision', 'recall']
X_MIN = 0.5
X_MAX = 1.4


def main(results_dir_path, yvalues):
    if not MULTIPLOT:
        for yvalue in yvalues:
            thres_and_met_to_yvalue = get_thres_and_met_to_yvalue(results_dir_path, yvalue)
            save_path = os.path.join(OUTPUT_PLOTS_DIR_PATH, yvalue + '.png') if WRITE else None
            plot_yvalues_vs_param(thres_and_met_to_yvalue, ylabels=yvalue, thres=True, metric=False, save_path=save_path)
        # plot_yvalues_vs_param(thres_and_met_to_yvalue, metric=True)
        # plot_f_measure_vs_threshold_and_metric(thres_and_met_to_yvalue)
        return

    postfix = f'_{POSTFIX}' if POSTFIX else ''
    save_path = os.path.join(OUTPUT_PLOTS_DIR_PATH, '_'.join(yvalues) + postfix + '.png') if WRITE else None
    plot_yvalues_vs_param(results_dir_path, ylabels=yvalues, thres=True, metric=False, x_min=X_MIN, x_max=X_MAX,
                          save_path=save_path)


def _setup_plots(title, xlabel, ylabel, x_min=0.5, x_max=1.0, eps=0.05):
    x_axes_limits = [x_min - eps, x_max + eps]
    y_axes_limits = [0 - eps, 1 + eps]

    fig, ax = plt.subplots()

    plt.title(title)
    # ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(x_axes_limits)
    ax.set_ylim(y_axes_limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_aspect('equal')
    # ax.grid(b=True, which='major', color='k', linestyle='--')
    return fig, ax


def get_thres_and_met_to_yvalue(results_dir_path, yvalue):
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
        thres_and_met_to_f_measure[thres_and_met] = results_dict[yvalue]

    return thres_and_met_to_f_measure


# def plot_f_measure_vs_metric(thres_and_met_to_yvalue):
#     print('hi')
#     met_f_measure = [
#         (thres_met.split('__')[0], f_measure)
#         for thres_met, f_measure in thres_and_met_to_yvalue.items()
#     ]
#     met_f_measure = [
#         (met.lstrip('L'), float(f_measure))
#         for met, f_measure in met_f_measure
#     ]
#     # TODO: Incorporate variance, too?
#
#     thres_met_f_measures_groups = group_pairs(met_f_measure, ret_dict=True)
#     map_dict_vals(thres_met_f_measures_groups, func=np.mean)
#     thres_met_f_measures_groups = sorted(thres_met_f_measures_groups.items())
#
#     # grouped_thres_met_f_measures = defaultdict(list)
#     # for met, f_measures in groupby(met_f_measure, key=lambda mf: mf[0]):
#     #     grouped_thres_met_f_measures[met].append(np.mean(f_measures))
#     # grouped_thres_met_f_measures = sorted(grouped_thres_met_f_measures.items())
#
#     xs = get_every_nth_item(thres_met_f_measures_groups, n=0)
#     ys = get_every_nth_item(thres_met_f_measures_groups, n=1)
#     plt.scatter(list(xs), list(ys))
#     plt.show()


def _plot_worker(xs, ys, xlabel='', ylabel='f-measure', title='', save_path=None):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0, 1

    x_eps, y_eps = (x_max - x_min) / 20, (y_max - y_min) / 20
    x_axis_limits = [x_min - x_eps, x_max + x_eps]
    y_axis_limits = [y_min - y_eps, y_max + y_eps]

    fig, ax = plt.subplots()
    ax.plot(list(xs), list(ys))
    plt.title(title)
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


def plot_yvalues_vs_param(results_dir_path, ylabels=None, thres=False, metric=False, x_min=0.5, x_max=1.0,
                          save_path=None):
    # dict_ = get_xs_and_ys(thres_and_met_to_yvalue, thres=thres, metric=metric)
    # params_and_f_measure_data_frame = pd.DataFrame(dict_)
    # print(params_and_f_measure_data_frame)

    xlabel = 'p-parameter of L_p-metric' if metric else 'threshold'
    if ylabels is None:
        ylabels = ['f-measure']

    title = ', '.join(ylabels) if len(ylabels) > 1 else f'{ylabels} vs. {xlabel}'
    fig, ax = _setup_plots(x_min=x_min, x_max=x_max, title=title, xlabel='threshold', ylabel='performance metric(s)')

    for ylabel in ylabels:
        thres_and_met_to_yvalue = get_thres_and_met_to_yvalue(results_dir_path, ylabel)
        xs, ys = get_xs_and_ys(thres_and_met_to_yvalue, thres=thres, metric=metric)
        ax.plot(xs, ys, label=ylabel)

    plt.legend(loc='lower left')
    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


def get_xs_and_ys(thres_and_met_to_f_measure, thres=False, metric=False):
    if thres:
        ind = 1
    elif metric:
        ind = 0
    else:
        log_error('thres or metric must be true')
        return

    param_f_measure = [
        (thres_met.split('__')[ind], f_measure)
        for thres_met, f_measure in thres_and_met_to_f_measure.items()
    ]
    param_f_measure = [
        (float(param.lstrip('LT')), float(f_measure))
        for param, f_measure in param_f_measure
    ]
    # TODO: Incorporate variance, too?

    # thres_met_f_measures_groups_dict = group_pairs(param_f_measure, ret_dict=True)
    # # map_dict_vals(thres_met_f_measures_groups_dict, func=np.mean)
    # # return thres_met_f_measures_groups_dict
    #
    # thres_met_f_measures_groups = sorted(thres_met_f_measures_groups_dict.items())

    # grouped_thres_met_f_measures = defaultdict(list)
    # for met, f_measures in groupby(met_f_measure, key=lambda mf: mf[0]):
    #     grouped_thres_met_f_measures[met].append(np.mean(f_measures))
    # grouped_thres_met_f_measures = sorted(grouped_thres_met_f_measures.items())

    xs = get_every_nth_item(param_f_measure, n=0)
    ys = get_every_nth_item(param_f_measure, n=1)
    return list(xs), list(ys)


def plot_f_measure_vs_threshold_and_metric(thres_and_met_to_f_measure):
    pass


def process_lines(lines, sep=': '):
    processed_lines = dict(line.split(sep) for line in lines)
    return processed_lines


if __name__ == '__main__':
    main(RESULTS_DIR_PATH, YVALUES)
