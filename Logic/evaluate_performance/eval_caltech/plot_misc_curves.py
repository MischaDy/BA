import operator
from functools import partial
from itertools import starmap

import numpy as np
from matplotlib import pyplot as plt

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.misc_helpers import get_every_nth_item, get_ext
from Logic.evaluate_performance.eval_caltech.evaluate_accuracy_caltech import IMAGES_PATH, get_img_name_to_id_dict, \
    caltech_are_same_person_func
from Logic.evaluate_performance.eval_caltech.plot_roc_caltech import make_save_path
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager


METRIC = 2
THRESHOLD = 0.73
TP_VS_THRES_SAVE_PATH = 'plots_caltech/caltech_tp_vs_thres'
FP_VS_THRES_SAVE_PATH = 'plots_caltech/caltech_fp_vs_thres'
F_MEASURE_VS_THRES_SAVE_PATH = 'plots_caltech/caltech_f_measure_vs_thres'

# TODO: Note: FN etc. rates are *with regards to start cluster*!!!

# TODO: Are you really calculating face PAIRS?!?!!
#       Or does it not matter bc. its face pairs, but focusing only on start emb (not even its cluster)?

# TODO: Make Threshold vs. f-measure plots!
#       --> Compare plots to proposed threshold (curve) in paper -> (remember their cut-off criterion!)
#       --> Generally: discuss!
#       --> Also vary metric?


def main_plot_misc(images_path):
    # TPs
    # tp_save_path = make_save_path(TP_VS_THRES_SAVE_PATH, metric=METRIC, threshold=THRESHOLD, format_='svg')
    # TODO: Remove ret_ option
    thresholds_and_tps = get_thresholds_and_pos_rate(images_path, True)  # , ret_abs_num_pos=True)
    # FPs
    # fp_save_path = make_save_path(FP_VS_THRES_SAVE_PATH, metric=METRIC, threshold=THRESHOLD, format_='svg')
    thresholds_and_fps = get_thresholds_and_pos_rate(images_path, False)

    # plot_thresholds_vs_pos(thresholds_and_tps, pos_type=True, save_path=tp_save_path)
    # plot_thresholds_vs_pos(thresholds_and_fps, pos_type=False, save_path=fp_save_path)

    # f-measure
    f_measure_save_path = make_save_path(F_MEASURE_VS_THRES_SAVE_PATH, metric=METRIC, threshold=THRESHOLD,
                                         format_='svg')
    thresholds_and_f_measures = get_thresholds_and_f_measures(thresholds_and_tps=thresholds_and_tps,
                                                              thresholds_and_fps=thresholds_and_fps)

    plot_thresholds_vs_f_measures(thresholds_and_f_measures, save_path=f_measure_save_path)


def get_thresholds_and_f_measures(images_path=None, thresholds_and_tps=None, thresholds_and_fps=None):
    if images_path is None and (thresholds_and_tps is None or thresholds_and_fps is None):
        raise ValueError('Either both thresholds or images_path must be provided')
    if thresholds_and_tps is None:
        thresholds_and_tps = get_thresholds_and_pos_rate(images_path, True)
    if thresholds_and_fps is None:
        thresholds_and_fps = get_thresholds_and_pos_rate(images_path, False)

    thresholds_and_fns = get_thresholds_and_fn_rate(thresholds_and_tps)

    get_every_2nd_item = partial(get_every_nth_item, n=1)
    tp_rates, fp_rates, fn_rates = (list(get_every_2nd_item(iterable))
                                    for iterable in [thresholds_and_tps, thresholds_and_fps, thresholds_and_fns])

    precisions = starmap(compute_precision, zip(tp_rates, fp_rates))
    recalls = starmap(compute_recall, zip(tp_rates, fn_rates))
    f_measures = starmap(compute_f_measure, zip(precisions, recalls))

    thresholds = get_every_nth_item(thresholds_and_fns, n=0)
    thresholds_and_f_measures = list(zip(thresholds, f_measures))
    return thresholds_and_f_measures


def compute_precision(tp_rate, fp_rate):
    return tp_rate / (tp_rate + fp_rate)


def compute_recall(tp_rate, fn_rate):
    return tp_rate / (tp_rate + fn_rate)


def compute_f_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def get_thresholds_and_fn_rate(thresholds_and_tps):
    """FN etc. rates *with regards to start cluster*!!!"""
    thresholds_and_fns = []
    for thresholds, tp_rate in thresholds_and_tps:
        fn_rate = 1 - tp_rate
        thresholds_and_fns.append((thresholds, fn_rate))
    return thresholds_and_fns


def count_false_negatives(self, clusters, emb_id_to_name_dict):
    """
    Number of face pairs incorrectly clustered to different clusters

    :return: Number of
    """

    def does_match(emb_id_pair):
        # Count iff result of check (yes/no) is same as wanted type (true/false positives)
        return self.are_same_person_func(*emb_id_pair, emb_id_to_name_dict)

    clusters_embedding_pairs = self._get_inter_clusters_embedding_id_pairs(clusters)
    total_negatives = 0
    for count, embedding_pairs in enumerate(clusters_embedding_pairs, start=1):
        # if count % INTERCLUSTER_ITERATIONS_PROGRESS == 0:
        #     logging.info(f'--- --- embeddings iteration: {count}')
        cluster_negatives = sum(map(does_match, embedding_pairs))
        total_negatives += cluster_negatives
    return total_negatives


def get_thresholds_and_pos_rate(images_path, pos_type, eps=10, ret_abs_num_pos=False):
    compute_pos_rate = compute_tp_rate if pos_type else compute_fp_rate

    embeddings_with_ids = list(EvalDBManager.get_all_embeddings(with_ids=True, as_dict=False))
    emb_id_to_img_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path)
    img_name_to_person_id_dict = get_img_name_to_id_dict(images_path)

    num_pos_list = []
    thresholds_and_pos = []
    for start_emb_id, start_emb in embeddings_with_ids:
        compute_dist_to_start_emb = partial(Cluster.compute_dist, start_emb)

        def get_emb_id_and_dist_to_start_emb(emb_id, emb):
            return emb_id, compute_dist_to_start_emb(emb)

        emb_ids_and_dists_to_start_emb = list(starmap(get_emb_id_and_dist_to_start_emb, embeddings_with_ids))
        # compute and sort distances to start emb
        sorted_embs_and_dists_to_start_emb = sorted(emb_ids_and_dists_to_start_emb,
                                                    key=lambda emb_and_dist: emb_and_dist[1])
        is_start_person = partial(caltech_are_same_person_func, start_emb_id,
                                  emb_id_to_img_name_dict=emb_id_to_img_name_dict,
                                  img_name_to_person_id_dict=img_name_to_person_id_dict)

        sorted_emb_ids = get_every_nth_item(sorted_embs_and_dists_to_start_emb, n=0)
        matches_with_start_emb = list(map(is_start_person, sorted_emb_ids))

        sorted_dists_to_start_emb = list(get_every_nth_item(sorted_embs_and_dists_to_start_emb, n=1))
        max_dist = max(sorted_dists_to_start_emb)
        thresholds = np.linspace(0, max_dist, num=max_dist * 100 + eps)
        pos_rate = compute_pos_rate(thresholds, sorted_dists_to_start_emb, matches_with_start_emb)
        thresholds_and_pos.append((thresholds, pos_rate))
        num_pos_list.append(sum(matches_with_start_emb))

    if ret_abs_num_pos:
        return thresholds_and_pos, num_pos_list
    return thresholds_and_pos


def compute_tp_rate(thresholds, sorted_dists_to_start_emb, matches_with_start_emb):
    tp_rate = np.zeros_like(thresholds)

    dists_pointer = 0
    next_dist = sorted_dists_to_start_emb[dists_pointer]
    seen_matches = 0
    break_for_loop = False
    thres_ind = 0
    for thres_ind, thres in enumerate(thresholds):
        while thres >= next_dist:
            if matches_with_start_emb[dists_pointer]:
                seen_matches += 1
            dists_pointer += 1
            try:
                next_dist = sorted_dists_to_start_emb[dists_pointer]
            except IndexError:
                # already processed all embeddings distances
                break_for_loop = True
                break
        if break_for_loop:
            break
        tp_rate[thres_ind] += seen_matches
    if break_for_loop:
        tp_rate[thres_ind:] += seen_matches
    num_tps = tp_rate[-1]
    return tp_rate / num_tps


def compute_fp_rate(thresholds, sorted_dists_to_start_emb, matches_with_start_emb):
    mismatches_with_start_emb = list(map(operator.not_, matches_with_start_emb))
    return compute_tp_rate(thresholds, sorted_dists_to_start_emb, mismatches_with_start_emb)


def plot_thresholds_vs_f_measures(thresholds_and_f_measures, y_eps=0.05, title=None, save_path=None):
    ylabel = 'f-measure'
    fig, ax = _plot_thresholds_vs_params_helper(thresholds_and_f_measures, ylabel, title=title, y_eps=y_eps)

    for thres, f_measure in thresholds_and_f_measures:
        ax.plot(thres, f_measure)

    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


def plot_thresholds_vs_pos(thresholds_and_pos, pos_type, y_eps=0.05, title=None, save_path=None):
    fig, ax = _plot_thresholds_vs_positives_helper(thresholds_and_pos, pos_type, title=title, y_eps=y_eps)

    for thres, pos_rate in thresholds_and_pos:
        ax.plot(thres, pos_rate)

    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


def _plot_thresholds_vs_positives_helper(thresholds_and_pos, pos_type, title=None, x_eps=None, y_eps=0.05):
    ylabel = f'{pos_type} Positive Rate'
    return _plot_thresholds_vs_params_helper(thresholds_and_pos, ylabel, title=title, x_eps=x_eps, y_eps=y_eps)


def _plot_thresholds_vs_params_helper(thresholds_and_params, ylabel, title=None, x_eps=None, y_eps=0.05):
    thresholds = list(get_every_nth_item(thresholds_and_params, n=0))
    min_thres, max_thres = min(map(min, thresholds)), max(map(max, thresholds))
    if x_eps is None:
        x_eps = (max_thres - min_thres) / 20  # 5% margin
    x_axis_limits = [min_thres - x_eps, max_thres + x_eps]
    y_axis_limits = [0 - y_eps, 1 + y_eps]

    fig, ax = plt.subplots()
    xlabel = 'Thresholds'
    if title is None:
        title = f'Threshold vs. {ylabel}'

    plt.title(title)
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


if __name__ == '__main__':
    main_plot_misc(IMAGES_PATH)
