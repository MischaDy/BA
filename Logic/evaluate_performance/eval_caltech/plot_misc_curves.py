from functools import partial
from itertools import starmap

import numpy as np
from matplotlib import pyplot as plt

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.misc_helpers import starfilter, ignore_first_n_args_decorator, get_every_nth_item, \
    ignore_last_n_args_decorator, unique_everseen, get_ext
from Logic.evaluate_performance.eval_caltech.evaluate_accuracy_caltech import IMAGES_PATH, get_img_name_to_id_dict, \
    caltech_are_same_person_func
from Logic.evaluate_performance.eval_caltech.plot_roc_caltech import plot_rocs, make_save_path
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager


METRIC = 2
THRESHOLD = 0.73
TP_VS_THRES_SAVE_PATH = 'plots_caltech/caltech_tp_vs_thres'

# TODO: Make Threshold vs. FP / TP / f-measure plots!
#       --> Compare to proposed threshold / threshold curve in paper (remember their cut-off criterion)
#       --> Generally: discuss!
#       --> Also vary metric?


def main_plot_misc(images_path):
    save_path = make_save_path(TP_VS_THRES_SAVE_PATH, metric=METRIC, threshold=THRESHOLD, format_='svg')
    thresholds_and_tps = get_thresholds_and_tps(images_path)
    plot_thresholds_vs_tps(thresholds_and_tps, save_path=save_path)


def get_thresholds_and_tps(images_path, eps=10):
    embeddings_with_ids = list(EvalDBManager.get_all_embeddings(with_ids=True, as_dict=False))
    emb_id_to_img_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path)
    img_name_to_person_id_dict = get_img_name_to_id_dict(images_path)

    thresholds_and_tps = []
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
        tp_rate = compute_tp_rate(thresholds, sorted_dists_to_start_emb, matches_with_start_emb)
        thresholds_and_tps.append((thresholds, tp_rate))
    return thresholds_and_tps


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


def plot_thresholds_vs_tps(thresholds_and_tps, y_eps=0.05, title=None, save_path=None):
    fig, ax = _plot_thresholds_vs_tps_helper(thresholds_and_tps, title=title, y_eps=y_eps)

    for thres, tp_rate in thresholds_and_tps:
        ax.plot(thres, tp_rate)

    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


def _plot_thresholds_vs_tps_helper(thresholds_and_tps, title=None, x_eps=None, y_eps=0.05):
    thresholds = list(get_every_nth_item(thresholds_and_tps, n=0))
    min_thres, max_thres = min(map(min, thresholds)), max(map(max, thresholds))
    if x_eps is None:
        x_eps = (max_thres - min_thres) / 20  # 5% margin
    x_axis_limits = [min_thres - x_eps, max_thres + x_eps]
    y_axis_limits = [0 - y_eps, 1 + y_eps]

    fig, ax = plt.subplots()
    xlabel = 'Thresholds'
    ylabel = 'True Positive Rate'
    if title is None:
        title = 'Threshold vs. TP Rate'

    plt.title(title)
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_aspect('equal')
    return fig, ax


def plot_bad_rocs_same_person_colors(bad_emb_ids_person_ids_img_names, emb_id_to_fps_and_tps):
    bad_emb_ids = list(get_every_nth_item(bad_emb_ids_person_ids_img_names, n=0))

    @ignore_last_n_args_decorator(n=1)
    def is_bad(emb_id):
        return emb_id in bad_emb_ids

    bad_emb_id_to_fps_and_tps = dict(
        starfilter(is_bad, emb_id_to_fps_and_tps.items())
    )

    unique_persons_triplets = list(unique_everseen(bad_emb_ids_person_ids_img_names, key=lambda triplet: triplet[1]))

    bad_emb_id_to_person_id_dict = dict(
        (emb_id,
         person_id if person_id != 'person_11' else '5 x person_11')
        for emb_id, person_id, _ in unique_persons_triplets
    )

    person_ids = get_every_nth_item(unique_persons_triplets, n=1)
    colors = ['b', 'g', 'r', 'c', 'm']
    person_id_to_color_dict = dict(zip(person_ids, colors))

    emb_id_to_equal_colors = dict(
        (emb_id, person_id_to_color_dict[person_id])
        for emb_id, person_id, _ in bad_emb_ids_person_ids_img_names
    )

    plot_rocs(bad_emb_id_to_fps_and_tps, bad_emb_id_to_person_id_dict, emb_id_to_equal_colors, title='Bad ROCs',
              save_path=BAD_ROCS_PERSON_SAME_COLOR_SAVE_PATH)


def print_bad_rocs(images_path, bad_rocs, write_output=True):
    emb_id_to_img_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path)
    img_name_to_person_id_dict = get_img_name_to_id_dict(images_path)
    bad_roc_triplets = []
    bad_roc_lines = []
    for start_emb_id in get_every_nth_item(bad_rocs):
        img_name = emb_id_to_img_name_dict[start_emb_id]
        person_id = img_name_to_person_id_dict[img_name]
        bad_roc_lines.append(f'emb_id: {start_emb_id}  |  person_id: {person_id}  |  img_name: {img_name}' + '\n')
        bad_roc_triplets.append((start_emb_id, person_id, img_name))
    if PRINT_OUTPUT:
        print('\n'.join(bad_roc_lines))
    if write_output:
        with open(BAD_ROCS_TXT_PATH, 'w') as file:
            file.writelines(bad_roc_lines)
    return bad_roc_triplets


def get_bad_rocs(emb_id_to_fps_and_tps):
    bad_rocs = starfilter(is_bad_roc, emb_id_to_fps_and_tps.items())
    return bad_rocs


@ignore_first_n_args_decorator(n=1)
def is_bad_roc(fp_and_tp_rates):
    tp_rate = fp_and_tp_rates[1]
    return np.percentile(tp_rate, q=20) < 0.9


if __name__ == '__main__':
    main_plot_misc(IMAGES_PATH)
