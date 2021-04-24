import operator
from functools import partial
import random

import matplotlib.pyplot as plt
import numpy as np

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.misc_helpers import take_first
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager
from evaluate_accuracy_caltech import IMAGES_PATH, get_img_name_to_id_dict, caltech_are_same_person_func

LABEL = '...label...'
USE_ALL = True
USE_RANDOM_START_IDS = False
NUM_RANDOM_START_IDS = 10


METRIC = 2
THRESHOLD = 0.73

SAVE_FORMAT = 'svg'
SAVE_PATH = f"plots_caltech/caltech_477_rocs___L{METRIC}__T{str(THRESHOLD).replace('.', '_pt_')}.{SAVE_FORMAT}"


def main(images_path):
    emb_id_to_fps_and_tps = get_emb_id_to_fps_and_tps(images_path, use_all=USE_ALL,
                                                      use_random_start_ids=USE_RANDOM_START_IDS,
                                                      num_random_start_ids=NUM_RANDOM_START_IDS)
    fp_and_tp_rates = list(emb_id_to_fps_and_tps.items())
    plot_rocs(fp_and_tp_rates)


def get_emb_id_to_fps_and_tps(images_path, start_embs_ids=None, use_all=False, use_random_start_ids=False,
                              num_random_start_ids=1):
    embeddings_with_ids = EvalDBManager.get_all_embeddings(with_ids=True, as_dict=True)
    if start_embs_ids is None:
        emb_ids = embeddings_with_ids.keys()
        if use_random_start_ids:
            start_embs_ids = random.choices(list(emb_ids), k=num_random_start_ids)
        elif use_all:
            start_embs_ids = emb_ids
        else:
            # use only the first one
            start_embs_ids = [take_first(emb_ids)]

    if use_all:
        start_embs = embeddings_with_ids.values()
    else:
        start_embs = map(embeddings_with_ids.get, start_embs_ids)

    emb_id_to_img_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path)
    img_name_to_person_id_dict = get_img_name_to_id_dict(images_path)

    emb_id_to_fps_and_tps = dict()
    for start_emb_id, start_emb in zip(start_embs_ids, start_embs):
        compute_dist_to_start_emb = partial(Cluster.compute_dist, start_emb)
        # sort by distance to start emb
        sorted_dists_to_start_emb = sorted(embeddings_with_ids.items(),
                                           key=lambda emb_and_id: compute_dist_to_start_emb(emb_and_id[1]))

        is_start_person = partial(caltech_are_same_person_func, start_emb_id,
                                  emb_id_to_img_name_dict=emb_id_to_img_name_dict,
                                  img_name_to_person_id_dict=img_name_to_person_id_dict)

        matches_with_start_emb = [
            is_start_person(emb_id)
            for emb_id, _ in sorted_dists_to_start_emb
        ]
        mismatches_with_start_emb = list(map(operator.not_, matches_with_start_emb))

        num_matches = sum(matches_with_start_emb)
        num_mismatches = len(matches_with_start_emb) - num_matches
        tp_rate = np.cumsum(matches_with_start_emb) / num_matches
        fp_rate = np.cumsum(mismatches_with_start_emb) / num_mismatches
        emb_id_to_fps_and_tps[start_emb_id] = (fp_rate, tp_rate)
    return emb_id_to_fps_and_tps


# def caltech_are_same_person_func(emb_id1, emb_id2, emb_id_to_img_name_dict, img_name_to_person_id_dict):
#     img_name1, img_name2 = map(emb_id_to_img_name_dict.get, [emb_id1, emb_id2])
#     return img_name_to_person_id_dict[img_name1] == img_name_to_person_id_dict[img_name2]


def plot_roc(fp_rate, tp_rate, eps=0.05):
    _plot_roc_helper(eps)
    plt.plot(fp_rate, tp_rate, 'b', label=LABEL)
    plt.show()


def plot_rocs(fp_and_tp_rates, eps=0.05):
    fig, ax = _plot_roc_helper(eps=eps)

    for counter, (fp_rate, tp_rate) in enumerate(fp_and_tp_rates, start=1):
        ax.plot(fp_rate, tp_rate, label=str(counter))
    if SAVE_PATH is not None:
        plt.savefig(SAVE_PATH, format=SAVE_FORMAT)
    plt.show()


def _plot_roc_helper(eps=0.05, will_plot_multi=False):
    axes_limits = [0 - eps, 1 + eps]
    plot_range = [0, 1]

    fig, ax = plt.subplots()
    title = 'ROC'
    xlabel = 'False Positive Rate'
    ylabel = 'True Positive Rate'
    if will_plot_multi:
        title += 's'
        xlabel += 's'
        ylabel += 's'
    plt.title(title)
    # ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(axes_limits)
    ax.set_ylim(axes_limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    # ax.grid(b=True, which='major', color='k', linestyle='--')
    # plt.legend(loc='lower right')
    ax.plot(plot_range, plot_range, 'k--')
    return fig, ax


if __name__ == '__main__':
    main(IMAGES_PATH)
