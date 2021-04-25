import numpy as np

from Logic.ProperLogic.misc_helpers import starfilter, ignore_first_n_args_decorator, get_every_nth_item, \
    ignore_last_n_args_decorator, unique_everseen
from Logic.evaluate_performance.eval_caltech.evaluate_accuracy_caltech import IMAGES_PATH, get_img_name_to_id_dict
from Logic.evaluate_performance.eval_caltech.plot_roc_caltech import get_emb_id_to_fps_and_tps, plot_rocs
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager


BAD_ROCS_PERSON_SAME_COLOR_SAVE_PATH = 'plots_caltech/caltech_bad_rocs_person_same_color.svg'
BAD_ROCS_TXT_PATH = 'bad_rocs.txt'
WRITE_OUTPUT = False
PRINT_OUTPUT = False


# TODO: Make Threshold vs. FP / TP / f-measure plots!
#       --> Compare to proposed threshold / threshold curve in paper (remember their cut-off criterion)
#       --> Generally: discuss!
#       --> Also vary metric?
# TODO: What is up with the 3 persons (showing similar trend)?
#       --> lighting issues?
#           --> any other images (esp. in same sets!) with flash but no problem?
#           --> problem persists after cropping, and after readjusting brightness?
# TODO: What is up with the the 1 person?
#       --> Small number of images?
#           --> Other folder with equally small, but > 1 number of people?
#       --> Plot 5 individual lines!
#           --> How are they inter-related?!
#           --> why are there "2 paths"?
# TODO: Time working with that dataset until everything is sorted correctly!
# TODO: Other evaluation methods needed?
# TODO: Average all ROC curves?


def main_investigate(images_path):
    emb_id_to_fps_and_tps = get_emb_id_to_fps_and_tps(images_path, use_all=True)
    # emb_id, fps, tps
    bad_rocs = get_bad_rocs(emb_id_to_fps_and_tps)
    bad_emb_ids_person_ids_img_names = print_bad_rocs(images_path, bad_rocs, write_output=WRITE_OUTPUT)
    plot_bad_rocs_same_person_colors(bad_emb_ids_person_ids_img_names, emb_id_to_fps_and_tps)


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
    main_investigate(IMAGES_PATH)
