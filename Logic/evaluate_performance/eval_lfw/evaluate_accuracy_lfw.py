import os

import numpy as np

from Logic.evaluate_performance.evaluate_accuracy import run_metric_evaluation

IMAGES_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\Backup unneeded files\BA\Logic\my_test\subset_cplfw\images'
SAVE_RESULTS = False
SAVE_PATH = 'results_thresholds_lfw4' if SAVE_RESULTS else None
DELETE_LOCAL_DB_FILE = False
DELETE_CENTRAL_DB_FILE = False
CLEAR_CLUSTERS = True


THRESHOLDS = list(map(lambda t: round(t, 2), np.linspace(0.9, 1.4, num=11)))  # ==>  step size = 0.05
METRICS = [2]  # , 1.5, 1, 0.75, 0.5, 0.3, 0.2, 0.1, 0]


def lfw_are_same_person_func(emb_id1, emb_id2, emb_id_to_name_dict):
    emb_name1, emb_name2 = map(emb_id_to_name_dict.get, [emb_id1, emb_id2])
    person1_numbered_name, _ = os.path.splitext(emb_name1)
    person2_numbered_name, _ = os.path.splitext(emb_name2)
    person1_name = _rstrip_underscored_part(person1_numbered_name)
    person2_name = _rstrip_underscored_part(person2_numbered_name)
    return person1_name == person2_name


def _rstrip_underscored_part(string):
    """Remove part after rightmost underscore in string if such a part exists."""
    underscore_ind = string.rfind('_')
    if underscore_ind != -1:
        return string[:underscore_ind]
    return string


if __name__ == '__main__':
    run_metric_evaluation(IMAGES_PATH, lfw_are_same_person_func, SAVE_PATH, THRESHOLDS, METRICS, DELETE_CENTRAL_DB_FILE,
                          DELETE_LOCAL_DB_FILE, CLEAR_CLUSTERS)
