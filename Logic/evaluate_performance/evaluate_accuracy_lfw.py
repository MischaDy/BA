import os

from Logic.evaluate_performance.evaluate_accuracy import run_metric_evaluation
import numpy as np

IMAGES_PATH = r'..\my_test\subset_cplfw\images'
SAVE_PATH = 'results_thresholds'
DELETE_LOCAL_DB_FILE = False
DELETE_CENTRAL_DB_FILE = False
CLEAR_CLUSTERS = True


THRESHOLDS = np.linspace(0.5, 1.0, num=11)  # ==>  step size = 0.05
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
    run_metric_evaluation(IMAGES_PATH, THRESHOLDS, METRICS, SAVE_PATH, DELETE_LOCAL_DB_FILE, DELETE_CENTRAL_DB_FILE,
                          CLEAR_CLUSTERS)  # lfw_are_same_person_func
