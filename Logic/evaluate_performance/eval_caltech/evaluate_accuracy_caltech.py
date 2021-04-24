import os
from functools import partial
from itertools import repeat

from Logic.ProperLogic.misc_helpers import add_multiple_to_dict
from Logic.evaluate_performance.evaluate_accuracy import run_metric_evaluation

IMAGES_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech'
SAVE_PATH = 'results_caltech'
DELETE_LOCAL_DB_FILE = False
DELETE_CENTRAL_DB_FILE = False
CLEAR_CLUSTERS = True

THRESHOLDS = [0.73]  # np.linspace(0.5, 1.0, num=11)  # ==>  step size = 0.05
METRICS = [2]  # , 1.5, 1, 0.75, 0.5, 0.3, 0.2, 0.1, 0]


def main(images_path):
    partial_caltech_are_same_person_func = partial(caltech_are_same_person_func,
                                                   img_name_to_id_dict=get_img_name_to_id_dict(images_path))
    run_metric_evaluation(images_path, partial_caltech_are_same_person_func, save_path=SAVE_PATH,
                          delete_central_db_file=DELETE_CENTRAL_DB_FILE, delete_local_db_file=DELETE_LOCAL_DB_FILE)
    # images_path, are_same_person_func, save_path, thresholds=(0.73,), metrics=(2,),
    #                           delete_central_db_file=False, delete_local_db_file=False, clear_clusters=True


def get_img_name_to_id_dict(images_path):
    def is_dir(file_obj):
        return os.path.isdir(os.path.join(images_path, file_obj))

    image_dirs = filter(is_dir, os.listdir(images_path))
    img_name_to_id_dict = {}
    for image_dir in image_dirs:
        image_dir_path = os.path.join(images_path, image_dir)
        images = os.listdir(image_dir_path)
        img_names_and_id = zip(images, repeat(image_dir))
        add_multiple_to_dict(img_name_to_id_dict, img_names_and_id)
    return img_name_to_id_dict


def caltech_are_same_person_func(emb_id1, emb_id2, emb_id_to_img_name_dict, img_name_to_person_id_dict):
    img_name1, img_name2 = map(emb_id_to_img_name_dict.get, [emb_id1, emb_id2])
    return img_name_to_person_id_dict[img_name1] == img_name_to_person_id_dict[img_name2]


if __name__ == '__main__':
    main(IMAGES_PATH)
