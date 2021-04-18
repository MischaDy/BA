from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from eval_handlers_versions.eval_process_image_dir import eval_process_image_dir

import f_measure


IMAGES_PATH = ''
SAVE_RESULTS = True
SAVE_PATH = 'results'


def main(images_path):
    cluster_dict = ClusterDict()
    emb_id_to_name = eval_process_image_dir(cluster_dict, images_path)
    clusters = cluster_dict.get_clusters()
    f_measure.main(clusters, emb_id_to_name, SAVE_RESULTS, SAVE_PATH)



if __name__ == '__main__':
    main(IMAGES_PATH)
