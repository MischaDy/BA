from Logic.ProperLogic.handlers.helpers import user_choose_cluster
from Logic.ProperLogic.misc_helpers import clean_str, wait_for_any_input


def show_cluster(clusters_path, **kwargs):
    # TODO: Needed? Or can be included in / adapted from edit_faces?
    # TODO: Finish implementing
    should_continue = ''
    while not should_continue.startswith('n'):
        cluster_name, cluster_path = user_choose_cluster(clusters_path)
        _output_cluster_content(cluster_name, cluster_path)
        ...
        should_continue = clean_str(input('Choose another cluster?\n'))


def _output_cluster_content(cluster_name, cluster_path):
    wait_for_any_input(f'Which face image in the cluster "{cluster_name}" would you like to view?')
    # TODO: finish
    # TODO: output faces and (-> separate function?) allow choice of image
