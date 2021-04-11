from functools import partial

from Logic.ProperLogic.misc_helpers import get_user_input_of_type, log_error, wait_for_any_input


# ----- I/O HELPERS -----

def user_choose_cluster(cluster_dict):
    # TODO: Refactor
    cluster_ids = cluster_dict.get_cluster_ids()
    print_cluster_ids(cluster_dict)

    get_user_cluster_id = partial(get_user_input_of_type, class_=int, obj_name='cluster id', allow_empty=True)
    chosen_cluster_id = get_user_cluster_id()
    while chosen_cluster_id is not None and chosen_cluster_id not in cluster_ids:
        log_error(f'cluster "{chosen_cluster_id}" not found; Please try again.')
        print_cluster_ids(cluster_dict)
        chosen_cluster_id = get_user_cluster_id()

    if chosen_cluster_id is None:
        return

    chosen_cluster = cluster_dict.get_cluster_by_id(chosen_cluster_id)
    return chosen_cluster


def print_cluster_ids(cluster_dict):
    # TODO: print limited number of clusters at a time (Enter=continue)
    cluster_labels_with_ids = cluster_dict.get_cluster_labels(with_ids=True)
    clusters_strs = (f"- Cluster {cluster_id} ('{label}')"
                     for cluster_id, label in cluster_labels_with_ids)
    wait_for_any_input('\nPlease enter the id of the cluster you would like to view, or press Enter to cancel.'
                       '\n(Press Enter to continue.)')
    print('\n'.join(clusters_strs))
