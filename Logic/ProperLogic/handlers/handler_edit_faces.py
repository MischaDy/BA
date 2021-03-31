from functools import partial

from Logic.ProperLogic.cluster import Cluster, Clusters
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.handlers.helpers import user_choose_cluster
from Logic.ProperLogic.misc_helpers import log_error, get_user_decision, get_user_input_of_type, wait_for_any_input


def edit_faces(clusters, **kwargs):
    # TODO: Refactor
    # TODO: Include option to delete people (and remember that in case same dir is read again? --> Probs optional)
    # TODO: Allow to abort
    # TODO: Allow deletion of faces(?)

    if not clusters:
        log_error('no clusters found, nothing to edit')
        return

    get_cluster_decision = partial(get_user_decision, 'Would you like to choose another cluster?')
    get_face_decision = partial(get_user_decision, 'Would you like to relabel another face in this cluster?')
    # TODO: Nicer parameter passing?
    get_label_scope_decision = partial(get_user_decision,
                                       'Should the whole cluster receive that label or just the picture?',
                                       choices_strs=('[c]luster', '[p]icture'), valid_choices=('c', 'p'))

    continue_cluster = ''
    while continue_cluster != 'n':
        cluster = user_choose_cluster(clusters)
        if cluster is None:  # TODO: Correct?
            continue_cluster = get_cluster_decision()
            continue
        continue_face = ''
        while continue_face != 'n':
            embedding_id = user_choose_embedding_id(cluster)
            if embedding_id is None:
                continue_face = get_face_decision()
                continue
            new_label = user_choose_face_label(cluster.label)
            if not new_label:
                continue_face = get_face_decision()
                continue

            scope = get_label_scope_decision()
            if scope == 'c':
                set_cluster_label(cluster, new_label)
            else:
                try:
                    # TODO: Undo actions that cause face to disappear --> Done
                    set_picture_label(embedding_id, new_label, cluster, clusters)
                except IncompleteDatabaseOperation:
                    pass
            continue_face = get_face_decision() if cluster.get_size() > 0 else 'n'
        continue_cluster = get_cluster_decision()


def user_choose_embedding_id(cluster):
    # TODO: Don't ask user twice if he wants to continue in that cluster!
    # TODO: Refactor
    # TODO: Give option of aborting.

    embeddings_ids_dict = cluster.get_embeddings(as_dict=True)
    faces_dict = dict(DBManager.get_thumbnails_from_cluster(cluster.cluster_id, with_embeddings_ids=True))
    label = cluster.label

    chosen_embedding_id = user_choose_embedding_id_worker(faces_dict, label)
    while chosen_embedding_id is not None and chosen_embedding_id not in embeddings_ids_dict:
        log_error(f"face id '{chosen_embedding_id}' not found. Please try again.")
        chosen_embedding_id = user_choose_embedding_id_worker(faces_dict, label)
    return chosen_embedding_id


def user_choose_embedding_id_worker(faces_dict, label):
    # TODO: Allow to abort
    # TODO: Allow specific command to label face as unknown

    get_id_decision = partial(get_user_decision, 'Would you like to view another face?')

    face_id = None
    continue_id = ''
    while continue_id != 'n':
        print_face_ids(faces_dict, label)
        face_id = get_user_input_of_type(int, 'face id')
        try:
            face = faces_dict[face_id]
        except KeyError:
            print(f'face id {face_id} could not be found. Please try again.')
            continue_id = get_id_decision()
            continue
        face.show()
        choose_cur_face_id = get_user_decision('Would you like to edit the face you just viewed?')
        if not choose_cur_face_id.startswith('n'):
            break
        face_id = None
        continue_id = get_id_decision()
    return face_id


def user_choose_face_label(old_label):
    new_label = input(f"The current label of the face is: '{old_label}'."
                      "\nWhat should the new label be? (Press Enter to abort).")
    return new_label


def set_cluster_label(cluster, new_label):
    # TODO: Use certain_labels here too? (Probably not)
    cluster.set_label(new_label)

    # TODO: Outsource as function to DBManager?
    def set_cluster_label_worker(con):
        DBManager.store_clusters([cluster], con=con, close_connection=False)

    # TODO: How to handle possible exception here?
    DBManager.connection_wrapper(set_cluster_label_worker, close_connections=True)


def set_picture_label(embedding_id, new_label, cluster, clusters):
    # TODO: Refactor! Extract parts to DBManager?
    # TODO: Don't accept label if it's the same as the old one!
    new_cluster_id = DBManager.get_max_cluster_id() + 1
    embedding = cluster.get_embedding(embedding_id)
    cluster.remove_embedding_by_id(embedding_id)
    new_cluster = Cluster(new_cluster_id, [embedding], [embedding_id], new_label)
    clusters.append(new_cluster)
    if cluster.get_size() == 0:
        clusters.remove(cluster)
        modified_clusters = Clusters([new_cluster])
    else:
        modified_clusters = Clusters([new_cluster, cluster])

    def set_pic_label_worker(con):
        if cluster.get_size() == 0:
            embeddings_row_dicts = DBManager.remove_clusters([cluster], con=con, close_connection=False)
            emb_id_to_face_dict = make_emb_id_to_face_dict_from_row_dicts(embeddings_row_dicts)
            emb_id_to_img_id_dict = make_emb_id_to_img_id_dict_from_row_dicts(embeddings_row_dicts)
        else:
            emb_id_to_face_dict = None
            emb_id_to_img_id_dict = None
        DBManager.store_clusters(modified_clusters, emb_id_to_face_dict=emb_id_to_face_dict,
                                 emb_id_to_img_id_dict=emb_id_to_img_id_dict, con=con, close_connection=False)
        DBManager.store_certain_labels(cluster=new_cluster, con=con, close_connection=False)

    con = DBManager.open_connection(open_local=False)
    try:
        DBManager.connection_wrapper(set_pic_label_worker, open_local=False, con=con)
    except IncompleteDatabaseOperation:
        cluster.add_embedding(embedding, embedding_id)
        if cluster.get_size() == 0:
            clusters.append(cluster)
        clusters.remove(new_cluster)
        raise


def make_emb_id_to_face_dict_from_row_dicts(row_dicts):
    # TODO: Rename(?)
    emb_id_to_face_dict = make_dict_from_row_dicts(row_dicts, key_col_name=Columns.embedding_id.col_name,
                                                   value_col_name=Columns.thumbnail.col_name)
    return emb_id_to_face_dict


def make_emb_id_to_img_id_dict_from_row_dicts(row_dicts):
    # TODO: Rename(?)
    emb_id_to_img_id_dict = make_dict_from_row_dicts(row_dicts, key_col_name=Columns.embedding_id.col_name,
                                                     value_col_name=Columns.image_id.col_name)
    return emb_id_to_img_id_dict


def make_dict_from_row_dicts(row_dicts, key_col_name, value_col_name):
    # TODO: Make sure to return None if row dicts are empty!
    new_dict = {
        row_dict[key_col_name]: row_dict[value_col_name]
        for row_dict in row_dicts
    }
    if new_dict:
        return new_dict


def print_face_ids(faces_dict, label):
    # TODO: print limited number of faces at a time (Enter=continue)
    # TODO: Explain to user how to abort.
    # TODO: Remove list casting
    faces_strs = list(map(lambda face_id: f'- Face {face_id}', faces_dict))
    print()
    wait_for_any_input(f"Please enter a face id to view the face, or press Enter to skip viewing. The current label of"
                       f" each of them is '{label}'."
                       "\n(Press Enter to continue.)")
    print('\n'.join(faces_strs))
