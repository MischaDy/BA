from functools import partial

from Logic.ProperLogic.cluster import Cluster, Clusters
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.handlers.helpers import user_choose_cluster
from Logic.ProperLogic.misc_helpers import log_error, get_user_decision, get_user_input_of_type, wait_for_any_input


# TODO: Square brackets around cluster numbers!
# TODO: Square brackets around face numbers!

def edit_labels(clusters, **kwargs):
    # TODO: Refactor
    # TODO: Include option to delete people (and remember that in case same dir is read again? --> Probs optional)

    if not clusters:
        log_error('no clusters found, no labels to edit')
        return

    get_cluster_decision = partial(get_user_decision, 'Would you like to choose another cluster?')
    get_face_decision = partial(get_user_decision, 'Would you like to relabel another face in this cluster?')
    # TODO: Nicer parameter passing?
    get_label_scope_decision = partial(get_user_decision,
                                       'Should the whole cluster receive that label or just the picture?',
                                       choices_strs=('[c]luster', '[p]icture'), valid_choices=('c', 'p'))

    continue_choosing_cluster = ''
    while continue_choosing_cluster != 'n':
        cluster = user_choose_cluster(clusters)
        if cluster is None:
            continue_choosing_cluster = get_cluster_decision()
            continue
        continue_choosing_face = ''
        while continue_choosing_face != 'n':
            embedding_id = user_choose_embedding_id(cluster)
            if embedding_id is None:
                # User *doesn't* want to relabel another face in this cluster!
                break
            new_label = user_choose_face_label(cluster.label)
            if new_label is None:
                continue_choosing_face = get_face_decision()
                continue

            label_scope = get_label_scope_decision()
            if label_scope == 'c':
                set_cluster_label(cluster, new_label)
            else:
                try:
                    set_picture_label(embedding_id, new_label, cluster, clusters)
                except IncompleteDatabaseOperation:
                    pass

            # Auto-stop choosing faces if cluster is empty or consists of only one face
            continue_choosing_face = get_face_decision() if cluster.get_size() > 2 else 'n'
        continue_choosing_cluster = get_cluster_decision()


def user_choose_embedding_id(cluster):
    # TODO: Refactor
    faces_dict = dict(DBManager.get_thumbnails_from_cluster(cluster.cluster_id, with_embeddings_ids=True))
    chosen_embedding_id = user_choose_embedding_id_worker(faces_dict, cluster.label)
    return chosen_embedding_id


def user_choose_embedding_id_worker(faces_dict, label):
    # TODO: Allow specific command to label face as unknown
    # TODO: Choose cluster add newline!
    #       "Please enter the cluster id (must be convertible to a(n) int or have the value <empty string>)."
    # TODO: Enter new label add newline!
    # TODO: "Would you like to choose another cluster" appears too late / after prompt?!

    get_id_decision = partial(get_user_decision, 'Would you like to relabel another face in this cluster?')

    face_id = None
    continue_choosing_id = ''
    while continue_choosing_id != 'n':
        print_face_ids(faces_dict, label)
        face_id = get_user_input_of_type(prompt='', obj_name='face id', class_=int, allow_empty=True,
                                         empty_as_none=True)
        if face_id is None:
            continue_choosing_id = get_id_decision()
            continue
        try:
            face = faces_dict[face_id]
        except KeyError:
            print(f'face id {face_id} could not be found. Please try again.')
            continue_choosing_id = get_id_decision()
            continue
        face.show()
        choose_cur_face_id = get_user_decision('Would you like to relabel the face you just viewed?')
        if choose_cur_face_id == 'y':
            break
        face_id = None
        continue_choosing_id = get_id_decision()
    return face_id


def user_choose_face_label(old_label):
    prompt = ("\n"
              f"The current label of the face is: '{old_label}'."
              "\nPlease enter a new label, or press Enter to cancel.")
    new_label = get_user_input_of_type(prompt, obj_name='label', allow_empty=True)
    return new_label


def set_cluster_label(cluster, new_label):
    # TODO: Use certain_labels here too? (Probably not)
    cluster.set_label(new_label)

    # TODO: Outsource as function to DBManager?
    def set_cluster_label_worker(con):
        DBManager.store_clusters([cluster], con=con, close_connections=False)

    # TODO: How to handle possible exception here?
    DBManager.connection_wrapper(set_cluster_label_worker)


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
            embeddings_row_dicts = DBManager.remove_clusters([cluster], con=con, close_connections=False)
            emb_id_to_face_dict = make_emb_id_to_face_dict_from_row_dicts(embeddings_row_dicts)
            emb_id_to_img_id_dict = make_emb_id_to_img_id_dict_from_row_dicts(embeddings_row_dicts)
        else:
            emb_id_to_face_dict = None
            emb_id_to_img_id_dict = None
        DBManager.store_clusters(modified_clusters, emb_id_to_face_dict=emb_id_to_face_dict,
                                 emb_id_to_img_id_dict=emb_id_to_img_id_dict, con=con, close_connections=False)
        DBManager.store_certain_labels(cluster=new_cluster, con=con, close_connections=False)

    con = DBManager.open_central_connection()
    try:
        DBManager.connection_wrapper(set_pic_label_worker, con=con)
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
    # TODO: Remove list casting
    print()
    wait_for_any_input(f"Please enter a face id to view the face, or press Enter to cancel viewing. The current label"
                       f" of each face is '{label}'."
                       "\n(Press Enter to continue.)")
    faces_strs = map(lambda face_id: f'- Face {face_id}', faces_dict)
    print('\n'.join(faces_strs))
