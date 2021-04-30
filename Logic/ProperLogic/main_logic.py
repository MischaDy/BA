"""
Program containing the main application logic.
"""
import datetime
import os
import time

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.handlers.handler_clear_data import clear_central_tables
from Logic.ProperLogic.handlers.handler_process_image_dir import load_imgs_from_path, face_to_embedding
from Logic.ProperLogic.handlers.handler_reclassify import reclassify
from Logic.ProperLogic.models_modules.models import Models

from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.misc_helpers import clean_string, wait_for_any_input, log_error, overwrite_dict

# -------------- TODOs --------------

# ------- NEEDED -------
# TODO: Everytime when run, determine own absolute path and use that for central database file(?)
# TODO: Convert project to exe and test!
# TODO: Replace all stored links with user choices!
# TODO: Test that rollbacks always work!
#       --> Make sure that ALL writing DB interactions of ALL handlers use con params!
# TODO: How to create CLI?
# TODO: Improve core algorithm (params + metric(?))!
# TODO: Test edge cases (e.g. calling handlers when nothing has been processed yet)!
# TODO: Make sure, no TODOs were overlooked!!
# TODO: Is the local path_id being compared to the global path id and see if the paths match?!?!
# TODO: Are images not found in a local directory being removed from the global namespace, too?!
# TODO: Handle case where directory path points to non-existent directory!!

# ------- HELPFUL -------
# TODO: Include option to delete people in edit labels handler (and remember that in case same dir is read again?
#                                                               --> Probs optional)
# TODO: Implement deletion cascade (also in clear data handler)!
#       --> Use cross-db FKs.
#       --> Google if sb has done that already!
# TODO: Implement logical conditional searching (and, or, not) of multiple people!
# TODO: Remove reset cluster ids command entirely by not allowing empty spots to form? (Not really possible if clusters
#       as a whole can be deleted?)
# TODO: Use less generic names for central and local database files to avoid collisions (and notify user if collision
#       would occur)!
# TODO: Check on start-up, which directory paths cannot be found anymore, so user can fix errors sooner!
# TODO: Prevent passing around connections by instead passing around DBManager object?!
# TODO: Give user option to choose existing local path instead of providing new one! Esp. in the context of deletion.
# TODO: Make splitting decision by computing average/median/... and max distance to center? Should be computationally
#       easier!
# TODO: Allow user to delete individual images as well as whole clusters (edit handler?!)
# TODO: When given a file path, accept both director path and path to file (e.g. db)
# TODO: Give option to not view face but edit directly
# TODO: Add reset cluster ids command
# TODO: Clean + figure out what to do with input_output module!
# TODO: Allow to exit at any time!
# TODO: Check out which data structures sqlite3 provides! (Row?) Create Row dict class if not provided?
# TODO: Use SQLAlchemy?
# TODO: Add comments & docstrings
# TODO: Always allow option to leave current menu item / loop rather than continue!
# TODO: Consistent paths!
# TODO: Give option to just start clustering completely anew (rebuilding db completely)?
# TODO: If local stuff known: return what's stored in table!(?)
# TODO: How to handle global paths etc.?
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?
# TODO: Closing images necessary?

# ------- OPTIONAL -------
# TODO: Add type hints where needed
# TODO: What should / shouldn't be private?
# TODO: Consistent naming
# TODO: Use property decorator?
# TODO: Give useful responses (and loading bar or sth like that?) after option is selected
#       and when time-consuming process is running
# TODO:  Check out Software Design Patterns for better params passing to handlers?
# TODO: Make handlers class
# TODO: Foreign keys despite separate db files? --> Implement manually? Needed?
# TODO: Consistent interface! When to pass objects (tables, columns), when to pass only their names?
# TODO: Allow instances, which have a 'current connection' as only instance attribute?
# TODO: Consistent abbreviations vs. full names (e.g. image vs. img)
# TODO: Remove unused imports at end
# TODO: Assume that every person from same photo is distinct from each other, unless user explicitly disagrees


# -------------- PROGRAM --------------

# TODO: How to properly store/use these globals and paths?

EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

# IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'


def run_program():
    init_program()
    cluster_dict = DBManager.load_cluster_dict()

    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        cmd = Command.get_command(cmd_name)
        call_handler(cmd.handler, cluster_dict=cluster_dict)
        cmd_name = get_user_command()


def run_program_with_user_stats():
    write = False
    command_stats_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech\commands_stats.txt'
    t0 = time.time()

    # Models.altered_mtcnn.keep_all = False
    init_program()
    cluster_dict = DBManager.load_cluster_dict()

    commands = []
    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        t1 = time.time()
        cmd = Command.get_command(cmd_name)
        call_handler(cmd.handler, cluster_dict=cluster_dict)
        t2 = time.time()
        commands.append([cmd_name, t2 - t1])
        cmd_name = get_user_command()

    tn = time.time()
    commands_str = '\n'.join(map(str, commands)) + '\n\n' + f'total runtime: {tn - t0}'
    if write:
        with open(command_stats_path, 'w') as file:
            file.write(commands_str)


def measure_commands():
    # TODO: process faces should be limited by n!
    write = False
    start, stop, step = 90, 450, 90
    COMMAND_STATS_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech\commands_stats.txt'
    DATASET_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech'

    def process_images_dir_measure(cluster_dict, n):
        images_path = DATASET_PATH
        try:
            print('------ PROCESSING FACES')
            process_faces_measure(images_path, n)
            print('------ DONE PROCESSING')
        except IncompleteDatabaseOperation as e:
            print('process_images_dir_measure error')
            log_error(e)
            return

        cluster_dict_copy = cluster_dict.copy()

        def cluster_processed_faces(con):
            embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))

            # TODO: Call reclassify handler here?
            # TODO: Clear existing clusters? Issues with ids etc.????
            core_algorithm = CoreAlgorithm()
            # passing result cluster dict already overwrites it
            clustering_result = core_algorithm.cluster_embeddings(embeddings_with_ids,
                                                                  existing_clusters_dict=cluster_dict,
                                                                  should_reset_cluster_ids=True,
                                                                  final_clusters_only=False)
            _, modified_clusters_dict, removed_clusters_dict = clustering_result
            DBManager.overwrite_clusters_simplified(modified_clusters_dict, removed_clusters_dict, con=con,
                                                    close_connections=False)
        try:
            DBManager.connection_wrapper(cluster_processed_faces)
        except IncompleteDatabaseOperation:
            overwrite_dict(cluster_dict, cluster_dict_copy)

    def process_faces_measure(images_path, n, central_con=None, local_con=None, close_connections=True):
        if local_con is None:
            path_to_local_db = DBManager.get_local_db_file_path(images_path)
        else:
            path_to_local_db = None

        def process_faces_worker(central_con, local_con):
            DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                          close_connections=False)
            extract_faces_measure(images_path, n, central_con=central_con, local_con=local_con, close_connections=False)

        DBManager.connection_wrapper(process_faces_worker, path_to_local_db=path_to_local_db,
                                     central_con=central_con, local_con=local_con, with_central=True, with_local=True,
                                     close_connections=close_connections)

    def extract_faces_measure(path, n, check_if_known=True, central_con=None, local_con=None, close_connections=True):
        path_to_local_db = DBManager.get_local_db_file_path(path)
        path_id = DBManager.get_path_id(path)
        if path_id is None:
            # path not yet known
            path_id = DBManager.store_directory_path(path, con=central_con, close_connections=False)
            DBManager.store_path_id(path_id, path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
        imgs_names_and_date = set(DBManager.get_images_attributes(path_to_local_db=path_to_local_db))

        # Note: 'MAX' returns None / (None, ) as a default value
        max_img_id = DBManager.get_max_image_id(path_to_local_db=path_to_local_db)
        start_img_id = max_img_id + 1
        initial_max_embedding_id = DBManager.get_max_embedding_id()

        def get_counted_img_loader():
            img_loader = load_imgs_from_path(path, recursive=True, output_file_names=True, output_file_paths=True)
            nums = range(start_img_id, start_img_id + n)
            return zip(nums, img_loader)

        def store_embedding_row_dicts(con):
            max_embedding_id = initial_max_embedding_id
            for img_id, (img_path, img_name, img) in get_counted_img_loader():
                # Check if image already stored --> don't process again
                # known = (name, last modified) as a pair known for this director
                last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
                if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                    continue

                DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                      path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
                DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

                faces = Models.altered_mtcnn.forward_return_results(img)
                if not faces:
                    log_error(f"no faces found in image '{img_path}'")
                    continue

                embeddings_row_dicts = [{Columns.cluster_id.col_name: 'NULL',
                                         Columns.embedding.col_name: face_to_embedding(face),
                                         Columns.thumbnail.col_name: face,
                                         Columns.image_id.col_name: img_id,
                                         Columns.embedding_id.col_name: embedding_id}
                                        for embedding_id, face in enumerate(faces, start=max_embedding_id + 1)]
                DBManager.store_embeddings(embeddings_row_dicts, con=con, close_connections=False)
                max_embedding_id += len(faces)

        DBManager.connection_wrapper(store_embedding_row_dicts, con=central_con, close_connections=close_connections)

    def clear_data_measure(cluster_dict):
        local_db_dir_path = DATASET_PATH
        path_to_local_db = DBManager.get_local_db_file_path(local_db_dir_path)

        def clear_data_worker(central_con, local_con):
            DBManager.clear_local_tables(path_to_local_db, con=local_con, close_connections=False)
            clear_central_tables(con=central_con, close_connections=False)
            overwrite_dict(cluster_dict, dict())

        try:
            DBManager.connection_wrapper(clear_data_worker, path_to_local_db=path_to_local_db, with_central=True,
                                         with_local=True)
        except IncompleteDatabaseOperation as e:
            print('clear_data_measure error')
            log_error(e)

    cmds_list = [
        ('process_images_dir', process_images_dir_measure),
        ('reclassify', reclassify),
        ('clear_data', clear_data_measure),
    ]

    clear_data_measure(dict())
    commands = []
    for n in range(start, stop + 1, step):
        print(f'ITERATION: {n}')
        init_program()
        cluster_dict = DBManager.load_cluster_dict()
        for cmd_name, cmd in cmds_list:
            print(f'--- COMMAND: {cmd_name}')
            t1 = time.time()
            args = [cluster_dict] if cmd_name != 'process_images_dir' else [cluster_dict, n]
            cmd(*args)
            t2 = time.time()
            commands.append([cmd_name, n, t2 - t1])

    commands_str = '\n'.join(map(str, commands))
    if write:
        with open(COMMAND_STATS_PATH, 'w') as file:
            file.write(commands_str)


def init_program():
    DBManager.create_central_tables(drop_existing_tables=False)
    Commands.initialize()


def call_handler(handler, *args, **kwargs):
    try:
        return handler(*args, **kwargs)
    except Exception as e:
        log_error(e)


def _show_thumbnail():
    thumbs = DBManager.fetch_from_table(Tables.embeddings_table, cols=[Columns.thumbnail])
    thumbs[0].show()


# ----- I/O -----


def get_user_command():
    cmd_shorthand = get_user_command_shorthand()
    while cmd_shorthand not in Command.get_command_shorthands():
        print(f"Unknown command '{cmd_shorthand}', please try again.")
        cmd_shorthand = get_user_command_shorthand()
    cmd = Command.get_cmd_name_by_shorthand(cmd_shorthand)
    return cmd


def get_user_command_shorthand():
    wait_for_any_input('\nWhat would you like to do next? (Press Enter to continue).')
    print_command_options()
    return clean_string(input(), to_lower=True)


def print_command_options():
    cmd_options_lines = map(lambda cmd: f"- {cmd.make_cli_cmd_string()}", Command.get_commands())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


if __name__ == '__main__':
    run_program()
