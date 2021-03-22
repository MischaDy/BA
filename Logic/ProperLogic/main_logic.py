"""
Program containing the main application logic.
"""

import os

from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_table_defs import Tables, Columns
from database_logic import DBManager
from input_output_logic import load_clusters_from_db
from misc_helpers import clean_str, log_error, wait_for_any_input, get_every_nth_item


# TODO: How to properly store/use these globals and paths?
EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'


# TODO: Using ground-truths in clustering - put every emb. in new cluster!

# TODO: Add type hints where needed
# TODO: What should / shouldn't be private?
# TODO: Consistent naming
# TODO: Add comments & docstrings
# TODO: Always allow option to leave current menu item / loop rather than continue!
# TODO: Consistent paths!

# TODO: Use property decorator?

# TODO: Handle db errors with rollbacks etc.!
# TODO: Give option to just start clustering completely anew (rebuilding db completely)?

# TODO:  Check out Software Design Patterns for better params passing to handlers?


# TODO: Remove
DROP_CENTRAL_TABLES = False
DROP_LOCAL_TABLES = False


def run_program(path_to_central_dir):
    path_to_local_db = os.path.join(path_to_central_dir, DBManager.local_db_file_name)
    db_manager = DBManager(path_to_local_db)

    # TODO: Remove
    if DROP_CENTRAL_TABLES:
        db_manager.delete_from_table(Tables.embeddings_table)
        db_manager.delete_from_table(Tables.cluster_attributes_table)
    if DROP_LOCAL_TABLES:
        my_images_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs'
        my_path_to_local_db = db_manager.get_db_path(my_images_path, local=True)
        db_manager.delete_from_table(Tables.images_table, path_to_local_db=my_path_to_local_db)

    db_manager.create_tables(create_local=False, drop_existing_tables=False)
    clusters = load_clusters_from_db(db_manager)
    Commands.initialize()

    cmd_name = ''
    while cmd_name not in Command.terminating_tokens:
        # TODO: What in this loop is printing some number? (only when calling add handler?)
        cmd_name = get_user_command()
        cmd = Command.get_command(cmd_name)
        cmd.handler(db_manager=db_manager, clusters=clusters)


def demo_program(path_to_central_dir):
    path_to_local_db = os.path.join(path_to_central_dir, DBManager.local_db_file_name)
    db_manager = DBManager(path_to_local_db)
    db_manager.create_tables(create_local=False, drop_existing_tables=True)
    clusters = load_clusters_from_db(db_manager)
    Commands.initialize()

    cmd_name = get_user_command()
    while cmd_name not in Command.terminating_tokens:
        cmd = Command.get_command(cmd_name)
        cmd.handler(db_manager=db_manager, clusters=clusters)
        cmd_name = get_user_command()
        cmd_name = 'exit'

    thumbs = db_manager.fetch_from_table(Tables.embeddings_table, col_names=[Columns.thumbnail])
    thumbs[0].show()


# ----- I/O -----

def get_user_command():
    # TODO: Let user choose command
    command = _get_user_command_subfunc()  # 'edit faces'
    while command not in Command.commands.keys() and command not in Command.terminating_tokens:
        log_error(f'Unknown command {command}, please try again.')
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    wait_for_any_input('\nWhat would you like to do next? (Press any key to continue).')
    print_command_options()
    return clean_str(input())


def print_command_options():
    cmd_options_lines = (f"- To {cmd_desc}, type '{cmd_name}'."
                         for cmd_name, cmd_desc in Command.get_command_descriptions())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)
    print(f"- To exit, type e.g. 'exit'.")


if __name__ == '__main__':
    run_program(CLUSTERS_PATH)
