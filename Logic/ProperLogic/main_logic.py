"""
Program containing the main application logic.
"""

from commands import *
from database_logic import DBManager
from input_output_logic import load_clusters_from_db
from misc_helpers import clean_str, log_error, wait_for_any_input

EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

TERMINATING_TOKENS = ('halt', 'stop', 'quit', 'exit',)


# TODO: Using ground-truths in clustering - put every emb. in new cluster!

# TODO: Add type hints where needed
# TODO: What should / shouldn't be private?
# TODO: Consistent naming
# TODO: Add comments & docstrings
# TODO: Always allow option to leave current menu item / loop rather than continue!
# TODO: Consistent paths!
# TODO: How should local tables be referenced?

# TODO: Use property decorator?

# TODO: Handle db errors with rollbacks etc.!
# TODO: Give option to just start clustering completely anew (rebuilding db completely)?


def run_program(terminating_tokes, path_to_central_dir):
    path_to_local_db = os.path.join(path_to_central_dir, DBManager.local_db_file_name)
    db_manager = DBManager(path_to_local_db)
    db_manager.create_tables(create_local=False, drop_existing_tables=True)
    clusters = load_clusters_from_db(db_manager)
    initialize_commands()

    cmd_name = ''
    while cmd_name not in terminating_tokes:
        cmd_name = get_user_command()
        cmd = Command.get_command(cmd_name)
        # TODO:  Check out Software Design Patterns for better params passing to handlers?
        cmd.handler(db_manager=db_manager, clusters=clusters)


# ----- I/O -----

def get_user_command():
    command = 'add'  # _get_user_command_subfunc()
    while command not in Command.commands.keys():
        log_error('Unknown command, please try again.')
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    wait_for_any_input('What would you like to do next? (Press any key to continue).')
    print_command_options()
    return clean_str(input())


def print_command_options():
    cmd_options_lines = (f"- To {cmd_desc}, type '{cmd_name}'."
                         for cmd_name, cmd_desc in Command.get_command_descriptions())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


if __name__ == '__main__':
    run_program(TERMINATING_TOKENS, CLUSTERS_PATH)
