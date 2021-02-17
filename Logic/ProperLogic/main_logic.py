"""
Program containing the main application logic.
"""

from Logic.ProperLogic.commands import *
from Logic.ProperLogic.database_logic import CENTRAL_DB_FILE, DBManager
from Logic.ProperLogic.input_output_logic import load_clusters_from_db
from Logic.misc_helpers import clean_str, log_error, wait_for_any_input

TENSORS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

TERMINATING_TOKENS = ('halt', 'stop', 'quit', 'exit',)


# TODO: What should / shouldn't be private?
# TODO: Turn Commands into an Enum?
# TODO: Consistent naming
# TODO: Add comments & docstring
# TODO: Always allow option to leave current menu item / loop rather than continue!
# TODO: consistent paths!
# TODO: consistent parameter names


def main(terminating_tokes, path_to_central_dir):
    initialize_commands(path_to_central_dir)
    path_to_central_db = os.path.join(path_to_central_dir, CENTRAL_DB_FILE)
    db_manager = DBManager(path_to_central_db)
    clusters = load_clusters_from_db(db_manager)

    cmd_name = ''
    while cmd_name not in terminating_tokes:
        cmd_name = get_user_command()
        cmd = Command.get_command(cmd_name)
        process_command(cmd)


# ----- I/O -----

def get_user_command():
    # TODO: make user choose command
    command = 'add'  # _get_user_command_subfunc()
    while command not in Command.commands.keys():
        log_error('Unknown command, please try again.')
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    wait_for_any_input('What would you like to do next?')
    print_command_options()
    return clean_str(input())


def print_command_options():
    cmd_options_lines = (f"- To {cmd_desc}, type '{cmd_name}'."
                         for cmd_name, cmd_desc in Command.get_command_descriptions())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


if __name__ == '__main__':
    main(TERMINATING_TOKENS, CLUSTERS_PATH)
