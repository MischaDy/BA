"""
Program containing the main application logic.
"""

from functools import partial

from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns
from Logic.ProperLogic.database_modules.database_logic import DBManager
from input_output_logic import load_clusters_from_db
from misc_helpers import clean_str, wait_for_any_input, get_user_decision


# -------------- TODOs --------------

# ------- NEEDED -------
# TODO: Using ground-truths in clustering - put every emb. in new cluster!
# TODO: Test that rollbacks always work!
# TODO: How to create CLI?
# TODO: Select good params for core algorithm
# TODO: Fix relative/absolute path problems

# ------- HELPFUL -------
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

# ------- OPTIONAL -------
# TODO: Add type hints where needed
# TODO: What should / shouldn't be private?
# TODO: Consistent naming
# TODO: Use property decorator?
# TODO: Give useful responses (and loading bar or sth like that?) after option is selected
#       and when time-consuming process is running
# TODO:  Check out Software Design Patterns for better params passing to handlers?
# TODO: Make handlers class in commands file
# TODO: Split commands file
# TODO: Foreign keys despite separate db files? --> Implement manually? Needed?
# TODO: Consistent interface! When to pass objects (tables, columns), when to pass only their names?
# TODO: Allow instances, which have a 'current connection' as only instance attribute?
# TODO: Consistent abbreviations vs. full names (e.g. image vs. img)


# -------------- PROGRAM --------------

# TODO: How to properly store/use these globals and paths?
EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

# TODO: Remove
ASK_FOR_DELETION = True


def run_program(path_to_central_dir):
    # path_to_local_db = os.path.join(path_to_central_dir, DBManager.local_db_file_name)

    if ASK_FOR_DELETION:
        # TODO: Remove
        print(f'Number of clusters: {len(load_clusters_from_db())}')
        prompt_user_clear_tables()

    DBManager.create_all_tables(create_local=False, drop_existing_tables=False)
    clusters = load_clusters_from_db()
    Commands.initialize()

    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        cmd = Command.get_command(cmd_name)
        cmd.handler(clusters=clusters)
        cmd_name = get_user_command()


def demo_program(path_to_central_dir):
    # path_to_local_db = os.path.join(path_to_central_dir, DBManager.local_db_file_name)
    DBManager.create_all_tables(create_local=False, drop_existing_tables=True)
    clusters = load_clusters_from_db()
    Commands.initialize()

    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        cmd = Command.get_command(cmd_name)
        cmd.handler(clusters=clusters)
        cmd_name = 'exit'

    thumbs = DBManager.fetch_from_table(Tables.embeddings_table, col_names=[Columns.thumbnail])
    thumbs[0].show()


# ----- I/O -----


def prompt_user_clear_tables():
    tables_kinds = {'l': '[l]ocal tables',
                    'g': '[g]lobal tables',
                    'b': '[b]oth kinds of tables',
                    'n': '[n]either'}
    warning = "----- WARNING: DESTRUCTIVE ACTION -----\n"

    should_drop_tables_func = partial(get_user_decision,
                                      warning
                                      + "Do you want to clear the local/global tables?"
                                        " Don't worry, you will have to re-confirm a 'yes'.")
    table_kind_to_drop_func = partial(get_user_decision,
                                      choices_strs=tuple(tables_kinds.values()),
                                      valid_choices=tuple(tables_kinds.keys()))

    should_drop_tables = should_drop_tables_func()
    while should_drop_tables == 'y':
        table_kind_to_drop = table_kind_to_drop_func(
            prompt=warning
                   + "Which kinds of tables would you like to clear?"
                     " Don't worry, you will have to re-confirm your choice."
        )
        if table_kind_to_drop == 'n':
            should_drop_tables = should_drop_tables_func()
            continue

        chosen_table_to_drop_str = tables_kinds[table_kind_to_drop].replace('[', '').replace(']', '')
        confirm_tables_to_drop = table_kind_to_drop_func(
            prompt=warning
                   + f"Are you sure that you want to clear {chosen_table_to_drop_str}?"
                     f" This action cannot be undone. To confirm your choice, simply re-enter it."
        )

        if confirm_tables_to_drop != table_kind_to_drop:
            should_drop_tables = should_drop_tables_func()
            continue

        if table_kind_to_drop in ('l', 'b'):
            __clear_local_tables()
        if table_kind_to_drop in ('g', 'b'):
            __clear_central_tables()
        should_drop_tables = 'n'


def __clear_local_tables():
    my_images_path = (r'C:\Users\Mischa\Desktop\Uni\20-21 WS'
                      r'\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs')
    my_path_to_local_db = DBManager.get_db_path(my_images_path, local=True)
    for table in Tables.local_tables:
        DBManager.delete_from_table(table, path_to_local_db=my_path_to_local_db)


def __clear_central_tables():
    for table in Tables.central_tables:
        DBManager.delete_from_table(table)


def get_user_command():
    cmd_shorthand = get_user_command_shorthand()
    while cmd_shorthand not in Command.get_command_shorthands():
        print(f'unknown command {cmd_shorthand}, please try again.')
        cmd_shorthand = get_user_command_shorthand()
    cmd = Command.get_cmd_name_by_shorthand(cmd_shorthand)
    return cmd


def get_user_command_shorthand():
    wait_for_any_input('\nWhat would you like to do next? (Press any key to continue).')
    print_command_options()
    return clean_str(input())


def print_command_options():
    cmd_options_lines = map(lambda cmd: f"- {cmd.make_cli_cmd_string()}", Command.get_commands())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


if __name__ == '__main__':
    run_program(CLUSTERS_PATH)
