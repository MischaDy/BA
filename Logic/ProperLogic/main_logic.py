"""
Program containing the main application logic.
"""

from functools import partial

from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns
from Logic.ProperLogic.database_modules.database_logic import DBManager
from misc_helpers import clean_string, wait_for_any_input, get_user_decision


# -------------- TODOs --------------

# ------- NEEDED -------

# TODO: Convert project to exe and test!
# TODO: Replace all stored links with user choices!
# TODO: Test that rollbacks always work!
#       --> Make sure that ALL writing DB interactions of ALL handlers use con params!
# TODO: How to create CLI?
# TODO: Improve core algorithm (params + metric(?))!
# TODO: Test edge cases (e.g. calling handlers when nothing has been processed yet)!
# TODO: Add function to reset cluster ids to smallest possible! (sequential too?)
# TODO: Test quality with private pictures + bigger mixed dataset!
# TODO: How to handle empty user inputs?

# ------- HELPFUL -------
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
# TODO: Make handlers class in commands file
# TODO: Split commands file
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

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'


def run_program(path_to_central_dir):
    # TODO: Make this failsafe!!!
    DBManager.create_central_tables(drop_existing_tables=False)
    clusters = DBManager.load_clusters()
    # TODO: More elegant way around this?
    Commands.initialize()

    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        cmd = Command.get_command(cmd_name)
        cmd.handler(clusters=clusters)
        cmd_name = get_user_command()


def _show_thumbnail():
    thumbs = DBManager.fetch_from_table(Tables.embeddings_table, col_names=[Columns.thumbnail])
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
    run_program(CLUSTERS_PATH)
