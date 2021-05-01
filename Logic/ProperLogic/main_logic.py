"""
Program containing the main application logic.
"""
from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import clean_string, wait_for_any_input, log_error

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


def run_program():
    init_program()
    cluster_dict = DBManager.load_cluster_dict()

    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        cmd = Command.get_command(cmd_name)
        call_handler(cmd.handler, cluster_dict=cluster_dict)
        cmd_name = get_user_command()


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
