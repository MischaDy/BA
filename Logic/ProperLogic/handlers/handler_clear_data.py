import os
from functools import partial

from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.misc_helpers import get_user_decision, overwrite_dict, log_error


# TODO: Add handler which not only allows to clear, but to completely drop the database files.
#       Or actually convert the clear data handler to that?

def clear_data(cluster_dict, **kwargs):
    # TODO: Include deletion cascade!
    data_kinds = {'l': '[l]ocal tables',
                  'g': '[g]lobal tables',
                  'b': '[b]oth local and global tables',
                  'c': '[c]lusters',
                  'n': '[n]either'}
    warning = "----- WARNING: DESTRUCTIVE ACTION -----\n"

    should_clear_data_func = partial(get_user_decision,
                                     warning
                                     + "Would you like to clear the local/global data?"
                                       " Don't worry, you will have to re-confirm a 'yes'.")
    data_kind_to_clear_func = partial(get_user_decision,
                                      choices_strs=tuple(data_kinds.values()),
                                      valid_choices=tuple(data_kinds.keys()))

    should_clear_data = should_clear_data_func()
    while should_clear_data == 'y':
        data_kind_to_clear = data_kind_to_clear_func(
            prompt=(warning
                    + "Which kind(s) of data would you like to clear?"
                      " Don't worry, you will have to re-confirm your choice.")
        )
        if data_kind_to_clear == 'n':
            should_clear_data = should_clear_data_func()
            continue

        chosen_data_to_clear_str = data_kinds[data_kind_to_clear].replace('[', '').replace(']', '')
        confirm_data_to_clear = data_kind_to_clear_func(
            prompt=(warning
                    + f"Are you sure that you want to clear {chosen_data_to_clear_str}?"
                      f" This action cannot be undone. To confirm your choice, simply re-enter it.")
        )

        if confirm_data_to_clear != data_kind_to_clear:
            should_clear_data = should_clear_data_func()
            continue

        def clear_data_worker(con):
            if data_kind_to_clear in ('l', 'b'):
                # TODO: How to use local connections here? Rollback on multiple?
                clear_local_tables()
            if data_kind_to_clear in ('g', 'b'):
                clear_central_tables(con=con, close_connections=False)
                overwrite_dict(cluster_dict, dict())
            if data_kind_to_clear == 'c':
                clear_clustering(con=con, close_connections=False)
                overwrite_dict(cluster_dict, dict())

        try:
            DBManager.connection_wrapper(clear_data_worker)
        except IncompleteDatabaseOperation:
            continue

        should_clear_data = 'n'


def clear_local_tables(con=None, close_connections=True):
    local_db_dir_path = user_choose_local_db_dir_path()
    if local_db_dir_path is None:
        return

    path_to_local_db = DBManager.get_local_db_file_path(local_db_dir_path)
    DBManager.clear_local_tables(path_to_local_db, con=con, close_connections=close_connections)


def clear_central_tables(con=None, close_connections=True):
    DBManager.clear_central_tables(con=con, close_connections=close_connections)


def clear_clustering(con=None, close_connections=True):
    DBManager.clear_clusters(con=con, close_connections=close_connections)


def user_choose_local_db_dir_path():
    # TODO: Refactor, use user_choose function!
    local_db_dir_path = input('Please enter a path containing a local table you would like to clear.\n')
    # local_db_dir_path = (r'C:\Users\Mischa\Desktop\Uni\20-21 WS'
    #                      r'\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs')
    while True:
        if not local_db_dir_path:
            local_db_dir_path = None
            break
        elif not os.path.exists(local_db_dir_path):
            log_error(f"unable to find path '{local_db_dir_path}'")
        elif not DBManager.is_local_db_in_dir(local_db_dir_path):
            log_error(f"unable to find local database file '{...}' in path '{local_db_dir_path}'")
        else:
            break
        print("\nPlease try again.")
        local_db_dir_path = input('Please enter a path with images of people you would like to add.\n')
    return local_db_dir_path
