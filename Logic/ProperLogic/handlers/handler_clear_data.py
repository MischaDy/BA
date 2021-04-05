from functools import partial

from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.database_modules.database_table_defs import Tables
from Logic.ProperLogic.misc_helpers import get_user_decision


def clear_data(**kwargs):
    tables_kinds = {'l': '[l]ocal tables',
                    'g': '[g]lobal tables',
                    'b': '[b]oth kinds of tables',
                    'n': '[n]either'}
    warning = "----- WARNING: DESTRUCTIVE ACTION -----\n"

    should_drop_tables_func = partial(get_user_decision,
                                      warning
                                      + "Would you like to clear the local/global tables?"
                                        " Don't worry, you will have to re-confirm a 'yes'.")
    table_kind_to_drop_func = partial(get_user_decision,
                                      choices_strs=tuple(tables_kinds.values()),
                                      valid_choices=tuple(tables_kinds.keys()))

    should_drop_tables = should_drop_tables_func()
    while should_drop_tables == 'y':
        table_kind_to_drop = table_kind_to_drop_func(
            prompt=(warning
                    + "Which kinds of tables would you like to clear?"
                      " Don't worry, you will have to re-confirm your choice.")
        )
        if table_kind_to_drop == 'n':
            should_drop_tables = should_drop_tables_func()
            continue

        chosen_table_to_drop_str = tables_kinds[table_kind_to_drop].replace('[', '').replace(']', '')
        confirm_tables_to_drop = table_kind_to_drop_func(
            prompt=(warning
                    + f"Are you sure that you want to clear {chosen_table_to_drop_str}?"
                      f" This action cannot be undone. To confirm your choice, simply re-enter it.")
        )

        if confirm_tables_to_drop != table_kind_to_drop:
            should_drop_tables = should_drop_tables_func()
            continue

        if table_kind_to_drop in ('l', 'b'):
            clear_local_tables()
        if table_kind_to_drop in ('g', 'b'):
            clear_central_tables()
        should_drop_tables = 'n'


def clear_local_tables():
    # TODO: Make user choose paths (multiple!) to clear!
    images_path = (r'C:\Users\Mischa\Desktop\Uni\20-21 WS'
                   r'\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs')
    path_to_local_db = DBManager.get_db_path(images_path, local=True)
    for table in Tables.local_tables:
        DBManager.delete_from_table(table, path_to_local_db=path_to_local_db)


def clear_central_tables():
    for table in Tables.central_tables:
        DBManager.delete_from_table(table)
