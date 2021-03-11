# Credits: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import datetime
import os
import sqlite3
import time
from functools import partial

import torch
from PIL import Image
import io

from Logic.ProperLogic.database_table_defs import Tables, Columns, ColumnTypes, ColumnDetails, _get_true_attr

# TODO: *Global* face_id - created in central table, then written to corresponding local table
# TODO: Foreign keys despite separate db files? --> Implement manually? Needed?
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?
# TODO: Use SQLAlchemy?

# TODO: FK faces -> embeddings other way around? Or remove completely?
# TODO: Consistent interface! When to pass objects (tables, columns), when to pass only their names??

# TODO: When to close connection? Optimize?
# TODO: Guarantee that connection is closed at end of methods (done?)

# TODO: Make DBManager a singleton object?


"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT face_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---

embeddings(INT cluster_id, INT face_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


class DBManager:
    db_files_path = 'database'
    central_db_file_name = 'central_db.sqlite'
    central_db_file_path = os.path.join(db_files_path, central_db_file_name)
    local_db_file_name = 'local_db.sqlite'
    central_db_connection = None

    def __init__(self, path_to_local_db=None):
        self.path_to_local_db = path_to_local_db
        self.local_db_connection = None

    def __del__(self):
        # Closing the connection to the database file
        for conn in (self.local_db_connection, self.central_db_connection):
            try:
                conn.close()
            except AttributeError:
                pass

    def open_connection(self, open_local, path_to_local_db=None):
        if open_local:
            path_to_local_db = path_to_local_db if path_to_local_db is not None else self.path_to_local_db
            self.local_db_connection = sqlite3.connect(path_to_local_db)
            cur = self.local_db_connection.cursor()
        else:
            self.central_db_connection = sqlite3.connect(DBManager.central_db_file_path)
            cur = self.central_db_connection.cursor()
        return cur

    def commit_and_close_connection(self, close_local):
        db_connection = self.local_db_connection if close_local else self.central_db_connection
        db_connection.commit()
        db_connection.close()

    def create_tables(self, create_local, path_to_local_db=None, drop_existing_tables=False):
        cur = self.open_connection(create_local, path_to_local_db)
        try:
            if drop_existing_tables:
                self._drop_tables(cur, create_local)

            if create_local:
                self._create_local_tables(cur)
            else:
                self._create_central_tables(cur)
        finally:
            self.commit_and_close_connection(create_local)

    @classmethod
    def _create_local_tables(cls, cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON;')

        # create images table
        cur.execute(cls.__table_to_creating_sql(Tables.images_table))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {Tables.images_table} ("
        #     f"{Columns.image_id_col} {Columns.image_id_col.col_type.value} UNIQUE NOT NULL, "
        #     f"{Columns.file_name_col} {Columns.file_name_col.col_type.value} NOT NULL, "
        #     f"{Columns.last_modified_col} {Columns.last_modified_col.col_type.value} NOT NULL, "
        #     f"PRIMARY KEY ({Columns.image_id_col})"
        #     ")"
        # )

        # create faces table
        cur.execute(cls.__table_to_creating_sql(Tables.faces_table))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {Tables.faces_table} ("
        #     f"{Columns.face_id_col} {Columns.face_id_col.col_type.value} UNIQUE NOT NULL, "
        #     f"{Columns.image_id_col} {Columns.image_id_col.col_type.value} NOT NULL, "
        #     f"{Columns.thumbnail_col} {Columns.thumbnail_col.col_type.value}, "
        #     f"PRIMARY KEY ({Columns.face_id_col})"
        #     f"FOREIGN KEY ({Columns.image_id_col}) REFERENCES {Tables.images_table} ({Columns.image_id_col})"
        #     " ON DELETE CASCADE"
        #     ")"
        # )

    @staticmethod
    def __table_to_creating_sql(table):
        constraints_sql = ", ".join(table.constraints)
        creating_sql = (
            f"CREATE TABLE IF NOT EXISTS {table} ("
            + ", ".join(f"{col.col_name} {col.col_type.value} {col.col_constraint}" for col in table.get_columns())
            + (", " if constraints_sql else "")
            + constraints_sql
            + ");"
        )
        return creating_sql

    @classmethod
    def _create_central_tables(cls, cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON;')

        # create embeddings table
        cur.execute(cls.__table_to_creating_sql(Tables.embeddings_table))
        # cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.embeddings_table} ('
        #             f'{Columns.cluster_id_col} {Columns.cluster_id_col.col_type.value} NOT NULL, '
        #             f'{Columns.face_id_col} {Columns.face_id_col.col_type.value} UNIQUE NOT NULL, '
        #             f'{Columns.embedding_col} {Columns.embedding_col.col_type.value} NOT NULL, '
        #             f'PRIMARY KEY ({Columns.face_id_col}), '
        #             f'FOREIGN KEY ({Columns.cluster_id_col}) REFERENCES {Tables.cluster_attributes_table} ({Columns.cluster_id_col})'
        #             ' ON DELETE CASCADE'
        #             ')')

        # create cluster attributes table
        cur.execute(cls.__table_to_creating_sql(Tables.cluster_attributes_table))
        # cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.cluster_attributes_table} ('
        #             f'{Columns.cluster_id_col} {Columns.cluster_id_col.col_type.value} NOT NULL, '
        #             f'{Columns.label_col} {Columns.label_col.col_type.value}, '
        #             f'{Columns.center_col} {Columns.center_col.col_type.value}, '
        #             f'PRIMARY KEY ({Columns.cluster_id_col})'
        #             ')')

    @staticmethod
    def _drop_tables(cur, drop_local=True):
        tables = Tables.local_tables if drop_local else Tables.central_tables
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS {table};')

    def store_in_table(self, table, row_dicts, path_to_local_db=None):
        """

        @param table:
        @param row_dicts: iterable of dicts storing (col_name, col_value)-pairs
        @param path_to_local_db:
        @return:
        """
        rows = DBManager.row_dicts_to_rows(table, row_dicts)
        store_in_local = Tables.is_local_table(table)
        values_template = self._make_values_template(len(row_dicts[0]))
        cur = self.open_connection(store_in_local, path_to_local_db)
        try:
            cur.executemany(f'INSERT INTO {table} VALUES ({values_template});', rows)
        finally:
            self.commit_and_close_connection(store_in_local)

    def fetch_from_table(self, table, path_to_local_db=None, cols=None, cond=''):
        """

        @param table:
        @param path_to_local_db:
        @param cols: An iterable of column names, or an iterable containing only the string '*' (default).
        @param cond:
        @return:
        """
        # TODO: allow for multiple conditions(?)
        if cols is None or '*' in cols:
            cols = table.get_column_names()
        cond_str = '' if len(cond) == 0 else f'WHERE {cond}'
        fetch_from_local = Tables.is_local_table(table)
        cols_template = ','.join(cols)
        cur = self.open_connection(fetch_from_local, path_to_local_db)
        try:
            sql_values = cur.execute(f'SELECT {cols_template} FROM {table} {cond_str};').fetchall()
        finally:
            self.commit_and_close_connection(fetch_from_local)
        result = [DBManager.sql_value_to_data(value, col) for value, col in zip(sql_values, cols)]
        return result

    def get_cluster_parts(self):
        # TODO: Refactor
        cur = self.open_connection(open_local=False)
        try:
            cluster_parts = cur.execute(
                f"SELECT {Columns.cluster_id_col}, {Columns.label_col},"
                f" {Columns.center_col}, {Columns.embedding_col}, "
                f"{Columns.face_id_col}"
                f" FROM {Tables.embeddings_table} INNER JOIN {Tables.cluster_attributes_table}"
                f" USING ({Columns.cluster_id_col});"
            ).fetchall()
        finally:
            self.commit_and_close_connection(close_local=False)
        # convert center point and embedding to tensors rather than bytes
        processed_cluster_parts = [
            (cluster_id,
             label,
             self.bytes_to_tensor(center_point),
             self.bytes_to_tensor(embedding),
             face_id)
            for cluster_id, label, center_point, embedding, face_id in cluster_parts
        ]
        return processed_cluster_parts

    def add_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Implement + Change signature + use
        pass

    def remove_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Implement + Change signature + use
        pass

    def check_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Change signature + use
        try:
            self.commit_and_close_connection(False)
            central_cur = self.central_db_connection.cursor()
        except ...:
            pass
        try:
            local_cur = self.local_db_connection.cursor()
        except ...:
            pass

    def aggregate_col(self, table, col, func, path_to_local_db=None):
        aggregate_from_local = Tables.is_local_table(table)
        cur = self.open_connection(aggregate_from_local, path_to_local_db)
        try:
            agg_value = cur.execute(
                f"SELECT {func}({col}) FROM {table};"
            ).fetchone()
        finally:
            self.commit_and_close_connection(aggregate_from_local)
        return agg_value

    def get_max_num(self, table, col, default=0, path_to_local_db=None):
        max_num = self.aggregate_col(table=table, col=col, func='MAX',
                                     path_to_local_db=path_to_local_db)[0]
        if isinstance(max_num, int) or isinstance(max_num, float):
            return max_num
        return default

    @classmethod
    def get_db_path(cls, path, local=True):
        if local:
            return os.path.join(path, cls.local_db_file_name)
        return cls.central_db_file_name

    @classmethod
    def row_dicts_to_rows(cls, table, row_dicts):
        sort_dict_by_cols = partial(table.sort_dict_by_cols, only_values=False)
        sorted_item_rows = list(map(sort_dict_by_cols,
                                    row_dicts))
        rows = []
        for item_row in sorted_item_rows:
            # col_names = get_every_nth_item(item_row, 0)
            # col_values = get_every_nth_item(item_row, 1)
            # is_blob_col = set(map(lambda k: table.get_column_type(k) == ColumnTypes.blob, col_names))
            # row = list(map(lambda is_blob, val: cls.data_to_bytes(val) if is_blob else val,
            #                zip(is_blob_col, col_values)))
            row = []
            for col_name, col_value in item_row:
                if isinstance(col_value, datetime.datetime):
                    row.append(cls.date_to_iso_string(col_value))
                elif table.get_column_type(col_name) == ColumnTypes.blob:
                    row.append(cls.data_to_bytes(col_value))
                else:
                    row.append(col_value)
            rows.append(row)
        return rows

    @staticmethod
    def _make_values_template(length, char_to_join='?', sep=','):
        chars_to_join = length * char_to_join if len(char_to_join) == 1 else char_to_join
        return sep.join(chars_to_join)

    @staticmethod
    def sql_value_to_data(value, details_obj):
        """

        @param value:
        @param details_obj: String or ColumnSchema
        @return:
        """
        details = _get_true_attr(details_obj, ColumnDetails, 'details_obj')
        if details == ColumnDetails.image:
            value = DBManager.bytes_to_image(value)
        elif details == ColumnDetails.tensor:
            value = DBManager.bytes_to_tensor(value)
        elif details == ColumnDetails.date:
            value = DBManager.iso_string_to_date(value)
        return value

    @staticmethod
    def data_to_bytes(data):
        """
        Convert the data (tensor or image) to bytes for storage as BLOB in DB.

        :param data: Either a PyTorch Tensor or a PILLOW Image.
        """
        buffer = io.BytesIO()
        if isinstance(data, torch.Tensor):  # case 1: embedding
            torch.save(data, buffer)
        else:  # case 2: thumbnail
            data.save(buffer, format='JPEG')
        data_bytes = buffer.getvalue()
        buffer.close()
        return data_bytes

    @classmethod
    def bytes_to_image(cls, data_bytes):
        return cls.bytes_to_data(data_bytes, data_type=ColumnDetails.image)

    @classmethod
    def bytes_to_tensor(cls, data_bytes):
        return cls.bytes_to_data(data_bytes, data_type=ColumnDetails.tensor)

    @staticmethod
    def bytes_to_data(data_bytes, data_type):
        """
        Convert the BLOB bytes from the DB to either a tensor or an image, depending on the data_type argument.

        :param data_bytes: Bytes from storing either a PyTorch Tensor or a PILLOW Image.
        :param data_type: String or ColumnDetails object denoting the original data type. One of 'tensor', 'image', or
        one of the corresponding ColumnDetails objects.
        """
        # TODO: Does this comparison always work?
        type_str = _get_true_attr(data_type, ColumnDetails, 'data_type')

        buffer = io.BytesIO(data_bytes)
        try:
            if type_str == str(ColumnDetails.tensor):
                obj = torch.load(buffer)
            elif type_str == str(ColumnDetails.image):
                obj = Image.open(buffer).convert('RGBA')
            else:
                raise ValueError(f"Unknown data type '{type_str}', expected '{str(ColumnDetails.tensor)}'"
                                 f" or '{str(ColumnDetails.image)}'.")
        finally:
            buffer.close()
        return obj

    @staticmethod
    def date_to_iso_string(date):
        return date.isoformat().replace('T', ' ')

    @staticmethod
    def iso_string_to_date(string):
        return datetime.datetime.fromisoformat(string)

    # def main(self):
    #     # connecting to the database file
    #     conn = sqlite3.connect(sqlite_file_name)
    #     c = conn.cursor()
    #
    #     create_tables(c, drop_existing_tables=True)
    #
    #     # populate images table
    #     store_in_images_table(c, [f'(1, "apple sauce", {round(time.time())})'])
    #     # c.execute(f'INSERT INTO {Tables.images_table} VALUES (1, "apple sauce", ?)',
    #     #           [round(time.time())])
    #
    #
    #     # populate faces table
    #     thumbnail = Image.open('preprocessed_Aaron_Eckhart_1.jpg')
    #     embedding = torch.load('Aaron_Eckhart_1.pt')
    #
    #     thumbnail_bytes = data_to_bytes(thumbnail)
    #     embedding_bytes = data_to_bytes(embedding)
    #
    #     store_in_faces_table(c, [f'(1, {thumbnail_bytes}, {embedding_bytes})'])
    #     # c.execute(f'INSERT INTO {Tables.faces_table} VALUES (1, ?, ?)',
    #     #           [thumbnail_bytes, embedding_bytes])
    #
    #     images_rows = fetch_all_from_images_table(c)
    #     faces_rows = fetch_all_from_faces_table(c)
    #     print(images_rows)
    #     print('\n'.join(map(str, faces_rows[0])))
    #
    #
    #     thumbnail_bytes, embedding_bytes = faces_rows[0][1:]
    #     # print(embedding_bytes)
    #
    #
    #     # convert back to tensor
    #     tensor = bytes_to_data(embedding_bytes, 'tensor')
    #     print(tensor)
    #
    #     # convert back to image
    #     image = bytes_to_data(thumbnail_bytes, 'image')
    #     image.show()
    #
    #     # Closing the connection to the database file
    #     conn.commit()
    #     conn.close()
    #
    #
    #     # col_type = ColumnTypes.text  # E.g., INTEGER, TEXT, NULL, REAL, BLOB
    #     # default_val = 'Hello World'  # a default value for the new col rows
    #
    #     # last_modified = os.stat('my_db.sqlite').st_atime
    #     # age = time.time() - last_modified
    #     # datetime.datetime.fromtimestamp(time.time())
    #
    #     # # Retrieve col information
    #     # # Every col will be represented by a tuple with the following attributes:
    #     # # (id, name, type, notnull, default_value, primary_key)
    #     # c.execute('PRAGMA TABLE_INFO({})'.format(images_table_name))
    #
    #     # # collect names in a list
    #     # names = [tup[1] for tup in c.fetchall()]
    #     # print(names)
    #     # # e.g., ['id', 'date', 'time', 'date_time']


# XXXXX

        #
        # # create embeddings table
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.embeddings_table} ('
        #                  f'{Columns.image_id_col} {Columns.image_id_col.col_type.value} NOT NULL, '
        #                  f'{Columns.thumbnail_col} {Columns.thumbnail_col.col_type.value}, '
        #                  f'{Columns.embedding_col} {Columns.embedding_col.col_type.value}, '
        #                  f'FOREIGN KEY ({Columns.image_id_col}) REFERENCES {Tables.images_table} ({Columns.image_id_col})'
        #                  ' ON DELETE CASCADE'
        #                  ')')
        #
        # # create cluster attributes table
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.cluster_attributes_table} ('
        #                  f'{Columns.image_id_col} {Columns.image_id_col.col_type.value} NOT NULL, '
        #                  f'{Columns.thumbnail_col} {Columns.thumbnail_col.col_type.value}, '
        #                  f'{Columns.embedding_col} {Columns.embedding_col.col_type.value}, '
        #                  f'FOREIGN KEY ({Columns.image_id_col}) REFERENCES {Tables.images_table} ({Columns.image_id_col})'
        #                  ' ON DELETE CASCADE'
        #                  ')')


if __name__ == '__main__':
    path_to_local_db = os.path.join(DBManager.db_files_path, DBManager.local_db_file_name)

    manager = DBManager(path_to_local_db)
    manager.create_tables(True)
    manager.create_tables(False)

    manager.store_in_table(Tables.images_table, [(1, "apple sauce", round(time.time()))])
    result = manager.fetch_from_table(Tables.images_table)
    print(result)

    """
    # populate images table
    c.execute(f'INSERT INTO {images_table} VALUES (1, "apple sauce", ?)',
              [round(time.time())])

    # populate faces table
    embedding = torch.load('Aaron_Eckhart_1.pt')
    thumbnail = Image.open('preprocessed_Aaron_Eckhart_1.jpg')

    buffer = io.BytesIO()
    thumbnail.save(buffer, format='JPEG')
    thumbnail_bytes = buffer.getvalue()
    buffer.close()

    buffer = io.BytesIO()
    torch.save(embedding, buffer)
    embedding_bytes = buffer.getvalue()
    buffer.close()

    c.execute(f'INSERT INTO {faces_table} VALUES (1, ?, ?)',
              [thumbnail_bytes, embedding_bytes])
    """
