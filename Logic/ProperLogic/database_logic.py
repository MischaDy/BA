# Credits: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import os
import sqlite3
import time

import torch
from PIL import Image
import io

from Logic.ProperLogic.misc_helpers import clean_str, have_equal_attrs, have_equal_type_names

# TODO: *Global* face_id - created in central table, then written to corresponding local table
# TODO: Foreign keys despite separate db files???
# TODO: Refactor to programmatically connect columns and their properties (uniqueness etc.). But multiple columns???
#       --> Make Tables class
# TODO: When to close connection? Optimize?
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?
# TODO: Use SQLAlchemy?

# TODO: Append semi-colons everywhere?
# TODO: FK faces -> embeddings other way around? Or remove completely?
# TODO: Consistent interface! When to pass objects (tables, columns), when to pass only their names??


"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT face_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---
embeddings(INT cluster_id, INT face_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


class TableSchema:
    def __init__(self, name, columns, constraints=None):
        self.name = name
        self.columns = {}
        for col in columns:
            self.columns[col.col_name] = col
        if constraints is None:
            constraints = []
        self.constraints = constraints

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.columns[item]
        elif isinstance(item, ColumnSchema):
            return self.columns[item.col_name]
        return TypeError(f'Item {item} must be of type str or ColumnSchema')

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def get_columns(self):
        return self.get_column_dict().values()

    def get_column_names(self):
        return self.get_column_dict().keys()

    def get_column_dict(self):
        return self.columns


class ColumnSchema:
    def __init__(self, col_name, col_type, col_constraint=''):
        self.col_name = col_name
        self.col_type = col_type
        self.col_constraint = col_constraint

    def __str__(self):
        return self.col_name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def with_constraint(self, col_constraint):
        return ColumnSchema(self.col_name, self.col_type, col_constraint)


class Columns:
    center_col = ColumnSchema('center', 'BLOB')
    cluster_id_col = ColumnSchema('cluster_id', 'INT')
    embedding_col = ColumnSchema('embedding', 'BLOB')
    face_id_col = ColumnSchema('face_id', 'INT')
    file_name_col = ColumnSchema('file_name', 'TEXT')
    image_id_col = ColumnSchema('image_id', 'INT')
    label_col = ColumnSchema('label', 'TEXT')
    last_modified_col = ColumnSchema('last_modified', 'INT')
    thumbnail_col = ColumnSchema('thumbnail', 'BLOB')


class Tables:
    images_table = TableSchema(
        'images',
        [Columns.image_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by faces table
         Columns.file_name_col.with_constraint('NOT NULL'),
         Columns.last_modified_col.with_constraint('NOT NULL')
         ],
        [f'PRIMARY KEY ({Columns.image_id_col})'
         ]
    )

    faces_table = TableSchema(
        'faces',
        [Columns.face_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by embeddings table
         Columns.image_id_col.with_constraint('NOT NULL'),
         Columns.thumbnail_col.with_constraint('BLOB')
         ],
        [f'PRIMARY KEY ({Columns.face_id_col})',
         f'FOREIGN KEY ({Columns.image_id_col}) REFERENCES {images_table} ({Columns.image_id_col})'
         + ' ON DELETE CASCADE'
         ]
    )

    local_tables = (images_table, faces_table)

    cluster_attributes_table = TableSchema(
        'cluster_attributes',
        [Columns.cluster_id_col.with_constraint('NOT NULL'),  # also used by cluster attributes table
         Columns.label_col,
         Columns.center_col
         ],
        [f'PRIMARY KEY ({Columns.cluster_id_col})']
    )

    embeddings_table = TableSchema(
        'embeddings',
        [Columns.cluster_id_col.with_constraint('NOT NULL'),  # also used by cluster attributes table
         Columns.face_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by embeddings table
         Columns.embedding_col.with_constraint('NOT NULL')
         ],
        [f'PRIMARY KEY ({Columns.face_id_col})',
         f'FOREIGN KEY ({Columns.cluster_id_col}) REFERENCES {cluster_attributes_table} ({Columns.cluster_id_col})'
         + ' ON DELETE CASCADE'
         ]
    )

    central_tables = (embeddings_table, cluster_attributes_table)

    @staticmethod
    def is_local_table(table):
        return table in Tables.local_tables

# TODO: Guarantee that connection is closed at end of methods
#       --> Using try (/ except) / finally??


# TODO: Make singleton object?
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
        if drop_existing_tables:
            self._drop_tables(cur, create_local)

        if create_local:
            self._create_local_tables(cur)
        else:
            self._create_central_tables(cur)
        self.commit_and_close_connection(create_local)

    @classmethod
    def _create_local_tables(cls, cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON')

        # create images table
        cur.execute(cls.__table_to_creating_sql(Tables.images_table))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {Tables.images_table} ("
        #     f"{Columns.image_id_col} {Columns.image_id_col.col_type} UNIQUE NOT NULL, "
        #     f"{Columns.file_name_col} {Columns.file_name_col.col_type} NOT NULL, "
        #     f"{Columns.last_modified_col} {Columns.last_modified_col.col_type} NOT NULL, "
        #     f"PRIMARY KEY ({Columns.image_id_col})"
        #     ")"
        # )

        # create faces table
        cur.execute(cls.__table_to_creating_sql(Tables.faces_table))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {Tables.faces_table} ("
        #     f"{Columns.face_id_col} {Columns.face_id_col.col_type} UNIQUE NOT NULL, "
        #     f"{Columns.image_id_col} {Columns.image_id_col.col_type} NOT NULL, "
        #     f"{Columns.thumbnail_col} {Columns.thumbnail_col.col_type}, "
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
            + ", ".join(f"{col.col_name} {col.col_type} {col.col_constraint}" for col in table.get_columns())
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
        #             f'{Columns.cluster_id_col} {Columns.cluster_id_col.col_type} NOT NULL, '
        #             f'{Columns.face_id_col} {Columns.face_id_col.col_type} UNIQUE NOT NULL, '
        #             f'{Columns.embedding_col} {Columns.embedding_col.col_type} NOT NULL, '
        #             f'PRIMARY KEY ({Columns.face_id_col}), '
        #             f'FOREIGN KEY ({Columns.cluster_id_col}) REFERENCES {Tables.cluster_attributes_table} ({Columns.cluster_id_col})'
        #             ' ON DELETE CASCADE'
        #             ')')

        # create cluster attributes table
        cur.execute(cls.__table_to_creating_sql(Tables.cluster_attributes_table))
        # cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.cluster_attributes_table} ('
        #             f'{Columns.cluster_id_col} {Columns.cluster_id_col.col_type} NOT NULL, '
        #             f'{Columns.label_col} {Columns.label_col.col_type}, '
        #             f'{Columns.center_col} {Columns.center_col.col_type}, '
        #             f'PRIMARY KEY ({Columns.cluster_id_col})'
        #             ')')

    @staticmethod
    def _drop_tables(cur, drop_local=True):
        tables = Tables.local_tables if drop_local else Tables.central_tables
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS {table};')

    def store_in_table(self, table, rows, path_to_local_db=None):
        # TODO: Call data_to_bytes when appropriate. --> How to know??
        store_in_local = Tables.is_local_table(table)
        cur = self.open_connection(store_in_local, path_to_local_db)
        # cur.execute(f'INSERT INTO {table_name} VALUES ?',
        #             rows)
        values_template = self._make_values_template(len(rows[0]))
        cur.executemany(f'INSERT INTO {table} VALUES ({values_template});', rows)
        self.commit_and_close_connection(store_in_local)

    def fetch_from_table(self, table_name, path_to_local_db=None, cols=None, cond=''):
        # TODO: Call bytes_to_data when appropriate. --> How to know??
        if cols is None:
            cols = ['*']
        cond_str = '' if len(cond) == 0 else f'WHERE {cond}'
        fetch_from_local = Tables.is_local_table(table_name)
        cur = self.open_connection(fetch_from_local, path_to_local_db)
        cols_template = ','.join(cols)
        result = cur.execute(f'SELECT {cols_template} FROM {table_name} {cond_str}').fetchall()
        self.commit_and_close_connection(fetch_from_local)
        return result

    def get_cluster_parts(self):
        cur = self.open_connection(open_local=False)
        cluster_parts = cur.execute(
            f"SELECT {Columns.cluster_id_col}, {Columns.label_col},"
            f" {Columns.center_col}, {Columns.embedding_col}, "
            f"{Columns.face_id_col}"
            f" FROM {Tables.embeddings_table} INNER JOIN {Tables.cluster_attributes_table}"
            f" USING ({Columns.cluster_id_col});"
        ).fetchall()
        self.commit_and_close_connection(close_local=False)
        return cluster_parts

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
        # TODO: Simplify signature? (pointer in Column class to parent table)
        aggregate_from_local = Tables.is_local_table(table)
        cur = self.open_connection(aggregate_from_local, path_to_local_db)
        agg_value = cur.execute(
            f"SELECT {func}({col}) FROM {table};"
        ).fetchone()
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

    @staticmethod
    def _make_values_template(length, char_to_join='?', sep=','):
        chars_to_join = length * char_to_join if len(char_to_join) == 1 else char_to_join
        return sep.join(chars_to_join)

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

    @staticmethod
    def bytes_to_data(data_bytes, data_type):
        """
        Convert the BLOB bytes from the DB to either a tensor or an image, depending on the data_type argument.

        :param data_bytes: Bytes from storing either a PyTorch Tensor or a PILLOW Image.
        :param data_type: String denoting the original data type. One of: 'tensor', 'image'.
        """
        type_str = clean_str(data_type)
        if type_str not in ('tensor', 'image'):
            raise ValueError("data_type must be one of 'tensor', 'image'")
        
        buffer = io.BytesIO(data_bytes)
        if type_str == 'tensor':
            obj = torch.load(buffer)
        else:
            obj = Image.open(buffer).convert('RGBA')
        buffer.close()
        return obj

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
    #     # col_type = 'TEXT'  # E.g., INTEGER, TEXT, NULL, REAL, BLOB
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
        #                  f'{Columns.image_id_col} {Columns.image_id_col.col_type} NOT NULL, '
        #                  f'{Columns.thumbnail_col} {Columns.thumbnail_col.col_type}, '
        #                  f'{Columns.embedding_col} {Columns.embedding_col.col_type}, '
        #                  f'FOREIGN KEY ({Columns.image_id_col}) REFERENCES {Tables.images_table} ({Columns.image_id_col})'
        #                  ' ON DELETE CASCADE'
        #                  ')')
        #
        # # create cluster attributes table
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {Tables.cluster_attributes_table} ('
        #                  f'{Columns.image_id_col} {Columns.image_id_col.col_type} NOT NULL, '
        #                  f'{Columns.thumbnail_col} {Columns.thumbnail_col.col_type}, '
        #                  f'{Columns.embedding_col} {Columns.embedding_col.col_type}, '
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
