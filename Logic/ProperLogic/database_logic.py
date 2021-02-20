# Credit: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import os
import sqlite3
import time

import torch
from PIL import Image
import io

from Logic.misc_helpers import clean_str

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
        return self.columns[item]

    def get_columns(self):
        return self.get_column_dict().values()

    def get_column_names(self):
        return self.get_column_dict().keys()

    def get_column_dict(self):
        return self.columns


class Column:
    def __init__(self, col_name, col_type, col_constraint=''):
        self.col_name = col_name
        self.col_type = col_type
        self.col_constraint = col_constraint


# images table
__image_id_col = Column('image_id', 'INT', 'UNIQUE NOT NULL')
__file_name_col = Column('file_name', 'TEXT', 'NOT NULL')
__last_modified_col = Column('last_modified', 'INT', 'NOT NULL')
IMAGES_TABLE = TableSchema(
    'images',
    [__image_id_col,  # also used by faces table
     __file_name_col,
     __last_modified_col
     ],
    [f'PRIMARY KEY ({__image_id_col.col_name})'
     ]
)
del __image_id_col, __file_name_col, __last_modified_col

# faces table
__face_id_col = Column('face_id', 'INT', 'UNIQUE NOT NULL')
__image_id_col = Column('image_id', 'INT', 'NOT NULL')
__thumbnail_col = Column('thumbnail', 'BLOB')
FACES_TABLE = TableSchema(
    'faces',
    [__face_id_col,  # also used by embeddings table
     __image_id_col,
     __thumbnail_col
     ],
    [f'PRIMARY KEY ({__face_id_col.col_name})',
     f'FOREIGN KEY ({__image_id_col.col_name}) REFERENCES {IMAGES_TABLE.name} ({__image_id_col.col_name})'
     + ' ON DELETE CASCADE'
     ]
)
del __face_id_col, __image_id_col, __thumbnail_col

LOCAL_TABLES = {IMAGES_TABLE, FACES_TABLE}


# cluster attributes table
__cluster_id_col = Column('cluster_id', 'INT', 'NOT NULL')
__label_col = Column('label', 'TEXT')
__center_col = Column('center', 'BLOB')
CLUSTER_ATTRIBUTES_TABLE = TableSchema(
    'cluster_attributes',
    [__cluster_id_col,  # also used by cluster attributes table
     __label_col,
     __center_col
     ],
    [f'PRIMARY KEY ({__cluster_id_col.col_name})']
)
del __cluster_id_col, __label_col, __center_col

# embeddings table
__cluster_id_col = Column('cluster_id', 'INT', 'NOT NULL')
__face_id_col = Column('face_id', 'INT', 'UNIQUE NOT NULL')
__embedding_col = Column('embedding', 'BLOB', 'NOT NULL')
EMBEDDINGS_TABLE = TableSchema(
    'embeddings',
    [__cluster_id_col,  # also used by cluster attributes table
     __face_id_col,  # also used by embeddings table
     __embedding_col
     ],
    [f'PRIMARY KEY ({__face_id_col.col_name})',
     f'FOREIGN KEY ({__cluster_id_col.col_name}) REFERENCES {CLUSTER_ATTRIBUTES_TABLE.name} ({__cluster_id_col.col_name})'
     + ' ON DELETE CASCADE'
     ]
)
del __cluster_id_col, __face_id_col, __embedding_col

CENTRAL_TABLES = {EMBEDDINGS_TABLE, CLUSTER_ATTRIBUTES_TABLE}


# TODO: Guarantee that connection is closed at end of methods
#       --> Using try (/ except) / finally??

# TODO: Make singleton object?
class DBManager:
    db_files_path = 'database'
    central_db_file_path = 'central_db.sqlite'
    local_db_file_name = 'local_db.sqlite'
    central_db_connection = None

    def __init__(self, path_to_central_db, path_to_local_db=None):
        self.path_to_central_db = path_to_central_db
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
            self.central_db_connection = sqlite3.connect(self.path_to_central_db)
            cur = self.central_db_connection.cursor()
        return cur

    def commit_and_close_connection(self, close_local):
        db_connection = self.local_db_connection if close_local else self.central_db_connection
        db_connection.commit()
        db_connection.close()

    def create_tables(self, create_local, path_to_local_db=None, drop_existing_tables=True):
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
        cur.execute(cls.__table_to_creating_sql(IMAGES_TABLE))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {IMAGES_TABLE.name} ("
        #     f"{IMAGES_TABLE['image_id'].col_name} {IMAGES_TABLE['image_id'].col_type} UNIQUE NOT NULL, "
        #     f"{IMAGES_TABLE['file_name'].col_name} {IMAGES_TABLE['file_name'].col_type} NOT NULL, "
        #     f"{LAST_MODIFIED_COL[0]} {LAST_MODIFIED_COL[1]} NOT NULL, "
        #     f"PRIMARY KEY ({IMAGE_ID_COL[0]})"
        #     ")"
        # )

        # create faces table
        cur.execute(cls.__table_to_creating_sql(FACES_TABLE))
        # cur.execute(
        #     f"CREATE TABLE IF NOT EXISTS {FACES_TABLE.name} ("
        #     f"{FACE_ID_COL[0]} {FACE_ID_COL[1]} UNIQUE NOT NULL, "
        #     f"{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} NOT NULL, "
        #     f"{THUMBNAIL_COL[0]} {THUMBNAIL_COL[1]}, "
        #     f"PRIMARY KEY ({FACE_ID_COL[0]})"
        #     f"FOREIGN KEY ({IMAGE_ID_COL[0]}) REFERENCES {IMAGES_TABLE.name} ({IMAGE_ID_COL[0]})"
        #     " ON DELETE CASCADE"
        #     ")"
        # )

    @staticmethod
    def __table_to_creating_sql(table):
        constraints_sql = ", ".join(table.constraints)
        creating_sql = (
            f"CREATE TABLE IF NOT EXISTS {table.name} ("
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
        cur.execute(cls.__table_to_creating_sql(EMBEDDINGS_TABLE))
        # cur.execute(f'CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE.name} ('
        #             f'{CLUSTER_ID_COL[0]} {CLUSTER_ID_COL[1]} NOT NULL, '
        #             f'{FACE_ID_COL[0]} {FACE_ID_COL[1]} UNIQUE NOT NULL, '
        #             f'{EMBEDDING_COL[0]} {EMBEDDING_COL[1]} NOT NULL, '
        #             f'PRIMARY KEY ({FACE_ID_COL[0]}), '
        #             f'FOREIGN KEY ({CLUSTER_ID_COL[0]}) REFERENCES {CLUSTER_ATTRIBUTES_TABLE.name} ({CLUSTER_ID_COL[0]})'
        #             ' ON DELETE CASCADE'
        #             ')')

        # create cluster attributes table
        cur.execute(cls.__table_to_creating_sql(CLUSTER_ATTRIBUTES_TABLE))
        # cur.execute(f'CREATE TABLE IF NOT EXISTS {CLUSTER_ATTRIBUTES_TABLE.name} ('
        #             f'{CLUSTER_ID_COL[0]} {CLUSTER_ID_COL[1]} NOT NULL, '
        #             f'{LABEL_COL[0]} {LABEL_COL[1]}, '
        #             f'{CENTER_COL[0]} {CENTER_COL[1]}, '
        #             f'PRIMARY KEY ({CLUSTER_ID_COL[0]})'
        #             ')')

    @staticmethod
    def _drop_tables(cur, drop_local=True):
        tables = LOCAL_TABLES if drop_local else CENTRAL_TABLES
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS {table.name};')

    def store_in_table(self, table_name, rows, path_to_local_db=None):
        store_in_local = self.is_local_table(table_name)
        cur = self.open_connection(store_in_local, path_to_local_db)
        # cur.execute(f'INSERT INTO {table_name} VALUES ?',
        #             rows)
        values_template = self._make_values_template(len(rows[0]))
        cur.executemany(f'INSERT INTO {table_name} VALUES {values_template};', rows)
        self.commit_and_close_connection(store_in_local)

    def fetch_from_table(self, table_name, path_to_local_db=None, cols=None, cond=''):
        if cols is None:
            cols = ['*']
        cond_str = '' if len(cond) == 0 else f'WHERE {cond}'
        fetch_from_local = self.is_local_table(table_name)
        cur = self.open_connection(fetch_from_local, path_to_local_db)
        cols_template = ','.join(cols)
        result = cur.execute(f'SELECT {cols_template} FROM {table_name} {cond_str}').fetchall()
        self.commit_and_close_connection(fetch_from_local)
        return result

    def get_cluster_parts(self):
        cur = self.open_connection(open_local=False)
        cluster_parts = cur.execute(
            f"SELECT {EMBEDDINGS_TABLE['cluster_id'].col_name}, {CLUSTER_ATTRIBUTES_TABLE['label'].col_name},"
            f" {CLUSTER_ATTRIBUTES_TABLE['center'].col_name}, {EMBEDDINGS_TABLE['embedding'].col_name}, "
            f"{EMBEDDINGS_TABLE['face_id'].col_name}"
            f" FROM {EMBEDDINGS_TABLE.name} INNER JOIN {CLUSTER_ATTRIBUTES_TABLE.name}"
            f" USING ({EMBEDDINGS_TABLE['cluster_id'].col_name});"
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

    @staticmethod
    def _make_values_template(length, char_to_join='?', sep=','):
        chars_to_join = length * char_to_join if len(char_to_join) == 1 else char_to_join
        return sep.join(chars_to_join)

    @staticmethod
    def is_local_table(table_name):
        return table_name in LOCAL_TABLES

    @staticmethod
    def data_to_bytes(data):
        """
        Convert the data (tensor or image) to bytes for storage as BLOB in DB.

        :param data: Either a PyTorch Tensor or a PILLOW Image.
        """
        buffer = io.BytesIO()
        if type(data) is torch.Tensor:  # case 1: embedding
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
        buffer = io.BytesIO(data_bytes)
        if clean_str(data_type) == 'tensor':
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
    #     # c.execute(f'INSERT INTO {IMAGES_TABLE} VALUES (1, "apple sauce", ?)',
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
    #     # c.execute(f'INSERT INTO {FACES_TABLE} VALUES (1, ?, ?)',
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
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} ('
        #                  f'{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} NOT NULL, '
        #                  f'{THUMBNAIL_COL[0]} {THUMBNAIL_COL[1]}, '
        #                  f'{EMBEDDING_COL[0]} {EMBEDDING_COL[1]}, '
        #                  f'FOREIGN KEY ({IMAGE_ID_COL[0]}) REFERENCES {IMAGES_TABLE} ({IMAGE_ID_COL[0]})'
        #                  ' ON DELETE CASCADE'
        #                  ')')
        #
        # # create cluster attributes table
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {CLUSTER_ATTRIBUTES_TABLE} ('
        #                  f'{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} NOT NULL, '
        #                  f'{THUMBNAIL_COL[0]} {THUMBNAIL_COL[1]}, '
        #                  f'{EMBEDDING_COL[0]} {EMBEDDING_COL[1]}, '
        #                  f'FOREIGN KEY ({IMAGE_ID_COL[0]}) REFERENCES {IMAGES_TABLE} ({IMAGE_ID_COL[0]})'
        #                  ' ON DELETE CASCADE'
        #                  ')')


if __name__ == '__main__':
    path_to_central_db = os.path.join(DBManager.db_files_path, DBManager.central_db_file_path)
    path_to_local_db = os.path.join(DBManager.db_files_path, DBManager.local_db_file_name)

    manager = DBManager(path_to_central_db, path_to_local_db)
    manager.create_tables(True)
    manager.create_tables(False)

    manager.store_in_table(IMAGES_TABLE.name, [(1, "apple sauce", round(time.time()))])
    result = manager.fetch_from_table(IMAGES_TABLE.name)
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
