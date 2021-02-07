# Credit: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import os
import sqlite3
import time
from functools import partial

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


CENTRAL_DB_FILE = 'central_db.sqlite'
LOCAL_DB_FILE = 'local_db.sqlite'

DB_FILES_PATH = 'database'


"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT face_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---
embeddings(INT cluster_id, INT face_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


# images table
IMAGES_TABLE = 'images'
IMAGE_ID_COL = ('image_id', 'INT')  # also used by faces table
FILE_NAME_COL = ('file_name', 'TEXT')
LAST_MODIFIED_COL = ('last_modified', 'INT')

# faces table
FACES_TABLE = 'faces'
FACE_ID_COL = ('face_id', 'INT')  # also used by embeddings table
THUMBNAIL_COL = ('thumbnail', 'BLOB')

LOCAL_TABLES = {IMAGES_TABLE, FACES_TABLE}

# embeddings table
EMBEDDINGS_TABLE = 'embeddings'
CLUSTER_ID_COL = ('cluster_id', 'INT')  # also used by cluster attributes table
EMBEDDING_COL = ('embedding', 'BLOB')

# cluster attributes table
CLUSTER_ATTRIBUTES_TABLE = 'cluster_attributes'
LABEL_COL = ('label', 'TEXT')
CENTER_COL = ('center', 'BLOB')

CENTRAL_TABLES = {EMBEDDINGS_TABLE, CLUSTER_ATTRIBUTES_TABLE}


class DBManager:
    def __init__(self, path_to_central_db, path_to_local_db):
        self.path_to_central_db = path_to_central_db
        self.path_to_local_db = path_to_local_db
        self.central_db_connection = None
        self.local_db_connection = None

    def __del__(self):
        # Closing the connection to the database file
        for conn in (self.local_db_connection, self.central_db_connection):
            try:
                conn.close()
            except AttributeError:
                pass

    # def handle_db_conn(self, func):
    #     def new_func(*args, **kwargs):
    #         cur = self.open_connection()
    #         result = func(cur, *args, **kwargs)
    #         self.close_connection()
    #         return result
    #     return new_func

    def open_connection(self, open_local):
        if open_local:
            self.local_db_connection = sqlite3.connect(self.path_to_local_db)
            cur = self.local_db_connection.cursor()
        else:
            self.central_db_connection = sqlite3.connect(self.path_to_central_db)
            cur = self.central_db_connection.cursor()
        return cur

    def close_connection(self, close_local):
        db_connection = self.local_db_connection if close_local else self.central_db_connection
        db_connection.commit()
        db_connection.close()

    def create_tables(self, create_local, drop_existing_tables=True):
        cur = self.open_connection(open_local=create_local)
        if drop_existing_tables:
            self._drop_tables(cur, drop_local=create_local)

        if create_local:
            self._create_local_tables(cur)
        else:
            self._create_central_tables(cur)
        self.close_connection(close_local=create_local)

    @staticmethod
    def _create_local_tables(cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON')

        # create images table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {IMAGES_TABLE} ('
                    f'{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} UNIQUE NOT NULL, '
                    f'{FILE_NAME_COL[0]} {FILE_NAME_COL[1]} NOT NULL, '
                    f'{LAST_MODIFIED_COL[0]} {LAST_MODIFIED_COL[1]} NOT NULL, '
                    f'PRIMARY KEY ({IMAGE_ID_COL[0]})'
                    ')')

        # create faces table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {FACES_TABLE} ('
                    f'{FACE_ID_COL[0]} {FACE_ID_COL[1]} UNIQUE NOT NULL, '
                    f'{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} NOT NULL, '
                    f'{THUMBNAIL_COL[0]} {THUMBNAIL_COL[1]}, '
                    f'PRIMARY KEY ({FACE_ID_COL[0]})'
                    f'FOREIGN KEY ({IMAGE_ID_COL[0]}) REFERENCES {IMAGES_TABLE} ({IMAGE_ID_COL[0]})'
                    ' ON DELETE CASCADE'
                    ')')

    @staticmethod
    def _create_central_tables(cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON')

        # create embeddings table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} ('
                    f'{CLUSTER_ID_COL[0]} {CLUSTER_ID_COL[1]} NOT NULL, '
                    f'{FACE_ID_COL[0]} {FACE_ID_COL[1]} UNIQUE NOT NULL, '
                    f'{EMBEDDING_COL[0]} {EMBEDDING_COL[1]} NOT NULL, '
                    f'PRIMARY KEY ({FACE_ID_COL[0]}), '
                    f'FOREIGN KEY ({CLUSTER_ID_COL[0]}) REFERENCES {CLUSTER_ATTRIBUTES_TABLE} ({CLUSTER_ID_COL[0]})'
                    ' ON DELETE CASCADE'
                    ')')

        # create cluster attributes table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {CLUSTER_ATTRIBUTES_TABLE} ('
                    f'{CLUSTER_ID_COL[0]} {CLUSTER_ID_COL[1]} NOT NULL, '
                    f'{LABEL_COL[0]} {LABEL_COL[1]}, '
                    f'{CENTER_COL[0]} {CENTER_COL[1]}, '
                    f'PRIMARY KEY ({CLUSTER_ID_COL[0]})'
                    ')')

    @staticmethod
    def _drop_tables(cur, drop_local=True):
        tables = LOCAL_TABLES if drop_local else CENTRAL_TABLES
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS {table}')

    def store_in_table(self, table_name, rows):
        store_in_local = self.is_local_table(table_name)
        cur = self.open_connection(open_local=store_in_local)
        # cur.execute(f'INSERT INTO {table_name} VALUES ?',
        #             rows)
        values_template = self._make_values_template(len(rows[0]))
        cur.executemany(f'INSERT INTO {table_name} VALUES {values_template}', rows)
        self.close_connection(close_local=store_in_local)

    def fetch_from_table(self, table_name, cols=None):
        if cols is None:
            cols = ['*']
        fetch_from_local = self.is_local_table(table_name)
        cur = self.open_connection(open_local=fetch_from_local)
        cols_template = ','.join(cols)
        result = cur.execute(f'SELECT {cols_template} FROM {table_name}').fetchall()
        self.close_connection(close_local=fetch_from_local)
        return result

    def add_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Implement + Change signature + use
        pass

    def remove_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Implement + Change signature + use
        pass

    def check_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: Change signature + use
        try:
            self.close_connection(False)
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
    path_to_central_db = os.path.join(DB_FILES_PATH, LOCAL_DB_FILE)
    path_to_local_db = os.path.join(DB_FILES_PATH, CENTRAL_DB_FILE)

    manager = DBManager(path_to_central_db, path_to_local_db)
    manager.create_tables(create_local=True)
    manager.create_tables(create_local=False)

    manager.store_in_table(IMAGES_TABLE, [(1, "apple sauce", round(time.time()))])
    result = manager.fetch_from_table(IMAGES_TABLE)
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
