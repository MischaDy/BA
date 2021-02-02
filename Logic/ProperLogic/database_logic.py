# Credit: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html


import sqlite3
import time
import torch
from PIL import Image
import io

from Logic.misc_helpers import clean_str


# TODO: Foreign keys despite separate db files???
# TODO: Refactor to programmatically connect columns and their properties (uniqueness etc.). But multiple columns???
#       --> Make Tables class
# TODO: When to close connection? Optimize?
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?


CENTRAL_DB_FILE = 'central_db.sqlite'
DECENTRAL_DB_FILE = 'db_realistic_test.sqlite'


"""
----- DB SCHEMA -----


--- Decentralized Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT face_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---
clusters(INT cluster_id, INT face_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB mean)
"""


# images table
IMAGES_TABLE = 'images'
IMAGE_ID_COL = ('image_id', 'INT')  # also used by faces table
FILE_NAME_COL = ('file_name', 'TEXT')
LAST_MODIFIED_COL = ('last_modified', 'INT')

# faces table
FACES_TABLE = 'faces'
FACE_ID_COL = ('face_id', 'INT')  # also used by clusters table
THUMBNAIL_COL = ('thumbnail', 'BLOB')

# clusters table
CLUSTERS_TABLE = 'clusters'
CLUSTER_ID_COL = ('cluster_id', 'INT')  # also used by cluster attributes table
EMBEDDING_COL = ('embedding', 'BLOB')

# cluster attributes table
CLUSTER_ATTRIBUTES_TABLE = 'cluster_attributes'
LABEL_COL = ('label', 'TEXT')
MEAN_COL = ('mean', 'BLOB')


class DBManager:
    def __init__(self, sqlite_file):
        self.file_name = sqlite_file
        self.connection = None

    def __del__(self):
        # Closing the connection to the database file
        self.connection.commit()
        self.connection.close()

    def open_connection(self):
        self.connection = sqlite3.connect(self.file_name)
        return self.connection.cursor()

    def close_connection(self):
        self.connection.commit()
        self.connection.close()

    def create_local_tables(self, drop_existing_tables=True):
        cur = self.open_connection()

        if drop_existing_tables:
            cur.execute(f'DROP TABLE IF EXISTS {IMAGES_TABLE}')
            cur.execute(f'DROP TABLE IF EXISTS {FACES_TABLE}')

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

        self.close_connection()

    def create_central_tables(self, drop_existing_tables=True):
        cur = self.open_connection()

        if drop_existing_tables:
            cur.execute(f'DROP TABLE IF EXISTS {CLUSTERS_TABLE}')
            cur.execute(f'DROP TABLE IF EXISTS {CLUSTER_ATTRIBUTES_TABLE}')

        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON')

        # create images table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {CLUSTERS_TABLE} ('
                    f'{CLUSTER_ID_COL[0]} {CLUSTER_ID_COL[1]} UNIQUE NOT NULL, '
                    f'{FACE_ID_COL[0]} {FACE_ID_COL[1]} NOT NULL, '
                    f'{LAST_MODIFIED_COL[0]} {LAST_MODIFIED_COL[1]} NOT NULL, '
                    f'PRIMARY KEY ({CLUSTER_ID_COL[0]})'
                    ')')

        # create faces table
        cur.execute(f'CREATE TABLE IF NOT EXISTS {FACES_TABLE} ('
                    f'{IMAGE_ID_COL[0]} {IMAGE_ID_COL[1]} NOT NULL, '
                    f'{THUMBNAIL_COL[0]} {THUMBNAIL_COL[1]}, '
                    f'{EMBEDDING_COL[0]} {EMBEDDING_COL[1]}, '
                    f'FOREIGN KEY ({IMAGE_ID_COL[0]}) REFERENCES {IMAGES_TABLE} ({IMAGE_ID_COL[0]})'
                    ' ON DELETE CASCADE'
                    ')')

        # --- Centralized Tables ---
        # clusters(INT cluster_id, INT face_id, BLOB face_embedding)
        # cluster_attributes(INT cluster_id, TEXT label, BLOB mean)

        # # clusters table
        # CLUSTERS_TABLE = 'clusters'
        # CLUSTER_ID_COL = ('cluster_id', 'INT')  # also used by cluster attributes table
        # EMBEDDING_COL = ('embedding', 'BLOB')
        #
        # # cluster attributes table
        # CLUSTER_ATTRIBUTES_TABLE = 'cluster_attributes'
        # LABEL_COL = ('label', 'TEXT')
        # MEAN_COL = ('mean', 'BLOB')

        self.close_connection()

    def store_in_images_table(self, rows):
        self._store_in_table(IMAGES_TABLE, rows)

    def store_in_faces_table(self, rows):
        self._store_in_table(FACES_TABLE, rows)

    def _store_in_table(self, table_name, rows):
        self.cur.execute(f'INSERT INTO {table_name} VALUES ?',
                         rows)

    def fetch_all_from_images_table(self):
        return self._fetch_all_from_table(IMAGES_TABLE)

    def fetch_all_from_faces_table(self):
        return self._fetch_all_from_table(FACES_TABLE)

    def _fetch_all_from_table(self, table_name):
        return self.cur.execute(f'SELECT * FROM {table_name}').fetchall()

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

    def main(self):
        # connecting to the database file
        conn = sqlite3.connect(sqlite_file_name)
        c = conn.cursor()

        create_tables(c, drop_existing_tables=True)

        # populate images table
        store_in_images_table(c, [f'(1, "apple sauce", {round(time.time())})'])
        # c.execute(f'INSERT INTO {IMAGES_TABLE} VALUES (1, "apple sauce", ?)',
        #           [round(time.time())])


        # populate faces table
        thumbnail = Image.open('preprocessed_Aaron_Eckhart_1.jpg')
        embedding = torch.load('Aaron_Eckhart_1.pt')

        thumbnail_bytes = data_to_bytes(thumbnail)
        embedding_bytes = data_to_bytes(embedding)

        store_in_faces_table(c, [f'(1, {thumbnail_bytes}, {embedding_bytes})'])
        # c.execute(f'INSERT INTO {FACES_TABLE} VALUES (1, ?, ?)',
        #           [thumbnail_bytes, embedding_bytes])

        images_rows = fetch_all_from_images_table(c)
        faces_rows = fetch_all_from_faces_table(c)
        print(images_rows)
        print('\n'.join(map(str, faces_rows[0])))


        thumbnail_bytes, embedding_bytes = faces_rows[0][1:]
        # print(embedding_bytes)


        # convert back to tensor
        tensor = bytes_to_data(embedding_bytes, 'tensor')
        print(tensor)

        # convert back to image
        image = bytes_to_data(thumbnail_bytes, 'image')
        image.show()

        # Closing the connection to the database file
        conn.commit()
        conn.close()


        # col_type = 'TEXT'  # E.g., INTEGER, TEXT, NULL, REAL, BLOB
        # default_val = 'Hello World'  # a default value for the new col rows

        # last_modified = os.stat('my_db.sqlite').st_atime
        # age = time.time() - last_modified
        # datetime.datetime.fromtimestamp(time.time())

        # # Retrieve col information
        # # Every col will be represented by a tuple with the following attributes:
        # # (id, name, type, notnull, default_value, primary_key)
        # c.execute('PRAGMA TABLE_INFO({})'.format(images_table_name))

        # # collect names in a list
        # names = [tup[1] for tup in c.fetchall()]
        # print(names)
        # # e.g., ['id', 'date', 'time', 'date_time']




# XXXXX

        #
        # # create clusters table
        # self.cur.execute(f'CREATE TABLE IF NOT EXISTS {CLUSTERS_TABLE} ('
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
    m = DBManager(sqlite_file_name)
    m.fetch_all_from_images_table()

