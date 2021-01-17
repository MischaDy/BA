# Credit: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html


import sqlite3
import time
import torch
from PIL import Image
import io


sqlite_file_name = 'db_realistic_test.sqlite'

# TODO: Refactor to programmatically connect columns and their properties (uniqueness etc.). But multiple columns???
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?


# images table
images_table = 'images'
file_name_col = ('file_name', 'TEXT')
last_modified_col = ('last_modified', 'INT')

# used by both tables
image_id_col = ('image_id', 'INT')

# faces table (= thumbnails + embeddings)
faces_table = 'faces'
thumbnail_col = ('thumbnail', 'BLOB')
embedding_col = ('embedding', 'BLOB')


# connecting to the database file
conn = sqlite3.connect(sqlite_file_name)
c = conn.cursor()

# enable foreign keys
c.execute('PRAGMA foreign_keys = ON')

c.execute(f'DROP TABLE IF EXISTS {images_table}')
c.execute(f'DROP TABLE IF EXISTS {faces_table}')

# create images table
c.execute(f'CREATE TABLE IF NOT EXISTS {images_table} ('
          f'{image_id_col[0]} {image_id_col[1]} UNIQUE NOT NULL, '
          f'{file_name_col[0]} {file_name_col[1]} NOT NULL, '
          f'{last_modified_col[0]} {last_modified_col[1]} NOT NULL, '
          f'PRIMARY KEY ({image_id_col[0]})'
          ')')

# create faces table
c.execute(f'CREATE TABLE IF NOT EXISTS {faces_table} ('
          f'{image_id_col[0]} {image_id_col[1]} NOT NULL, '
          f'{thumbnail_col[0]} {thumbnail_col[1]}, '
          f'{embedding_col[0]} {embedding_col[1]}, '
          f'FOREIGN KEY ({image_id_col[0]}) REFERENCES {images_table} ({image_id_col[0]}) ON DELETE CASCADE'
          ')')


# populate images table
c.execute(f'INSERT INTO {images_table} VALUES (1, "apple sauce", ?)',
          [round(time.time())])


# populate faces table
embedding = torch.load('Aaron_Eckhart_1.pt')
thumbnail = Image.open('preprocessed_Aaron_Eckhart_1.jpg')

stream = io.BytesIO()
thumbnail.save(stream, format='JPEG')
thumbnail_bytes = stream.getvalue()
torch.save(embedding, stream)
embedding_bytes = stream.getvalue()
stream.close()
c.execute(f'INSERT INTO {faces_table} VALUES (1, ?, ?)',
          [thumbnail_bytes, embedding_bytes])


images_rows = c.execute(f'SELECT * FROM {images_table}').fetchall()
faces_rows = c.execute(f'SELECT * FROM {faces_table}').fetchall()
# print(images_rows)
# print('\n'.join(map(str, faces_rows[0])))

thumb, emb = faces_rows[0][1:]


print(emb)

# convert back to image
stream = io.BytesIO(thumb)
image = Image.open(stream).convert('RGBA')  # 'RGBA'
stream.close()
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
