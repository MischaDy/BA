# Credit: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html


import sqlite3

sqlite_file = 'my_db.sqlite'  # name of the sqlite database file
table_name = 'my_table_2'  # name of the table to be created
id_column = 'my_1st_column'  # name of the PRIMARY KEY column
id_column_type = 'INTEGER'
new_column1 = 'my_2nd_column'  # name of the new column
new_column2 = 'my_3rd_column'  # name of the new column
column_type = 'TEXT'  # E.g., INTEGER, TEXT, NULL, REAL, BLOB
default_val = 'Hello World'  # a default value for the new column rows

# time.time() - os.stat('my_db.sqlite').st_atime

# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Retrieve column information
# Every column will be represented by a tuple with the following attributes:
# (id, name, type, notnull, default_value, primary_key)
c.execute('PRAGMA TABLE_INFO({})'.format(table_name))

# collect names in a list
names = [tup[1] for tup in c.fetchall()]
print(names)
# e.g., ['id', 'date', 'time', 'date_time']

# Closing the connection to the database file
conn.close()
