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

conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

c.execute(f"INSERT OR IGNORE INTO {table_name}({id_column},{new_column1}, {new_column2}) "
          "VALUES (123456, 'testyyy', 'bestyyy')")

c.execute(f"UPDATE {table_name} SET {new_column1}=('Hey Baybey') WHERE {id_column}=(123456)")

conn.commit()
conn.close()

