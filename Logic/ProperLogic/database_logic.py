# Credits: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import datetime
import logging
import os
import sqlite3
import time
from functools import partial
from itertools import starmap

import torch
from PIL import Image
import io

from database_table_defs import Tables, Columns, ColumnTypes, ColumnDetails, ColumnSchema
from misc_helpers import is_instance_by_type_name, log_error

# TODO: Foreign keys despite separate db files? --> Implement manually? Needed?
# TODO: (When to) use VACUUM?
# TODO: Locking db necessary?
# TODO: Backup necessary?
# TODO: Use SQLAlchemy?

# TODO: FK faces -> embeddings other way around? Or remove completely?
# TODO: Consistent interface! When to pass objects (tables, columns), when to pass only their names??

# TODO: When to close connection? Optimize?
# TODO: Guarantee that connection is closed at end of methods (done?)
#       ---> Use wrapper/decorator including con as context manager!

# TODO: Make DBManager a singleton object?

"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT embedding_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---

embeddings(INT cluster_id, INT embedding_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


# TODO: Make all methods class/static
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
        # TODO: Needed?
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
    def _create_temp_table(cls, cur, temp_table=None):
        # TODO: Create table in memory? (sqlite3.connect(":memory:"))
        #       ---> Not possible, since other stuff isn't in memory(?)
        if temp_table is None:
            temp_table = Tables.temp_cluster_ids_table
        create_table_sql = cls.build_create_table_sql(temp_table, create_temp=True)
        cur.execute(create_table_sql)

    @classmethod
    def _create_local_tables(cls, cur):
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON;')

        # create images table
        cur.execute(cls.build_create_table_sql(Tables.images_table))

    @staticmethod
    def build_create_table_sql(table, create_temp=False):
        temp_clause = 'TEMP' if create_temp else ''
        constraints_sql = ", ".join(table.constraints)
        creating_sql = (
            f"CREATE {temp_clause} TABLE IF NOT EXISTS {table} ("
            + ", ".join(f"{col} {col.col_type.value} {col.col_constraint}" for col in table.get_columns())
            + (", " if constraints_sql else "")
            + constraints_sql
            + ");"
        )
        return creating_sql

    @staticmethod
    def build_on_conflict_sql(update_cols, update_exprs, conflict_target_cols=None, add_noop_where=False):
        if conflict_target_cols is None:
            conflict_target_cols = []
        noop_where = 'WHERE true' if add_noop_where else ''
        conflict_target = f"({', '.join(map(str, conflict_target_cols))})"
        update_targets = (f'{update_col} = {update_expr}'
                          for update_col, update_expr in zip(update_cols, update_exprs))
        update_clause = ', '.join(update_targets)
        on_conflict = f'{noop_where} ON CONFLICT {conflict_target} DO UPDATE SET {update_clause}'
        return on_conflict

    @classmethod
    def _create_central_tables(cls, cur):
        # TODO: Create third table!
        # enable foreign keys
        cur.execute('PRAGMA foreign_keys = ON;')

        # create embeddings table
        cur.execute(cls.build_create_table_sql(Tables.embeddings_table))

        # create cluster attributes table
        cur.execute(cls.build_create_table_sql(Tables.cluster_attributes_table))

    @staticmethod
    def _drop_tables(cur, drop_local=True):
        tables = Tables.local_tables if drop_local else Tables.central_tables
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS {table};')

    def store_in_table(self, table, row_dicts, on_conflict='', cur=None, path_to_local_db=None, close_connection=True):
        """

        :param table:
        :param row_dicts: iterable of dicts storing (col_name, col_value)-pairs
        :param on_conflict:
        :param path_to_local_db:
        :return:
        """
        rows = self.row_dicts_to_rows(table, row_dicts)
        if not rows:
            return
        store_in_local = Tables.is_local_table(table)
        values_template = self._make_values_template(len(row_dicts[0]))

        if cur is None:
            cur = self.open_connection(store_in_local, path_to_local_db)

        def execute():
            cur.executemany(f'INSERT INTO {table} VALUES ({values_template}) {on_conflict};', rows)

        if not close_connection:
            execute()
            return
        try:
            execute()
        finally:
            self.commit_and_close_connection(store_in_local)

    def store_clusters(self, clusters, emb_id_to_face_dict, emb_id_to_img_id_dict):
        """
        Store the data in clusters in the central DB-tables ('cluster_attributes' and 'embeddings').

        :param clusters: Iterable of clusters to store.
        :return: None
        """
        # TODO: Default argument / other name for param?
        # Store in cluster_attributes table
        # Use on conflict clause for when cluster label and/or center change
        attrs_update_cols = [Columns.label, Columns.center]
        attrs_update_exprs = [f'excluded.{Columns.label}', f'excluded.{Columns.center}']
        attrs_on_conflict = self.build_on_conflict_sql(conflict_target_cols=[Columns.cluster_id],
                                                       update_cols=attrs_update_cols,
                                                       update_exprs=attrs_update_exprs)
        attributes_row_dicts = self.make_attr_row_dicts(clusters)
        self.store_in_table(Tables.cluster_attributes_table, attributes_row_dicts, on_conflict=attrs_on_conflict)

        # Store in embeddings table
        # Use on conflict clause for when cluster id changes
        embs_on_conflict = self.build_on_conflict_sql(conflict_target_cols=[Columns.embedding_id],
                                                      update_cols=[Columns.cluster_id],
                                                      update_exprs=[f'excluded.{Columns.cluster_id}'])
        embeddings_row_dicts = self.make_embs_row_dicts(clusters, emb_id_to_face_dict, emb_id_to_img_id_dict)
        self.store_in_table(Tables.embeddings_table, embeddings_row_dicts, on_conflict=embs_on_conflict)

    def remove_clusters(self, clusters_to_remove):
        """
        Removes the data in clusters from the central DB-tables ('cluster_attributes' and 'embeddings').

        :param clusters_to_remove: Iterable of clusters to remove.
        :return: None
        """
        temp_table = Tables.temp_cluster_ids_table
        embs_table = Tables.embeddings_table
        attrs_table = Tables.cluster_attributes_table

        embs_condition = f'{embs_table}.{Columns.cluster_id} IN {temp_table}'
        attrs_condition = f'{attrs_table}.{Columns.cluster_id} IN {temp_table}'

        cluster_ids_to_remove = map(lambda c: c.cluster_id, clusters_to_remove)
        rows_dicts = [{Columns.cluster_id.col_name: cluster_id}
                      for cluster_id in cluster_ids_to_remove]

        is_local_table = Tables.is_local_table(temp_table)
        close_connection = False
        cur = self.open_connection(open_local=is_local_table)
        try:
            self._create_temp_table(cur, temp_table)
            self.store_in_table(temp_table, rows_dicts, cur=cur, close_connection=close_connection)
            self.delete_from_table(embs_table, condition=embs_condition, cur=cur, close_connection=close_connection)
            self.delete_from_table(attrs_table, condition=attrs_condition, cur=cur, close_connection=close_connection)
        finally:
            self.commit_and_close_connection(close_local=is_local_table)

    def delete_from_table(self, table, with_clause_part='', condition='', cur=None, path_to_local_db=None,
                          close_connection=True):
        """

        :param table:
        :param with_clause_part:
        :param condition:
        :param path_to_local_db:
        :return:
        """
        delete_from_local = Tables.is_local_table(table)
        with_clause = f'WITH {with_clause_part}' if with_clause_part else ''
        where_clause = f'WHERE {condition}' if condition else ''
        if cur is None:
            cur = self.open_connection(delete_from_local, path_to_local_db)

        def execute():
            cur.execute(f'{with_clause} DELETE FROM {table} {where_clause};')

        if not close_connection:
            execute()
            return
        try:
            execute()
        finally:
            self.commit_and_close_connection(delete_from_local)

    @staticmethod
    def make_attr_row_dicts(clusters):
        attributes_row_dicts = [
            {
                Columns.cluster_id.col_name: cluster.cluster_id,
                Columns.label.col_name: cluster.label,
                Columns.center.col_name: cluster.center_point,
            }
            for cluster in clusters
        ]
        return attributes_row_dicts

    @staticmethod
    def make_embs_row_dicts(clusters, emb_id_to_face_dict, emb_id_to_img_id_dict):
        embeddings_row_dicts = []
        for cluster in clusters:
            embeddings_row = [
                {
                    Columns.cluster_id.col_name: cluster.cluster_id,
                    Columns.embedding.col_name: embedding,
                    Columns.thumbnail.col_name: emb_id_to_face_dict[face_id],
                    Columns.image_id.col_name: emb_id_to_img_id_dict[face_id],
                    Columns.embedding_id.col_name: face_id,
                }
                for face_id, embedding in cluster.get_embeddings(with_embedding_ids=True)
            ]

            # embeddings_row = [
            #     {Columns.cluster_id.col_name: cluster.cluster_id,
            #      Columns.embedding.col_name: embedding,
            #      Columns.embedding_id.col_name: face_id}
            #     for face_id, embedding in cluster.get_embeddings(with_embedding_ids=True)
            # ]

            embeddings_row_dicts.extend(embeddings_row)
        return embeddings_row_dicts

    def fetch_from_table(self, table, path_to_local_db=None, col_names=None, cond=''):
        """

        :param table:
        :param path_to_local_db:
        :param col_names: An iterable of column names, or an iterable containing only the string '*' (default).
        :param cond:
        :return:
        """
        # TODO: allow for multiple conditions(?)
        # TODO: Refactor?
        # TODO: More elegant solution?
        if col_names is None or '*' in col_names:
            col_names = table.get_column_names()
        cond_str = '' if len(cond) == 0 else f'WHERE {cond}'
        fetch_from_local = Tables.is_local_table(table)
        cols_template = ','.join(col_names)
        cur = self.open_connection(fetch_from_local, path_to_local_db)
        try:
            rows = cur.execute(f'SELECT {cols_template} FROM {table} {cond_str};').fetchall()
        finally:
            self.commit_and_close_connection(fetch_from_local)

        # cast row of query results to row of usable data
        for row in rows:
            processed_row = starmap(self.sql_value_to_data, zip(row, col_names))
            yield tuple(processed_row)

    def get_cluster_parts(self):
        # TODO: Refactor
        cur = self.open_connection(open_local=False)
        try:
            cluster_parts = cur.execute(
                f"SELECT {Columns.cluster_id}, {Columns.label},"
                f" {Columns.center}, {Columns.embedding}, "
                f"{Columns.embedding_id}"
                f" FROM {Tables.embeddings_table} INNER JOIN {Tables.cluster_attributes_table}"
                f" USING ({Columns.cluster_id});"
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
        # TODO: implement + change signature + use
        pass

    def remove_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: implement + change signature + use
        pass

    def check_crossdb_foreign_key(self, child_table, fk, parent_table, candidate_key):
        # TODO: implement + change signature + use
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

    def get_max_cluster_id(self):
        return self.get_max_num(table=Tables.cluster_attributes_table, col=Columns.cluster_id)

    def get_max_embedding_id(self):
        return self.get_max_num(table=Tables.embeddings_table, col=Columns.embedding_id)

    def get_max_image_id(self, path_to_local_db):
        return self.get_max_num(table=Tables.images_table, col=Columns.image_id, path_to_local_db=path_to_local_db)

    def get_imgs_attrs(self, path_to_local_db=None):
        # TODO: Rename to sth more descriptive!
        col_names = [Columns.file_name.col_name, Columns.last_modified.col_name]
        rows = self.fetch_from_table(Tables.images_table, path_to_local_db=path_to_local_db,
                                     col_names=col_names)
        return rows

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

    @classmethod
    def sql_value_to_data(cls, value, column):
        """

        :param value:
        :param column: str or ColumnSchema
        :return:
        """
        if isinstance(column, str):
            column = Columns.get_column(column)
        elif not is_instance_by_type_name(column, ColumnSchema):
            raise TypeError(f"'column' must be a string or ColumnSchema, not '{type(column)}'.")
        col_details = column.col_details
        if col_details == ColumnDetails.image:
            value = cls.bytes_to_image(value)
        elif col_details == ColumnDetails.tensor:
            value = cls.bytes_to_tensor(value)
        elif col_details == ColumnDetails.date:
            value = cls.iso_string_to_date(value)
        # logging.info("sql_value_to_data: Didn't match any ColumnDetails")
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
        # TODO: ONLY use in generators/DBs on disk with images, otherwise possibly way too much use
        buffer = io.BytesIO(data_bytes)
        try:
            if data_type == ColumnDetails.tensor:
                obj = torch.load(buffer)
            elif data_type == ColumnDetails.image:
                obj = Image.open(buffer).convert('RGBA')
            else:
                raise ValueError(f"Unknown data type '{data_type}', expected '{ColumnDetails.tensor}'"
                                 f" or '{ColumnDetails.image}'.")
        finally:
            buffer.close()
        return obj

    @staticmethod
    def date_to_iso_string(date):
        return date.isoformat().replace('T', ' ')

    @staticmethod
    def iso_string_to_date(string):
        return datetime.datetime.fromisoformat(string)


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
