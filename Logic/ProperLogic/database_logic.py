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
            con = self.local_db_connection
        else:
            self.central_db_connection = sqlite3.connect(DBManager.central_db_file_path)
            con = self.central_db_connection
        return con

    # TODO: Needed when con can just be used as context manager?
    # def commit_and_close_connection(self, close_local):
    #     db_connection = self.local_db_connection if close_local else self.central_db_connection
    #     db_connection.commit()
    #     db_connection.close()

    # TODO: Needed when con can just be used as context manager?
    # def rollback(self, rollback_local):
    #     db_connection = self.local_db_connection if rollback_local else self.central_db_connection
    #     db_connection.rollback()

    def create_tables(self, create_local, path_to_local_db=None, drop_existing_tables=False):
        def create_tables_body(con):
            if drop_existing_tables:
                self._drop_tables(con, create_local)

            if create_local:
                self._create_local_tables(con)
            else:
                self._create_central_tables(con)
        self.connection_wrapper(create_tables_body, create_local, path_to_local_db)

    def connection_wrapper(self, func, open_local=None, path_to_local_db=None, con=None):
        # TODO: Give option of not closing connection (?)
        # TODO: How to make this a decorator?
        # TODO: Make sure callers catch the exceptions!
        if con is None:
            con = self.open_connection(open_local, path_to_local_db)
        try:
            with con:
                result = func(con)
        except sqlite3.DatabaseError as e:
            log_error(f'error in {e.__traceback__.tb_frame}: {e}')
            raise
        finally:
            con.close()
        return result

    @classmethod
    def _create_temp_table(cls, con, temp_table=None):
        # TODO: Create table in memory? (sqlite3.connect(":memory:"))
        #       ---> Not possible, since other stuff isn't in memory(?)
        if temp_table is None:
            temp_table = Tables.temp_cluster_ids_table
        create_table_sql = cls.build_create_table_sql(temp_table, create_temp=True)
        con.execute(create_table_sql)

    @classmethod
    def _create_local_tables(cls, con):
        # enable foreign keys
        con.execute('PRAGMA foreign_keys = ON;')

        # create images table
        con.execute(cls.build_create_table_sql(Tables.images_table))

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
    def _create_central_tables(cls, con):
        # TODO: Create third table!
        # enable foreign keys
        con.execute('PRAGMA foreign_keys = ON;')

        # create embeddings table
        con.execute(cls.build_create_table_sql(Tables.embeddings_table))

        # create cluster attributes table
        con.execute(cls.build_create_table_sql(Tables.cluster_attributes_table))

    @staticmethod
    def _drop_tables(con, drop_local=True):
        tables = Tables.local_tables if drop_local else Tables.central_tables
        for table in tables:
            con.execute(f'DROP TABLE IF EXISTS {table};')

    def store_in_table(self, table, row_dicts, on_conflict='', con=None, path_to_local_db=None, close_connection=True):
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

        def execute(con):
            con.executemany(f'INSERT INTO {table} VALUES ({values_template}) {on_conflict};', rows)

        if close_connection:
            self.connection_wrapper(execute, store_in_local, path_to_local_db)
        else:
            if con is None:
                con = self.open_connection(store_in_local, path_to_local_db)
            execute(con)

    def store_clusters(self, clusters, emb_id_to_face_dict=None, emb_id_to_img_id_dict=None, con=None):
        """
        Store the data in clusters in the central DB-tables ('cluster_attributes' and 'embeddings').

        :param clusters: Iterable of clusters to store.
        :return: None
        """
        # TODO: Default argument / other name for param?
        # TODO: Add parameter whether clusters should be stored even if that would overwrite existing clusters
        # TODO: Improve efficiency - don't build rows etc. if cluster already exists

        if emb_id_to_face_dict is None:
            emb_id_to_face_dict = self.get_thumbnails(with_embeddings_ids=True, as_dict=True)
        if emb_id_to_img_id_dict is None:
            emb_id_to_img_id_dict = self.get_image_ids(with_embeddings_ids=True, as_dict=True)

        # Store in cluster_attributes and embeddings tables
        # Use on conflict clause for when cluster label and/or center change
        attrs_update_cols = [Columns.label, Columns.center]
        attrs_update_exprs = [f'excluded.{Columns.label}', f'excluded.{Columns.center}']
        attrs_on_conflict = self.build_on_conflict_sql(conflict_target_cols=[Columns.cluster_id],
                                                       update_cols=attrs_update_cols,
                                                       update_exprs=attrs_update_exprs)

        # Use on conflict clause for when cluster id changes
        embs_on_conflict = self.build_on_conflict_sql(conflict_target_cols=[Columns.embedding_id],
                                                      update_cols=[Columns.cluster_id],
                                                      update_exprs=[f'excluded.{Columns.cluster_id}'])

        attributes_row_dicts = self.make_attr_row_dicts(clusters)
        embeddings_row_dicts = self.make_embs_row_dicts(clusters, emb_id_to_face_dict, emb_id_to_img_id_dict)

        def store_in_tables(con):
            self.store_in_table(Tables.cluster_attributes_table, attributes_row_dicts, on_conflict=attrs_on_conflict,
                                con=con)
            self.store_in_table(Tables.embeddings_table, embeddings_row_dicts, on_conflict=embs_on_conflict,
                                con=con)

        self.connection_wrapper(store_in_tables, open_local=False, con=con)

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

        def remove_clusters_worker(con):
            self._create_temp_table(con, temp_table)
            self.store_in_table(temp_table, rows_dicts, con=con, close_connection=False)
            self.delete_from_table(embs_table, condition=embs_condition, con=con, close_connection=False)
            self.delete_from_table(attrs_table, condition=attrs_condition, con=con, close_connection=False)

        self.connection_wrapper(remove_clusters_worker, open_local=False)

    def delete_from_table(self, table, with_clause_part='', condition='', con=None, path_to_local_db=None,
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

        def execute(con):
            con.execute(f'{with_clause} DELETE FROM {table} {where_clause};')

        if not close_connection:
            if con is None:
                con = self.open_connection(delete_from_local, path_to_local_db)
            execute(con)
            return

        self.connection_wrapper(execute, delete_from_local, path_to_local_db)

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

        def fetch_worker(con):
            rows = con.execute(f'SELECT {cols_template} FROM {table} {cond_str};').fetchall()
            return rows

        rows = self.connection_wrapper(fetch_worker, fetch_from_local, path_to_local_db)

        # cast row of query results to row of usable data
        for row in rows:
            processed_row = starmap(self.sql_value_to_data, zip(row, col_names))
            yield tuple(processed_row)

    def get_clusters_parts(self):
        # TODO: Refactor + improve efficiency (don't let attributes of same cluster be processed multiple times)

        def get_parts_worker(con):
            clusters_parts_list = con.execute(
                f"SELECT {Columns.cluster_id}, {Columns.label}, {Columns.center}"
                f" FROM {Tables.cluster_attributes_table};"
            ).fetchall()

            embeddings_parts_list = con.execute(
                f"SELECT {Columns.cluster_id}, {Columns.embedding}, {Columns.embedding_id}"
                f" FROM {Tables.embeddings_table};"
            ).fetchall()
            return clusters_parts_list, embeddings_parts_list

        parts_lists = self.connection_wrapper(get_parts_worker, open_local=False)
        clusters_parts_list, embeddings_parts_list = parts_lists

        # TODO: Refactor(?)
        # convert center point and embedding to tensors rather than bytes
        proc_clusters_parts_list = [  # processed clusters
            (cluster_id, label, self.bytes_to_tensor(center_point))
            for cluster_id, label, center_point in clusters_parts_list
        ]

        proc_embeddings_parts_list = [
            (cluster_id, self.bytes_to_tensor(embedding), embedding_id)
            for cluster_id, embedding, embedding_id in embeddings_parts_list
        ]

        return proc_clusters_parts_list, proc_embeddings_parts_list

    def get_thumbnails_from_cluster(self, cluster_id, with_embedding_ids=False, as_dict=True):
        return self.get_thumbnails(with_embedding_ids, as_dict, cond=f'cluster_id = {cluster_id}')

    def get_thumbnails(self, with_embeddings_ids=False, as_dict=True, cond=''):
        return self.get_column(Columns.thumbnail, Tables.embeddings_table, with_embeddings_ids, as_dict, cond)

    def get_image_ids(self, with_embeddings_ids=False, as_dict=True):
        return self.get_column(Columns.image_id, Tables.embeddings_table, with_embeddings_ids, as_dict)

    def get_column(self, col, table, with_embeddings_ids=False, as_dict=True, cond=''):
        col_names = [Columns.embedding_id.col_name] if with_embeddings_ids else []
        col_names.append(col.col_name)
        query_results = self.fetch_from_table(table, col_names=col_names, cond=cond)
        if with_embeddings_ids and as_dict:
            return dict(query_results)
        return query_results

    def aggregate_col(self, table, col, func, path_to_local_db=None):
        aggregate_from_local = Tables.is_local_table(table)

        def aggregate_worker(con):
            agg_value = con.execute(
                f"SELECT {func}({col}) FROM {table};"
            ).fetchone()
            return agg_value

        agg_value = self.connection_wrapper(aggregate_worker, aggregate_from_local, path_to_local_db)
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
        data_bytes = None
        buffer = io.BytesIO()
        try:
            if isinstance(data, torch.Tensor):  # case 1: embedding
                torch.save(data, buffer)
            else:  # case 2: thumbnail
                data.save(buffer, format='JPEG')
            data_bytes = buffer.getvalue()
        finally:
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
                # TODO: More efficient way to provide access to these images long-term?
                obj = Image.open(buffer).convert('RGB')  # Conversion also copies the object
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
