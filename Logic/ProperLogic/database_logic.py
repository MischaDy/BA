# Credits: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
import datetime
import os
import sqlite3
import sys
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


"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT embedding_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---

embeddings(INT cluster_id, INT embedding_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


# TODO: Allow instances, which have a 'current connection' as only instance attribute?
class DBManager:
    db_files_path = 'database'
    central_db_file_name = 'central_db.sqlite'
    central_db_file_path = os.path.join(db_files_path, central_db_file_name)
    local_db_file_name = 'local_db.sqlite'

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def open_connection(cls, open_local, path_to_local_db=None):
        path = path_to_local_db if open_local else cls.central_db_file_path
        return sqlite3.connect(path)

    @classmethod
    def create_tables(cls, create_local, path_to_local_db=None, drop_existing_tables=False):
        def create_tables_body(con):
            if drop_existing_tables:
                cls.drop_tables(drop_local=create_local, con=con, close_connection=False)

            if create_local:
                cls.create_local_tables(con, close_connection=False)
            else:
                cls.create_central_tables(con, close_connection=False)
        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_tables_body, create_local, path_to_local_db)

    @classmethod
    def connection_wrapper(cls, func, open_local=None, path_to_local_db=None, con=None, close_connection=True):
        # TODO: How to make this a decorator?
        # TODO: Make sure callers undo their tasks if exception is raised!
        if con is None:
            con = cls.open_connection(open_local, path_to_local_db)
        try:
            with con:
                result = func(con)
        except Exception as e:
            log_error(f'{e.__class__}, {e.args}')
            tb = sys.exc_info()[2]
            raise IncompleteDatabaseOperation(e).with_traceback(tb)
        finally:
            if close_connection:
                con.close()
        return result

    @classmethod
    def create_temp_table(cls, con, temp_table=None):
        # TODO: Create table in memory? (sqlite3.connect(":memory:"))
        #       ---> Not possible, since other stuff isn't in memory(?)

        def create_temp_table_worker(con):
            if temp_table is None:
                local_temp_table = Tables.temp_cluster_ids_table
            else:
                local_temp_table = temp_table
            create_table_sql = cls.build_create_table_sql(local_temp_table, create_temp=True)
            con.execute(create_table_sql)
        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_temp_table_worker, con=con, close_connection=False)

    @classmethod
    def create_local_tables(cls, con, close_connection=True):
        # TODO: default param for close_connection as False?
        def create_images_table(con):
            con.execute('PRAGMA foreign_keys = ON;')
            con.execute(cls.build_create_table_sql(Tables.images_table))
        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_images_table, con=con, close_connection=close_connection)

    @classmethod
    def create_central_tables(cls, con, close_connection=True):
        def create_central_tables_worker(con):
            con.execute('PRAGMA foreign_keys = ON;')
            # create embeddings table
            con.execute(cls.build_create_table_sql(Tables.embeddings_table))
            # create cluster attributes table
            con.execute(cls.build_create_table_sql(Tables.cluster_attributes_table))
        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_central_tables_worker, con=con, close_connection=close_connection)

    @classmethod
    def drop_tables(cls, drop_local, path_to_local_db=None, con=None, close_connection=True):
        def drop_tables_worker(con):
            tables = Tables.local_tables if drop_local else Tables.central_tables
            # TODO: Use executemany?
            for table in tables:
                con.execute(f'DROP TABLE IF EXISTS {table};')

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(drop_tables_worker, path_to_local_db=path_to_local_db, con=con,
                               close_connection=close_connection)

    @classmethod
    def store_in_table(cls, table, row_dicts, on_conflict='', con=None, path_to_local_db=None, close_connection=True):
        """

        :param con:
        :param close_connection:
        :param table:
        :param row_dicts: iterable of dicts storing (col_name, col_value)-pairs
        :param on_conflict:
        :param path_to_local_db:
        :return:
        """
        rows = cls.row_dicts_to_rows(table, row_dicts)
        if not rows:
            return
        store_in_local = Tables.is_local_table(table)
        values_template = cls.make_values_template(len(row_dicts[0]))

        def execute(con):
            con.executemany(f'INSERT INTO {table} VALUES ({values_template}) {on_conflict};', rows)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(execute, store_in_local, path_to_local_db, con=con, close_connection=close_connection)

    @classmethod
    def store_clusters(cls, clusters, emb_id_to_face_dict=None, emb_id_to_img_id_dict=None, con=None,
                       close_connection=True):
        """
        Store the data in clusters in the central DB-tables ('cluster_attributes' and 'embeddings').

        :param close_connection:
        :param con:
        :param emb_id_to_img_id_dict:
        :param emb_id_to_face_dict:
        :param clusters: Iterable of clusters to store.
        :return: None
        """
        # TODO: Default argument / other name for param?
        # TODO: Add parameter whether clusters should be stored even if that would overwrite existing clusters
        # TODO: Improve efficiency - don't build rows etc. if cluster already exists

        if emb_id_to_face_dict is None:
            emb_id_to_face_dict = cls.get_thumbnails(with_embeddings_ids=True, as_dict=True)
        if emb_id_to_img_id_dict is None:
            emb_id_to_img_id_dict = cls.get_image_ids(with_embeddings_ids=True, as_dict=True)

        # Store in cluster_attributes and embeddings tables
        # Use on conflict clause for when cluster label and/or center change
        attrs_update_cols = [Columns.label, Columns.center]
        attrs_update_expressions = [f'excluded.{Columns.label}', f'excluded.{Columns.center}']
        attrs_on_conflict = cls.build_on_conflict_sql(conflict_target_cols=[Columns.cluster_id],
                                                      update_cols=attrs_update_cols,
                                                      update_expressions=attrs_update_expressions)

        # Use on conflict clause for when cluster id changes
        embs_on_conflict = cls.build_on_conflict_sql(conflict_target_cols=[Columns.embedding_id],
                                                     update_cols=[Columns.cluster_id],
                                                     update_expressions=[f'excluded.{Columns.cluster_id}'])

        attributes_row_dicts = cls.make_attr_row_dicts(clusters)
        embeddings_row_dicts = cls.make_embs_row_dicts(clusters, emb_id_to_face_dict, emb_id_to_img_id_dict)

        def store_in_tables(con):
            cls.store_in_table(Tables.cluster_attributes_table, attributes_row_dicts, on_conflict=attrs_on_conflict,
                               con=con, close_connection=False)
            cls.store_in_table(Tables.embeddings_table, embeddings_row_dicts, on_conflict=embs_on_conflict,
                               con=con, close_connection=False)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(store_in_tables, open_local=False, con=con, close_connection=close_connection)

    @classmethod
    def remove_clusters(cls, clusters_to_remove, con=None, close_connection=True):
        """
        Removes the data in clusters from the central DB-tables ('cluster_attributes' and 'embeddings').

        :param close_connection:
        :param con:
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
            cls.create_temp_table(con, temp_table)
            cls.store_in_table(temp_table, rows_dicts, con=con, close_connection=False)
            deleted_embeddings_row_dicts = cls.delete_from_table(embs_table, condition=embs_condition, con=con,
                                                                 close_connection=False)
            cls.delete_from_table(attrs_table, condition=attrs_condition, con=con, close_connection=False)
            return deleted_embeddings_row_dicts

        # TODO: How to handle possible exception here?
        deleted_embeddings_row_dicts = cls.connection_wrapper(remove_clusters_worker, open_local=False, con=con,
                                                              close_connection=close_connection)
        return deleted_embeddings_row_dicts

    @classmethod
    def delete_from_table(cls, table, with_clause_part='', condition='', con=None, path_to_local_db=None,
                          close_connection=True):
        """

        :param close_connection:
        :param con:
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
            # TODO: 'Copy' generator instead of cast to list? (Saves space)
            # Cast to list is *necessary* here, since fetch function only returns a generator. It will be executed after
            # the corresponding rows are deleted from table and will thus yield nothing.
            deleted_row_dicts = list(cls.fetch_from_table(table, path_to_local_db, condition=condition, con=con,
                                                          close_connection=False))
            con.execute(f'{with_clause} DELETE FROM {table} {where_clause};')
            return deleted_row_dicts

        # TODO: How to handle possible exception here?
        deleted_row_dicts = cls.connection_wrapper(execute, delete_from_local, path_to_local_db, con=con,
                                                   close_connection=close_connection)
        return deleted_row_dicts

    @classmethod
    def fetch_from_table(cls, table, path_to_local_db=None, col_names=None, condition='', con=None,
                         close_connection=True):
        """

        :param close_connection:
        :param con:
        :param table:
        :param path_to_local_db:
        :param col_names: An iterable of column names, or an iterable containing only the string '*' (default).
        :param condition:
        :return:
        """
        # TODO: allow for multiple conditions(?)
        # TODO: Refactor?
        # TODO: More elegant solution?
        if col_names is None or '*' in col_names:
            col_names = table.get_column_names()
        cond_str = '' if len(condition) == 0 else f'WHERE {condition}'
        fetch_from_local = Tables.is_local_table(table)
        cols_template = ','.join(col_names)

        def fetch_worker(con):
            rows = con.execute(f'SELECT {cols_template} FROM {table} {cond_str};').fetchall()
            return rows

        # TODO: How to handle possible exception here?
        rows = cls.connection_wrapper(fetch_worker, fetch_from_local, path_to_local_db, con=con,
                                      close_connection=close_connection)

        # cast row of query results to row of usable data
        for row in rows:
            processed_row = starmap(cls.sql_value_to_data, zip(row, col_names))
            yield tuple(processed_row)

    @classmethod
    def get_clusters_parts(cls):
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

        # TODO: How to handle possible exception here?
        parts_lists = cls.connection_wrapper(get_parts_worker, open_local=False)
        clusters_parts_list, embeddings_parts_list = parts_lists

        # TODO: Refactor(?)
        # convert center point and embedding to tensors rather than bytes
        proc_clusters_parts_list = [  # processed clusters
            (cluster_id, label, cls.bytes_to_tensor(center_point))
            for cluster_id, label, center_point in clusters_parts_list
        ]

        proc_embeddings_parts_list = [
            (cluster_id, cls.bytes_to_tensor(embedding), embedding_id)
            for cluster_id, embedding, embedding_id in embeddings_parts_list
        ]

        return proc_clusters_parts_list, proc_embeddings_parts_list

    @classmethod
    def get_thumbnails_from_cluster(cls, cluster_id, with_embedding_ids=False, as_dict=True):
        return cls.get_thumbnails(with_embedding_ids, as_dict, cond=f'cluster_id = {cluster_id}')

    @classmethod
    def get_thumbnails(cls, with_embeddings_ids=False, as_dict=True, cond='', con=None, close_connection=True):
        thumbnails = cls.get_column(Columns.thumbnail, Tables.embeddings_table, with_embeddings_ids, as_dict, cond,
                                    con=con, close_connection=close_connection)
        return thumbnails

    @classmethod
    def get_image_ids(cls, with_embeddings_ids=False, as_dict=True, con=None, close_connection=True):
        image_ids = cls.get_column(Columns.image_id, Tables.embeddings_table, with_embeddings_ids, as_dict, con=con,
                                   close_connection=close_connection)
        return image_ids

    @classmethod
    def get_column(cls, col, table, with_embeddings_ids=False, as_dict=True, cond='', con=None, close_connection=True):
        col_names = [Columns.embedding_id.col_name] if with_embeddings_ids else []
        col_names.append(col.col_name)
        query_results = cls.fetch_from_table(table, col_names=col_names, condition=cond, con=con,
                                             close_connection=close_connection)
        if with_embeddings_ids and as_dict:
            return dict(query_results)
        return query_results

    @classmethod
    def aggregate_col(cls, table, col, func, path_to_local_db=None, con=None, close_connection=True):
        aggregate_from_local = Tables.is_local_table(table)

        def aggregate_worker(con):
            agg_value = con.execute(
                f"SELECT {func}({col}) FROM {table};"
            ).fetchone()
            return agg_value

        # TODO: How to handle possible exception here?
        agg_value = cls.connection_wrapper(aggregate_worker, aggregate_from_local, path_to_local_db,
                                           con=con, close_connection=close_connection)
        return agg_value

    @classmethod
    def get_max_num(cls, table, col, default=0, path_to_local_db=None, con=None, close_connection=True):
        max_num_rows = cls.aggregate_col(table=table, col=col, func='MAX', path_to_local_db=path_to_local_db, con=con,
                                         close_connection=close_connection)
        max_num = max_num_rows[0]
        if isinstance(max_num, int) or isinstance(max_num, float):
            return max_num
        return default

    @classmethod
    def get_max_cluster_id(cls):
        max_cluster_id = cls.get_max_num(table=Tables.cluster_attributes_table, col=Columns.cluster_id)
        return max_cluster_id

    @classmethod
    def get_max_embedding_id(cls):
        max_embedding_id = cls.get_max_num(table=Tables.embeddings_table, col=Columns.embedding_id)
        return max_embedding_id

    @classmethod
    def get_max_image_id(cls, path_to_local_db):
        max_image_id = cls.get_max_num(table=Tables.images_table, col=Columns.image_id,
                                       path_to_local_db=path_to_local_db)
        return max_image_id

    @classmethod
    def get_images_attributes(cls, path_to_local_db=None, con=None, close_connection=True):
        col_names = [Columns.file_name.col_name, Columns.last_modified.col_name]
        rows = cls.fetch_from_table(Tables.images_table, path_to_local_db=path_to_local_db,
                                    col_names=col_names, con=con, close_connection=close_connection)
        return rows

    @classmethod
    def get_db_path(cls, path, local=True):
        if local:
            return os.path.join(path, cls.local_db_file_name)
        return cls.central_db_file_name

    @classmethod
    def row_dicts_to_rows(cls, table, row_dicts):
        # TODO: Improve efficiency?
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
        return value

    @classmethod
    def bytes_to_image(cls, data_bytes):
        return cls.bytes_to_data(data_bytes, data_type=ColumnDetails.image)

    @classmethod
    def bytes_to_tensor(cls, data_bytes):
        return cls.bytes_to_data(data_bytes, data_type=ColumnDetails.tensor)

    @staticmethod
    def make_values_template(length, char_to_join='?', sep=','):
        chars_to_join = length * char_to_join if len(char_to_join) == 1 else char_to_join
        return sep.join(chars_to_join)

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
    def build_on_conflict_sql(update_cols, update_expressions, conflict_target_cols=None, add_noop_where=False):
        if conflict_target_cols is None:
            conflict_target_cols = []
        noop_where = 'WHERE true' if add_noop_where else ''
        conflict_target = f"({', '.join(map(str, conflict_target_cols))})"
        update_targets = (f'{update_col} = {update_expr}'
                          for update_col, update_expr in zip(update_cols, update_expressions))
        update_clause = ', '.join(update_targets)
        on_conflict = f'{noop_where} ON CONFLICT {conflict_target} DO UPDATE SET {update_clause}'
        return on_conflict

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

    @staticmethod
    def date_to_iso_string(date):
        return date.isoformat().replace('T', ' ')

    @staticmethod
    def iso_string_to_date(string):
        return datetime.datetime.fromisoformat(string)


class IncompleteDatabaseOperation(RuntimeError):
    pass
