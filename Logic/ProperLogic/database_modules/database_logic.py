# Credits: https://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html

import datetime
import os
import sqlite3
import sys
from functools import partial
from itertools import starmap, repeat

import torch
from PIL import Image
import io

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns, ColumnTypes, ColumnDetails, \
    ColumnSchema
from Logic.ProperLogic.misc_helpers import is_instance_by_type_name, log_error, get_every_nth_item

"""
----- DB SCHEMA -----


--- Local Tables ---

images(INT image_id, TEXT file_name, INT last_modified)
faces(INT embedding_id, INT image_id, BLOB thumbnail)

--- Centralized Tables ---

embeddings(INT cluster_id, INT embedding_id, BLOB face_embedding)
cluster_attributes(INT cluster_id, TEXT label, BLOB center)
"""


# TODO: Create iterator which, starting from current cluster_id, provides them sequentially (disregarding when new ones
#       are stored in the DB


class DBManager:
    __strange_sep = ' || '
    db_files_path = 'database_modules'
    central_db_file_name = 'central_db.sqlite'
    central_db_file_path = os.path.join(db_files_path, central_db_file_name)
    local_db_file_name = 'local_db.sqlite'

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def open_central_connection(cls):
        """
        More intuitive alias for open_connection when called without arguments.

        :return:
        """
        return cls.open_connection()

    @classmethod
    def open_local_connection(cls, path_to_local_db):
        """
        More intuitive alias for open_connection when it is known that local connection is needed.

        :param path_to_local_db:
        :return:
        """
        return cls.open_connection(path_to_local_db)

    @classmethod
    def open_connection(cls, path_to_local_db=None):
        if path_to_local_db is None:
            path = cls.central_db_file_path
        else:
            path = path_to_local_db
        return sqlite3.connect(path)

    @classmethod
    def connection_wrapper(cls, func, path_to_local_db=None, con=None, global_con=None, local_con=None,
                           close_connections=True):
        """
        If con is provided, the other two con params are ignored.

        :param func:
        :param path_to_local_db:
        :param con:
        :param global_con:
        :param local_con:
        :param close_connections:
        :return:
        """
        # TODO: Does everything work even without explicit local/global call?
        # TODO: Allow to not pass global/local con, but still open it here (and thus close it)?
        # TODO: Make sure callers undo their tasks if exception is raised!
        # TODO: Generalize to allow for one global and any number of local connections?
        # TODO: How to make this a decorator?

        if not any([con, global_con, local_con]):
            # no connections provided
            con = cls.open_connection(path_to_local_db)
            close_connections = True

        if con is not None:
            # con provided, other two con params ignored
            connections_dict = {'con': con}
        else:
            # at least one specific con parameter provided
            connections_dict = {}
            if global_con is not None:
                connections_dict['global_con'] = global_con
            if local_con is not None:
                connections_dict['local_con'] = local_con

        connections = connections_dict.values()
        commit_connections = True
        try:
            result = func(**connections_dict)
        except Exception as e:
            commit_connections = False
            for con in connections:
                con.rollback()
            log_error(e)
            tb = sys.exc_info()[2]
            raise IncompleteDatabaseOperation(e).with_traceback(tb)
        finally:
            # TODO: Check if connection is still open before committing + closing?
            if close_connections:
                for con in connections:
                    if commit_connections:
                        con.commit()
                    con.close()
        return result

    @classmethod
    def create_temp_table(cls, con, temp_table):
        # TODO: Create table in memory? (sqlite3.connect(":memory:"))
        #       ---> Not possible, since other stuff isn't in memory(?)
        cls._create_tables([temp_table], create_temp_flags=[True], con=con, close_connections=False)

    @classmethod
    def create_local_tables(cls, drop_existing_tables=False, path_to_local_db=None, con=None, close_connections=True):

        def create_local_tables_worker(con):
            if drop_existing_tables:
                cls.drop_tables(drop_local=True, path_to_local_db=path_to_local_db, con=con, close_connections=False)
            cls._create_tables(Tables.local_tables, fk_on=True, con=con, close_connections=False)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_local_tables_worker, path_to_local_db=path_to_local_db, con=con,
                               close_connections=close_connections)

    @classmethod
    def create_central_tables(cls, drop_existing_tables=False, con=None, close_connections=True):

        def create_central_tables_worker(con):
            if drop_existing_tables:
                cls.drop_tables(drop_local=False, con=con, close_connections=False)
            cls._create_tables(Tables.central_tables, fk_on=True, con=con, close_connections=False)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(create_central_tables_worker, con=con,
                               close_connections=close_connections)

    @classmethod
    def _create_tables(cls, tables, fk_on=False, create_temp_flags=None, con=None, close_connections=True):
        # TODO: Rename?
        if create_temp_flags is None:
            create_temp_flags = repeat(False, len(tables))

        def _create_tables_worker(con):
            if fk_on:
                con.execute('PRAGMA foreign_keys = ON;')
            for create_temp_flag, table in zip(create_temp_flags, tables):
                create_table_sql = cls.build_create_table_sql(table, create_temp=create_temp_flag)
                con.execute(create_table_sql)

        cls.connection_wrapper(_create_tables_worker, con=con, close_connections=close_connections)

    @classmethod
    def drop_tables(cls, drop_local, path_to_local_db=None, con=None, close_connections=True):
        def drop_tables_worker(con):
            tables = Tables.local_tables if drop_local else Tables.central_tables
            # TODO: Use executemany?
            for table in tables:
                con.execute(f'DROP TABLE IF EXISTS {table};')

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(drop_tables_worker, path_to_local_db=path_to_local_db, con=con,
                               close_connections=close_connections)

    @classmethod
    def store_in_table(cls, table, row_dicts, on_conflict='', path_to_local_db=None, con=None, close_connections=True):
        """

        :param con:
        :param close_connections:
        :param table:
        :param row_dicts: iterable of dicts storing (col_name, col_value)-pairs
        :param on_conflict:
        :param path_to_local_db:
        :return:
        """
        rows = cls.row_dicts_to_rows(table, row_dicts)
        if not rows:
            return
        values_template = cls.make_values_template(len(row_dicts[0]))

        def store_in_table_worker(con):
            con.executemany(f'INSERT INTO {table} VALUES ({values_template}) {on_conflict};', rows)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(store_in_table_worker, path_to_local_db, con=con,
                               close_connections=close_connections)

    @classmethod
    def store_clusters(cls, clusters, emb_id_to_face_dict=None, emb_id_to_img_id_dict=None, con=None,
                       close_connections=True):
        """
        Store the data in clusters in the central DB-tables ('cluster_attributes' and 'embeddings').

        :param close_connections:
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

        if is_instance_by_type_name(clusters, ClusterDict):
            clusters = clusters.get_clusters()

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

        def store_clusters_worker(con):
            cls.store_in_table(Tables.cluster_attributes_table, attributes_row_dicts, on_conflict=attrs_on_conflict,
                               con=con, close_connections=False)
            cls.store_in_table(Tables.embeddings_table, embeddings_row_dicts, on_conflict=embs_on_conflict,
                               con=con, close_connections=False)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(store_clusters_worker, con=con, close_connections=close_connections)

    @classmethod
    def store_certain_labels(cls, embeddings_ids=None, label=None, cluster=None, con=None, close_connections=True):
        if cluster is not None:
            embeddings_ids = cluster.get_embeddings_ids()
            label = cluster.label
        elif embeddings_ids is None or label is None:
            raise ValueError('At least one of [embeddings_ids and label] and [cluster] must not be None')

        table = Tables.certain_labels_table
        # row_dicts = table.make_row_dicts(
        #     values_objects=[embeddings_ids, label],
        #     repetition_flags=[False, True]
        # )
        row_dicts = [
            {Columns.embedding_id.col_name: embedding_id,
             Columns.label.col_name: label}
            for embedding_id in embeddings_ids
        ]

        # Use on conflict clause for when user wants to relabel a known image
        labels_on_conflict = cls.build_on_conflict_sql(conflict_target_cols=[Columns.embedding_id],
                                                       update_cols=[Columns.label],
                                                       update_expressions=[f'excluded.{Columns.label}'])

        def store_certain_tables_worker(con):
            cls.store_in_table(table, row_dicts, on_conflict=labels_on_conflict, con=con, close_connections=False)

        # TODO: How to handle possible exception here?
        cls.connection_wrapper(store_certain_tables_worker, con=con, close_connections=close_connections)

    @classmethod
    def store_directory_path(cls, path, path_id=None, con=None, close_connections=True):
        if path_id is None:
            max_path_id = cls.get_max_path_id()
            path_id = max_path_id + 1
        path_row = {
            Columns.path_id_col.col_name: path_id,
            Columns.path.col_name: path
        }

        def store_directory_path_worker(con):
            cls.store_in_table(Tables.directory_paths_table, [path_row], con=con, close_connections=False)

        cls.connection_wrapper(store_directory_path_worker, con=con, close_connections=close_connections)
        return path_id

    @classmethod
    def store_image_path(cls, img_id=None, path_id=None, con=None, close_connections=True):
        # TODO: Allow img_id to be None?
        # if img_id is None:
        #     max_img_id = cls.get_max_image_id(path_to_local_db)
        #     img_id = max_img_id + 1
        if path_id is None:
            max_path_id = cls.get_max_path_id()
            path_id = max_path_id + 1
        img_path_row = {Columns.image_id.col_name: img_id,
                        Columns.path_id_col.col_name: path_id}

        def store_image_path_worker(con):
            cls.store_in_table(Tables.image_paths_table, [img_path_row], con=con, close_connections=False)

        cls.connection_wrapper(store_image_path_worker, con=con, close_connections=close_connections)

    @classmethod
    def store_image(cls, img_id, file_name, last_modified, path_to_local_db, con=None, close_connections=True):
        # TODO: Allow img_id to be None?
        img_row = {Columns.image_id.col_name: img_id,
                   Columns.file_name.col_name: file_name,
                   Columns.last_modified.col_name: last_modified}

        def store_image_worker(con):
            cls.store_in_table(Tables.images_table, [img_row], path_to_local_db=path_to_local_db, con=con,
                               close_connections=False)

        cls.connection_wrapper(store_image_worker, path_to_local_db=path_to_local_db, con=con,
                               close_connections=close_connections)

    @classmethod
    def store_path_id(cls, path_id, path_to_local_db, con=None, close_connections=True):
        # TODO: Allow img_id to be None?
        path_id_row = {Columns.path_id_col.col_name: path_id}

        def store_path_id_worker(con):
            cls.store_in_table(Tables.path_id_table, [path_id_row], path_to_local_db=path_to_local_db, con=con,
                               close_connections=False)

        cls.connection_wrapper(store_path_id_worker, path_to_local_db=path_to_local_db, con=con,
                               close_connections=close_connections)

    @classmethod
    def remove_cluster(cls, cluster_to_remove, con=None, close_connections=True):
        clusters_to_remove = ClusterDict([cluster_to_remove])
        return cls.remove_clusters(clusters_to_remove, con=con, close_connections=close_connections)

    @classmethod
    def remove_clusters(cls, clusters_to_remove=None, remove_all=False, con=None, close_connections=True):
        """
        Removes the data in clusters from the central DB-tables ('cluster_attributes' and 'embeddings').
        Exactly one of *clusters_to_remove* and *remove_all* must be set.

        :param clusters_to_remove: Iterable of clusters to remove.
        :param remove_all: If true, all clusters are removed.
        :param con:
        :param close_connections:
        :return: None
        """
        # TODO: More efficient way of deleting all clusters!
        if clusters_to_remove is None and not remove_all:
            log_error(f"'clusters_to_remove' or 'remove_all' must be provided")
            return
        elif not clusters_to_remove:
            # Iterable has been provided, but is empty. Not an error.
            return
        elif clusters_to_remove and remove_all:
            log_error(f"cannot provide both 'clusters_to_remove' and 'remove_all' (safety feature)")
            return

        if remove_all:
            clusters_to_remove = DBManager.load_cluster_dict()
        elif not is_instance_by_type_name(clusters_to_remove, ClusterDict):
            clusters_to_remove = ClusterDict(clusters_to_remove)

        temp_table = Tables.temp_cluster_ids_table
        embs_table = Tables.embeddings_table
        attrs_table = Tables.cluster_attributes_table

        embs_cond = f'{embs_table}.{Columns.cluster_id} IN {temp_table}'
        attrs_cond = f'{attrs_table}.{Columns.cluster_id} IN {temp_table}'

        cluster_ids_to_remove = clusters_to_remove.get_cluster_ids()
        # TODO: Rename
        rows_dicts = [{Columns.cluster_id.col_name: cluster_id}
                      for cluster_id in cluster_ids_to_remove]

        def remove_clusters_worker(con):
            cls.create_temp_table(con, temp_table)
            cls.store_in_table(temp_table, rows_dicts, con=con, close_connections=False)
            deleted_embeddings_row_dicts = cls.delete_from_table(embs_table, cond=embs_cond, con=con,
                                                                 close_connections=False)
            cls.delete_from_table(attrs_table, cond=attrs_cond, con=con, close_connections=False)
            return deleted_embeddings_row_dicts

        # TODO: How to handle possible exception here?
        deleted_embeddings_row_dicts = cls.connection_wrapper(remove_clusters_worker, con=con,
                                                              close_connections=close_connections)
        return deleted_embeddings_row_dicts

    @classmethod
    def delete_from_table(cls, table, with_clause_part='', cond='', con=None, path_to_local_db=None,
                          close_connections=True):
        """

        :param close_connections:
        :param con:
        :param table:
        :param with_clause_part:
        :param cond:
        :param path_to_local_db:
        :return:
        """
        with_clause = cls._build_with_clause(with_clause_part)
        where_clause = cls._build_where_clause(cond)

        def delete_from_table_worker(con):
            # TODO: 'Copy' generator instead of cast to list? (Saves space)
            # Cast to list is *necessary* here, since fetch function only returns a generator. It will be executed after
            # the corresponding rows are deleted from table and will thus yield nothing.
            deleted_row_dicts = list(
                cls.fetch_from_table(table, path_to_local_db, cond=cond, as_dicts=True, con=con,
                                     close_connections=False)
            )
            con.execute(f'{with_clause} DELETE FROM {table} {where_clause};')
            return deleted_row_dicts

        # TODO: How to handle possible exception here?
        deleted_row_dicts = cls.connection_wrapper(delete_from_table_worker, path_to_local_db, con=con,
                                                   close_connections=close_connections)
        return deleted_row_dicts

    @classmethod
    def fetch_from_table(cls, table, path_to_local_db=None, col_names=None, cond='', cond_params=None, as_dicts=False,
                         con=None, close_connections=True):
        """

        :param cond_params:
        :param as_dicts:
        :param close_connections:
        :param con:
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
        cols_template = ','.join(col_names)
        where_clause = cls._build_where_clause(cond)
        fetch_sql = f'SELECT {cols_template} FROM {table} {where_clause};'

        def fetch_from_table_worker(con):
            if cond_params is None:
                return con.execute(fetch_sql).fetchall()
            return con.execute(fetch_sql, cond_params).fetchall()

        # TODO: How to handle possible exception here?
        rows = cls.connection_wrapper(fetch_from_table_worker, path_to_local_db, con=con,
                                      close_connections=close_connections)

        # cast row of query results to row of usable data
        processed_rows = (starmap(cls.sql_value_to_data, zip(row, col_names))
                          for row in rows)
        output_func = table.row_to_row_dict if as_dicts else tuple
        return map(output_func, processed_rows)

    @classmethod
    def load_cluster_dict(cls):
        # TODO: Refactor + improve efficiency
        cluster_attributes_parts = DBManager.get_cluster_attributes_parts()
        embeddings_parts = DBManager.get_embeddings_parts()

        # clusters_dict = dict(
        #     (kwargs[Columns.cluster_id.col_name], Cluster(**kwargs))
        #     for kwargs in clusters_parts
        # )

        clusters_dict = ClusterDict()
        for cluster_id, label, center in cluster_attributes_parts:
            clusters_dict[cluster_id] = Cluster(cluster_id, label=label, center_point=center)

        for cluster_id, embedding, embedding_id in embeddings_parts:
            cluster = clusters_dict[cluster_id]
            cluster.add_embedding(embedding, embedding_id)

        return clusters_dict

    @classmethod
    def get_embeddings_parts(cls, cond=''):
        # TODO: Add con params?
        where_clause = cls._build_where_clause(cond)

        def get_embeddings_parts_worker(con):
            embeddings_parts = con.execute(
                f"SELECT {Columns.cluster_id}, {Columns.embedding}, {Columns.embedding_id}"
                f" FROM {Tables.embeddings_table}"
                f" {where_clause};"
            ).fetchall()
            return embeddings_parts

        # TODO: How to handle possible exception here?
        embeddings_parts = cls.connection_wrapper(get_embeddings_parts_worker)

        proc_embeddings_parts = [
            (cluster_id, cls.bytes_to_tensor(embedding), embedding_id)
            for cluster_id, embedding, embedding_id in embeddings_parts
        ]
        return proc_embeddings_parts

    @classmethod
    def get_cluster_attributes_parts(cls, cond=''):
        # TODO: Add con params?
        # TODO: Refactor + improve efficiency (don't let attributes of same cluster be processed multiple times)
        where_clause = cls._build_where_clause(cond)

        def get_cluster_attributes_parts_worker(con):
            cluster_attributes_parts = con.execute(
                f"SELECT {Columns.cluster_id}, {Columns.label}, {Columns.center}"
                f" FROM {Tables.cluster_attributes_table}"
                f" {where_clause};"
            ).fetchall()
            return cluster_attributes_parts

        # TODO: How to handle possible exception here?
        cluster_attributes_parts = cls.connection_wrapper(get_cluster_attributes_parts_worker)

        proc_cluster_attributes_parts = [
            (cluster_id, label, cls.bytes_to_tensor(center_point))
            for cluster_id, label, center_point in cluster_attributes_parts
        ]
        return proc_cluster_attributes_parts

    @classmethod
    def get_thumbnails_from_cluster(cls, cluster_id, with_embeddings_ids=False, as_dict=True):
        return cls.get_thumbnails(with_embeddings_ids, as_dict, cond=f'cluster_id = {cluster_id}')

    @classmethod
    def get_thumbnails(cls, with_embeddings_ids=False, as_dict=True, cond=''):
        thumbnails = cls.get_column(Columns.thumbnail, Tables.embeddings_table, with_embeddings_ids, as_dict, cond)
        return thumbnails

    @classmethod
    def get_image_ids(cls, with_embeddings_ids=False, as_dict=True):
        image_ids = cls.get_column(Columns.image_id, Tables.embeddings_table, with_embeddings_ids, as_dict)
        return image_ids

    @classmethod
    def get_column(cls, col, table, with_embeddings_ids=False, as_dict=True, cond='', con=None, close_connections=True):
        col_names = [Columns.embedding_id.col_name] if with_embeddings_ids else []
        col_names.append(col.col_name)
        query_results = cls.fetch_from_table(table, col_names=col_names, cond=cond, con=con,
                                             close_connections=close_connections)
        if with_embeddings_ids and as_dict:
            return dict(query_results)
        return query_results

    @classmethod
    def aggregate_col(cls, table, col, func, path_to_local_db=None, con=None, close_connections=True):

        def aggregate_worker(con):
            agg_value = con.execute(
                f"SELECT {func}({col}) FROM {table};"
            ).fetchone()
            return agg_value

        # TODO: How to handle possible exception here?
        agg_value = cls.connection_wrapper(aggregate_worker, path_to_local_db, con=con,
                                           close_connections=close_connections)
        return agg_value

    @classmethod
    def get_max_num(cls, table, col, default=0, path_to_local_db=None, con=None, close_connections=True):
        max_num_rows = cls.aggregate_col(table=table, col=col, func='MAX', path_to_local_db=path_to_local_db, con=con,
                                         close_connections=close_connections)
        max_num = max_num_rows[0]
        if isinstance(max_num, int) or isinstance(max_num, float):
            return max_num
        return default

    @classmethod
    def get_max_cluster_id(cls):
        # TODO: Change to 'get_next_cluster_id'?
        max_cluster_id = cls.get_max_num(table=Tables.cluster_attributes_table, col=Columns.cluster_id)
        return max_cluster_id

    @classmethod
    def get_max_embedding_id(cls):
        # TODO: Change to 'get_next_embedding_id'?
        max_embedding_id = cls.get_max_num(table=Tables.embeddings_table, col=Columns.embedding_id)
        return max_embedding_id

    @classmethod
    def get_max_image_id(cls, path_to_local_db):
        # TODO: Change to 'get_next_image_id'?
        # TODO: path_to_local_db needed? Provide option without it by referring to embeddings/image_paths table?
        max_image_id = cls.get_max_num(table=Tables.images_table, col=Columns.image_id,
                                       path_to_local_db=path_to_local_db)
        return max_image_id

    @classmethod
    def get_max_path_id(cls):
        # TODO: Change to 'get_next_path_id'?
        max_path_id = cls.get_max_num(table=Tables.directory_paths_table, col=Columns.path_id_col)
        return max_path_id

    @classmethod
    def get_images_attributes(cls, path_to_local_db=None):
        col_names = [Columns.file_name.col_name, Columns.last_modified.col_name]
        rows = cls.fetch_from_table(Tables.images_table, path_to_local_db=path_to_local_db, col_names=col_names)
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

    @classmethod
    def get_path_id(cls, path):
        # TODO: con params needed?
        cond = f"{Columns.path} = ?"
        cond_params = [path]
        path_id_rows = list(cls.fetch_from_table(
            Tables.directory_paths_table,
            col_names=[Columns.path_id_col.col_name],
            cond=cond,
            cond_params=cond_params
        ))
        if path_id_rows:
            path_id_row = path_id_rows[0]
            path_id = path_id_row[0]
        else:
            path_id = None
        return path_id

    @classmethod
    def get_all_embeddings(cls, with_ids=False, as_dict=False):
        # TODO: Use con params?!
        # TODO: Refactor?
        col_names = [Columns.embedding_id.col_name] if with_ids else []
        col_names.append(Columns.embedding.col_name)
        embeddings = cls.fetch_from_table(Tables.embeddings_table, col_names=col_names)
        if with_ids and as_dict:
            return dict(embeddings)
        return embeddings

    @classmethod
    def get_certain_clusters(cls):
        # TODO: Refactor!

        # select_stmt = cls.build_select(Columns.embedding_id, Tables.certain_labels_table)
        # cond = f'{Columns.embedding_id} IN ({select_stmt})'
        # embeddings_parts = cls.get_embeddings_parts(cond=cond)

        certain_cluster_parts_sql = (
            f"SELECT {Columns.embedding_id}, {Columns.embedding}, {Columns.label}"
            f" FROM {Tables.embeddings_table}"
            f" INNER JOIN {Tables.certain_labels_table} USING ({Columns.embedding_id});"
        )

        def get_certain_clusters_worker(con):
            certain_clusters_parts = con.execute(certain_cluster_parts_sql).fetchall()
            return certain_clusters_parts

        certain_clusters_parts = cls.connection_wrapper(get_certain_clusters_worker)

        max_cluster_id = cls.get_max_cluster_id()
        certain_clusters_dict = ClusterDict()
        for next_cluster_id, (embedding_id, embedding, label) in enumerate(certain_clusters_parts,
                                                                           start=max_cluster_id + 1):
            proc_embedding_id = int(embedding_id)
            proc_embedding = cls.bytes_to_tensor(embedding)
            certain_clusters_dict.add_cluster(
                Cluster(next_cluster_id, [proc_embedding], [proc_embedding_id], label)
            )
        return certain_clusters_dict

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
        data_bytes = None  # noqa
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
        # TODO: Improve efficiency?
        embeddings_row_dicts = [
            {
                Columns.cluster_id.col_name: cluster.cluster_id,
                Columns.embedding.col_name: embedding,
                Columns.thumbnail.col_name: emb_id_to_face_dict[face_id],
                Columns.image_id.col_name: emb_id_to_img_id_dict[face_id],
                Columns.embedding_id.col_name: face_id,
            }
            for cluster in clusters
            for face_id, embedding in cluster.get_embeddings(with_embeddings_ids=True)
        ]
        return embeddings_row_dicts

    @staticmethod
    def date_to_iso_string(date):
        return date.isoformat().replace('T', ' ')

    @staticmethod
    def iso_string_to_date(string):
        return datetime.datetime.fromisoformat(string)

    @classmethod
    def _build_with_clause(cls, with_clause_part):
        return cls._build_clause('WITH', with_clause_part)

    @classmethod
    def _build_where_clause(cls, cond):
        return cls._build_clause('WHERE', cond)

    @classmethod
    def _build_from_clause(cls, from_clause):
        return cls._build_clause('FROM', from_clause)

    @staticmethod
    def _build_clause(keyword, clause_part):
        return f'{keyword} {clause_part}' if clause_part else ''

    @classmethod
    def build_select(cls, select_clause, from_clause='', cond=''):
        where_clause = cls._build_where_clause(cond)
        from_clause = cls._build_from_clause(from_clause)
        return f'SELECT {select_clause} {from_clause} {where_clause}'

    @classmethod
    def get_dir_paths_to_img_ids(cls, person_label):
        """
        1. get all clusters with given label
        2. get all embeddings of these clusters
        3. get all unique image ids of these embeddings
        4. get all directory paths of these image ids (grouped)
        5. return these directory paths and image ids

        :param person_label:
        :return:
        """
        # TODO: Correct docstring
        # TODO: Add con params?
        # TODO: Refactor!
        # TODO: More efficient distinct image ids?

        # Using strange separator " || ", because windows doesn't allow pipes in file names, according to this:
        # https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#naming-conventions
        # Note that the double quotes need to be part of the sql, so must be preserved here
        strange_separator = f'"{cls.__strange_sep}"'

        person_cluster_ids_table = 'person_cluster_ids'
        with_clause_sql = f"""
            WITH {person_cluster_ids_table}({Columns.cluster_id}) AS (
                SELECT {Columns.cluster_id}
                FROM {Tables.cluster_attributes_table}
                WHERE {Columns.label} = ?
            )
        """
        from_clause_sql = f"""
            {person_cluster_ids_table}
             NATURAL JOIN {Tables.embeddings_table}
             NATURAL JOIN {Tables.image_paths_table}
             NATURAL JOIN {Tables.directory_paths_table}
        """

        # JOIN {Tables.embeddings_table} USING ({Columns.cluster_id})
        # JOIN {Tables.image_paths_table} USING ({Columns.image_id})
        # JOIN {Tables.directory_paths_table} USING ({Columns.path_id_col})

        get_dir_paths_sql = f"""
            {with_clause_sql}
            SELECT DISTINCT {Columns.path}, group_concat({Columns.image_id}, {strange_separator})
            FROM {from_clause_sql}
            GROUP BY {Columns.path};
        """

        cond_params = [person_label]

        def get_dir_paths_with_img_ids_worker(con):
            return con.execute(get_dir_paths_sql, cond_params).fetchall()

        dir_paths_with_img_ids = cls.connection_wrapper(get_dir_paths_with_img_ids_worker)
        dir_paths_to_img_ids_dict = cls._key_values_str_pairs_to_dict(dir_paths_with_img_ids)
        return dir_paths_to_img_ids_dict

    @classmethod
    def get_image_name_to_path_dict(cls, dir_path, image_ids):
        """
        1. create an image path for each image
        2. return these image paths

        :param dir_path:
        :param image_ids:
        :return:
        """
        # TODO: Add con params?
        # TODO: Refactor!
        path_to_local_db = cls.get_db_path(dir_path, local=True)
        temp_table = Tables.temp_image_ids_table
        images_table = Tables.images_table

        cond = f'{images_table}.{Columns.image_id} IN {temp_table}'
        image_id_row_dicts = [{Columns.image_id.col_name: image_id}
                              for image_id in image_ids]

        get_image_names_sql = f"""
            SELECT {Columns.file_name}
            FROM {images_table}
            WHERE {cond};
        """

        def get_image_names(con):
            cls.create_temp_table(con, temp_table)
            cls.store_in_table(temp_table, image_id_row_dicts, con=con, close_connections=False)
            return con.execute(get_image_names_sql).fetchall()

        image_names_tuples = cls.connection_wrapper(get_image_names, path_to_local_db=path_to_local_db)
        image_names = get_every_nth_item(image_names_tuples, n=0)
        image_name_to_path_dict = {
            image_name: os.path.join(dir_path, image_name)
            for image_name in image_names
        }
        return image_name_to_path_dict

    @classmethod
    def _key_values_str_pairs_to_dict(cls, key_values_str_pairs, sep=None):
        """
        *aggregate_result* is iterable of (key, values string) pairs, with the values string being of the form:
        <value 1><sep><value 2><sep>...<sep><value n>

        :param key_values_str_pairs: Iterable of (key, values string) pairs
        :param sep: Separator of values in values string. Default: ' || '
        :return:
        """
        if sep is None:
            sep = cls.__strange_sep

        result_dict = {
            key: values_str.split(sep)
            for key, values_str in key_values_str_pairs
        }
        return result_dict

    @classmethod
    def reset_cluster_ids(cls, old_ids, new_ids, con=None, close_connections=True):
        # TODO: Refactor!!
        temp_table = Tables.temp_old_and_new_ids
        temp_row_dicts = [
            {Columns.old_cluster_id.col_name: old_id,
             Columns.new_cluster_id.col_name: new_id}
            for old_id, new_id in zip(old_ids, new_ids)
        ]

        col_names_to_update = [Columns.cluster_id.col_name]
        set_values = f"{temp_table}.{Columns.new_cluster_id.col_name}"

        def reset_cluster_ids_worker(con):
            cls.create_temp_table(con, temp_table)
            cls.store_in_table(temp_table, temp_row_dicts, con=con, close_connections=False)
            cls._custom_update_table(Tables.cluster_attributes_table, col_names_to_update, set_values, con=con,
                                     close_connections=False)
            cls._custom_update_table(Tables.embeddings_table, col_names_to_update, set_values, con=con,
                                     close_connections=False)

        cls.connection_wrapper(reset_cluster_ids_worker, con=con, close_connections=close_connections)

    @classmethod
    def _custom_update_table(cls, table, col_names, set_values, on_clause=None, con=None, close_connections=True):
        if on_clause is None:
            on_clause = ''

        col_names_str = ', '.join(col_names)
        # values_template = cls.make_values_template(len(col_names))

        temp_table = Tables.temp_old_and_new_ids
        where_clause = f"{table}.{Columns.cluster_id} = {temp_table}.{Columns.old_cluster_id}"

        update_sql = f"""
            UPDATE {on_clause} {table}
            SET ({col_names_str}) = ({set_values})
            FROM ({temp_table}) AS {temp_table}
            WHERE {where_clause} 
        """

        #     UPDATE inventory
        #     SET quantity = quantity - daily.amt
        #     FROM (SELECT sum(quantity) AS amt, itemId FROM sales GROUP BY 2) AS daily
        #     WHERE inventory.itemId = daily.itemId;

        def custom_update_table_worker(con):
            con.execute(update_sql)

        cls.connection_wrapper(custom_update_table_worker, con=con, close_connections=close_connections)


class IncompleteDatabaseOperation(RuntimeError):
    pass
