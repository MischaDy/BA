from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.database_modules.database_table_defs import Tables, Columns


class EvalDBManager(DBManager):
    @classmethod
    def get_emb_id_to_name_dict(cls, images_path, central_con=None, local_con=None, close_connections=True):
        temp_table = Tables.temp_img_ids_and_names_table
        images_cols = [Columns.image_id, Columns.file_name]

        from_clause_sql = f"""
            {Tables.embeddings_table}
            NATURAL JOIN {temp_table}
        """

        emb_id_and_name_sql = f"""
            SELECT {Columns.embedding_id}, {Columns.file_name} 
            FROM {from_clause_sql};
        """

        def get_emb_id_to_name_dict_worker(central_con, local_con):
            img_ids_and_names_rows = cls.fetch_from_table(Tables.images_table, images_cols,
                                                          path_to_local_db=path_to_local_db, con=local_con,
                                                          close_connections=False)
            img_ids_and_names_row_dicts = [
                {Columns.image_id.col_name: image_id,
                 Columns.file_name.col_name: file_name}
                for image_id, file_name in img_ids_and_names_rows
            ]
            cls.create_temp_table(temp_table, con=central_con)
            cls.store_in_table(temp_table, img_ids_and_names_row_dicts, con=central_con, close_connections=False)
            emb_ids_and_names = central_con.execute(emb_id_and_name_sql).fetchall()
            return emb_ids_and_names

        path_to_local_db = cls.get_db_file_path(images_path, local=True)

        emb_ids_and_names = cls.connection_wrapper(get_emb_id_to_name_dict_worker, path_to_local_db=path_to_local_db,
                                                   central_con=central_con, local_con=local_con, with_central=True,
                                                   with_local=True, close_connections=close_connections)
        emb_id_to_name_dict = dict(emb_ids_and_names)
        return emb_id_to_name_dict

    @classmethod
    def clear_clusters(cls, con=None, close_connections=True):
        def clear_clusters_worker(con):
            cls.delete_from_table(Tables.cluster_attributes_table, con=con, close_connections=False)
            cls.delete_from_table(Tables.certain_labels_table, con=con, close_connections=False)

        cls.connection_wrapper(clear_clusters_worker, con=con, close_connections=close_connections)
