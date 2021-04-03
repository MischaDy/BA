from collections import OrderedDict
from enum import Enum
from itertools import repeat, starmap

from Logic.ProperLogic.misc_helpers import have_equal_type_names, have_equal_attrs, get_every_nth_item, first_true


# Design decision: Change of DB Schema
#       --> Remove faces table, add img_id col to embeddings_table, add cross-db FK to images table (do? and how?)
#       --> Does embeddings_table need an extra id for each? Where is that used? Can rowid be used instead?
#           --> Probably wouldn't hurt!
#       --> Add column to embeddings_table to record whether label assignment (stored in cluster_attributes) is
#           user- or algorithm-based
#       --> Make own table for labels so clusters and embeddings can have separate labels???
#           OR: Store label itself in embeddings_table to indicate that it came from user (other rows: NULL)
#           OR: Store embeddings label separate table
#           --> Choice: Extra-table! Least space use, uses embeddings_ids (making them more useful themselves)
#                       Duplicates some labels, but few and storing labels separately from clusters is kinda the
#                       point!

class TableSchema:
    # TODO: Create a make_row_dicts function?
    def __init__(self, name, columns, constraints=None):
        if constraints is None:
            constraints = []
        self.name = name
        self.columns = OrderedDict((col.col_name, col) for col in columns)
        self.constraints = constraints

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def get_columns(self):
        """

        :return: The stored ColumnSchema objects.
        """
        return list(self.get_column_dict().values())

    def get_column_names(self):
        """

        :return: The names of the stored ColumnSchema objects.
        """
        return list(self.get_column_dict().keys())

    def get_column_dict(self):
        """

        :return: The OrderedDict storing the (column_name, ColumnSchema)-pairs.
        """
        return self.columns

    def get_column_type(self, col_name):
        return self.get_column_dict()[col_name].col_type

    def sort_dict_by_cols(self, row_dict, only_values=True):
        """
        Return a list containing the elements in row_dict, sorted by the order the corresponding keys appear in this
        tables' columns. If only_values is True, the dict values rather than items are returned.

        Example:
        self = TableSchema(first_name, last_name, age)
        row_dict = {'last_name': 'Olafson', 'age': 29, 'first_name': 'Lina'}
        --> ['Lina', 'Olafson', 29]

        :param row_dict:
        :param only_values:
        :return:
        """
        cols = self.get_column_names()
        sorted_row_items = sorted(row_dict.items(),
                                  key=lambda kv_pair: cols.index(kv_pair[0]))  # kv = key-value
        if only_values:
            return list(get_every_nth_item(sorted_row_items, n=1))
        return sorted_row_items

    def row_to_row_dict(self, row):
        return dict(zip(self.get_column_names(), row))

    def make_row_dicts(self, values_objects, repetition_flags=None):
        # TODO: Fix bug!
        cols_names = self.get_column_names()
        values_iterables = self.make_values_iterables(values_objects, repetition_flags)
        row_dicts = [dict(zip(cols_names, values_iterable))
                     for values_iterable in values_iterables]
        return row_dicts

    @staticmethod
    def make_values_iterables(values_objects, repetition_flags=None, num_repetitions=None):
        """

        :param values_objects:
        :param repetition_flags:
        :param num_repetitions:
        :return:
        """
        # TODO: Allow for values_objects to consist of non-iterables!
        # TODO: Use max/min appropriate values_iterable instead of first?

        if repetition_flags is None:
            repetition_flags = list(repeat(False, len(values_objects)))
        if num_repetitions is None:
            if not any(repetition_flags):
                raise ValueError("'num_repetitions' must be provided, or an element in 'values_objects' must not be"
                                 " flagged as to-be-repeated")

            def is_not_repeated(values_obj_with_ind):
                ind, values_obj = values_obj_with_ind
                return not repetition_flags[ind]

            first_values_iterable = first_true(enumerate(values_objects), default=None,
                                               pred=is_not_repeated)
            num_repetitions = len(first_values_iterable)

        def repeat_object_if_needed(repetition_flag, values_object):
            if repetition_flag:
                return list(repeat(values_object, num_repetitions))
            return values_object

        values_iterables = starmap(repeat_object_if_needed, zip(repetition_flags, values_objects))
        return values_iterables


class ColumnSchema:
    def __init__(self, col_name, col_type, col_constraint='', col_details=None):
        if col_details is not None and col_details not in ColumnDetails:
            raise ValueError(f"Unknown column details '{col_details}'.")
        if col_type is not None and col_type not in ColumnTypes:
            raise ValueError(f"Unknown column type '{col_type}'.")

        self.col_type = col_type
        self.col_name = col_name
        self.col_constraint = col_constraint
        self.col_details = col_details

    def __str__(self):
        return self.col_name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def with_constraint(self, col_constraint):
        # TODO: Allow 'KEY' as constraint with same effect as primary key
        # TODO: Allow multiple constraints and use constraints enum!
        if 'PRIMARY KEY' in col_constraint:
            phrase_unique = 'UNIQUE' if 'UNIQUE' not in col_constraint else ''
            phrase_not_null = 'NOT NULL' if 'NOT NULL' not in col_constraint else ''
            col_constraint = col_constraint.replace('PRIMARY KEY', f'PRIMARY KEY {phrase_unique} {phrase_not_null}')
        return ColumnSchema(self.col_name, self.col_type, str(col_constraint), self.col_details)

    # @classmethod
    # def get_column_schema(cls, col_schema_name):
    #     return cls.__dict__[col_schema_name]


class ColumnTypes(Enum):
    null = 'NULL'
    integer = 'INTEGER'
    real = 'REAL'
    text = 'TEXT'
    blob = 'BLOB'

    def __eq__(self, other):
        return have_equal_type_names(self, other) and self.value == other.value

    def __str__(self):
        return self.name


class ColumnDetails(Enum):
    # TODO: Store in better way?
    tensor = 'tensor'
    date = 'date'
    image = 'image'

    def __eq__(self, other):
        return have_equal_type_names(self, other) and self.value == other.value

    def __str__(self):
        return self.name


# class ColumnConstraints(Enum):
#     # TODO: Use!
#     # TODO: Store in better way?
#     not_null = 'NOT NULL'
#     unique = 'UNIQUE'
#     primary_key = 'PRIMARY KEY'
#
#     def __eq__(self, other):
#         return have_equal_type_names(self, other) and self.value == other.value
#
#     def __str__(self):
#         return self.name


class Columns:
    center = ColumnSchema('center', ColumnTypes.blob, col_details=ColumnDetails.tensor)
    cluster_id = ColumnSchema('cluster_id', ColumnTypes.integer)
    embedding = ColumnSchema('embedding', ColumnTypes.blob, col_details=ColumnDetails.tensor)
    embedding_id = ColumnSchema('embedding_id', ColumnTypes.integer)
    file_name = ColumnSchema('file_name', ColumnTypes.text)
    image_id = ColumnSchema('image_id', ColumnTypes.integer)
    label = ColumnSchema('label', ColumnTypes.text)
    last_modified = ColumnSchema('last_modified', ColumnTypes.text, col_details=ColumnDetails.date)
    thumbnail = ColumnSchema('thumbnail', ColumnTypes.blob, col_details=ColumnDetails.image)
    path_id_col = ColumnSchema('path_id_col', ColumnTypes.integer)
    path = ColumnSchema('path', ColumnTypes.text)
    old_cluster_id = ColumnSchema('old_cluster_id', ColumnTypes.integer)
    new_cluster_id = ColumnSchema('new_cluster_id', ColumnTypes.integer)

    @classmethod
    def get_column(cls, col_name):
        return cls.__dict__[col_name]


class Tables:
    # ----- local tables -----

    images_table = TableSchema(
        'images',
        [Columns.image_id.with_constraint('PRIMARY KEY'),
         Columns.file_name.with_constraint('NOT NULL'),
         Columns.last_modified.with_constraint('NOT NULL')
         ],
        []
    )

    path_id_table = TableSchema(
        'path_id',
        [Columns.path_id_col.with_constraint('PRIMARY KEY')
         ],
        []
    )

    local_tables = (images_table, path_id_table)

    # ----- central tables -----

    cluster_attributes_table = TableSchema(
        'cluster_attributes',
        [Columns.cluster_id.with_constraint('PRIMARY KEY'),  # also used by embeddings table
         Columns.label,
         Columns.center
         ],
        []
    )

    embeddings_table = TableSchema(
        'embeddings',
        [Columns.cluster_id.with_constraint('NOT NULL'),  # also used by cluster attributes table
         Columns.image_id.with_constraint('NOT NULL'),
         Columns.embedding_id.with_constraint('PRIMARY KEY'),
         Columns.embedding.with_constraint('NOT NULL'),
         Columns.thumbnail.with_constraint('NOT NULL'),
         ],
        [f'FOREIGN KEY ({Columns.cluster_id}) REFERENCES {cluster_attributes_table} ({Columns.cluster_id})'
         + ' ON DELETE CASCADE',
         ],
    )

    certain_labels_table = TableSchema(
        'certain_labels',
        [Columns.embedding_id.with_constraint('PRIMARY KEY'),
         Columns.label.with_constraint('NOT NULL'),
         ],
        [f'FOREIGN KEY ({Columns.embedding_id}) REFERENCES {embeddings_table} ({Columns.embedding_id})'
         + ' ON DELETE CASCADE',
         ],
    )

    directory_paths_table = TableSchema(
        'directory_paths',
        [Columns.path_id_col.with_constraint('PRIMARY KEY'),
         Columns.path.with_constraint('UNIQUE NOT NULL')
         ],
        [],
    )

    # TODO: FK with image_id col not possible, since not unique in embeddings table!
    image_paths_table = TableSchema(
        'image_paths',
        [Columns.image_id.with_constraint('PRIMARY KEY'),
         Columns.path_id_col.with_constraint('NOT NULL'),
         ],
        [f'FOREIGN KEY ({Columns.path_id_col}) REFERENCES {directory_paths_table} ({Columns.path_id_col})'
         + ' ON DELETE CASCADE',
         ],
    )

    central_tables = (embeddings_table, cluster_attributes_table, certain_labels_table, directory_paths_table,
                      image_paths_table)

    # ----- temp tables -----

    temp_cluster_ids_table = TableSchema(
        'temp_cluster_ids',
        [Columns.cluster_id]
    )

    temp_image_ids_table = TableSchema(
        'temp_image_ids',
        [Columns.image_id]
    )

    temp_old_and_new_ids = TableSchema(
        'temp_old_and_new_ids',
        [Columns.old_cluster_id,
         Columns.new_cluster_id]
    )

    temp_tables = (temp_cluster_ids_table, temp_image_ids_table)

    @classmethod
    def is_local_table(cls, table):
        if isinstance(table, str):
            if table in cls.get_table_names(local=True):
                return True
            elif table in cls.get_table_names(local=False):
                return False
            raise ValueError(f"table '{table}' not found")

        if table in cls.local_tables:
            return True
        elif table in cls.central_tables or table in cls.temp_tables:
            return False
        raise ValueError(f"table '{table}' not found (its type, {type(table)}, might not be TableSchema)")

    @classmethod
    def get_table_names(cls, local):
        tables = cls.local_tables if local else cls.central_tables
        return map(lambda t: t.name, tables)
