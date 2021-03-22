from collections import OrderedDict
from enum import Enum

from Logic.ProperLogic.misc_helpers import have_equal_type_names, have_equal_attrs, get_every_nth_item


# TODO: Change DB Schema!
#       --> Remove faces table, add img_id col to embeddings_table, add cross-db FK to images table (do? and how?)
#       --> Does embeddings_table need an extra id for each? Where is that used? Can rowid be used instead?
#           --> Probably wouldn't hurt!
#       --> Add column to embeddings_table to record whether label assignment (stored in cluster_attributes) is
#           user- or algorithm-based
#       --> Make own table for labels so clusters and embeddings can have separate labels???
#           OR: Store label itself in embeddings_table to indicate that it came from user (other rows: NULL)
#           OR: Store embeddings label separate table
#           --> Choice: Extra-table! Least space use, uses embedding_ids (making them more useful themselves)
#                       Duplicates some labels, but few and storing labels separately from clusters is kinda the
#                       point!


# # TODO: Comparison problems when using _get_true__attr??
# def _get_true_attr(obj, enum_, obj_var_name=None):
#     if isinstance(obj, enum_):
#         return obj.name
#     elif isinstance(obj, str):
#         return enum_[obj]
#     name_error_str = "The variable" if obj_var_name is None else f"'{obj_var_name}'"
#     raise TypeError(f"{name_error_str} must be a string or a member of {enum_.__name__}, not {obj}")


# TODO: Create Row(dict) class?

class TableSchema:
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
        # TODO: Allow multiple constraints and use constraints enum!
        if 'PRIMARY KEY' in col_constraint:
            unique_phrase = 'UNIQUE' if 'UNIQUE' not in col_constraint else ''
            not_null_phrase = 'NOT NULL' if 'NOT NULL' not in col_constraint else ''
            col_constraint = col_constraint.replace(f'PRIMARY KEY {unique_phrase} {not_null_phrase}')
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

    local_tables = (images_table,)
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

         + ' ON DELETE CASCADE',
         # f'FOREIGN KEY ({Columns.image_id}) REFERENCES {images_table} ({Columns.image_id})'
         # + ' ON DELETE CASCADE'
         ],
    )

    central_tables = (embeddings_table, cluster_attributes_table)

    temp_cluster_ids_table = TableSchema(
        'temp_cluster_ids',
        [Columns.cluster_id]
    )

    temp_tables = (temp_cluster_ids_table, )

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
