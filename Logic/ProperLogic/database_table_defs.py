from collections import OrderedDict
from enum import Enum

from Logic.ProperLogic.misc_helpers import have_equal_type_names, have_equal_attrs, get_every_nth_item


# # TODO: Comparison problems when using _get_true__attr??
# def _get_true_attr(obj, enum_, obj_var_name=None):
#     # TODO: Fix comparison problems!
#     if isinstance(obj, enum_):
#         return obj.name
#     elif isinstance(obj, str):
#         return enum_[obj]
#     name_error_str = "The variable" if obj_var_name is None else f"'{obj_var_name}'"
#     raise TypeError(f"{name_error_str} must be a string or a member of {enum_.__name__}, not {obj}")


class TableSchema:
    def __init__(self, name, columns, constraints=None):
        if constraints is None:
            constraints = []
        self.name = name
        self.columns = OrderedDict((col.col_name, col) for col in columns)
        self.constraints = constraints

    def __getitem__(self, item):
        return ColumnSchema[item]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def get_columns(self):
        """

        @return: The stored ColumnSchema objects.
        """
        return list(self.get_column_dict().values())

    def get_column_names(self):
        """

        @return: The names of the stored ColumnSchema objects.
        """
        return list(self.get_column_dict().keys())

    def get_column_dict(self):
        """

        @return: The OrderedDict storing the (column_name, ColumnSchema)-pairs.
        """
        return self.columns

    def get_column_type(self, col_name):
        return self.get_column_dict()[col_name].col_type

    def sort_dict_by_cols(self, row_dict, only_values=False):
        """
        Return a list containing the items in row_dict, sorted by the order the corresponding keys appear in this
        tables' columns. If only_values is True, the dict values rather than items are returned.

        Example:
        self = TableSchema(first_name, last_name, age)
        row_dict = {'last_name': 'Olafson', 'age': 29, 'first_name': 'Lina'}
        --> ['Lina', 'Olafson', 29]

        @param row_dict:
        @param only_values:
        @return:
        """
        # if isinstance(list(row_dict.keys())[0], str):
        #     cols = self.get_column_names()
        # else:
        #     cols = self.get_columns()
        cols = self.get_column_names()
        sorted_row_items = sorted(row_dict.items(),
                                  key=lambda kv_pair: cols.index(kv_pair[0]))  # kv = key-value
        if not only_values:
            return sorted_row_items
        return get_every_nth_item(sorted_row_items, n=1)


class ColumnSchema:
    def __init__(self, col_name, col_type, col_constraint='', col_details=None):
        if col_details is not None and col_details not in ColumnDetails:
            raise ValueError(f"Unknown column details '{col_details}'.")
        if col_type is not None and col_type not in ColumnTypes:
            raise ValueError(f"Unknown column type '{col_type}'.")

        self.col_type = col_type
        self.col_name = col_name
        self.col_constraint = col_constraint
        self.details = col_details

    def __str__(self):
        return self.col_name

    def __eq__(self, other):
        if not have_equal_type_names(self, other):
            return NotImplemented
        return have_equal_attrs(self, other)

    def with_constraint(self, col_constraint):
        # TODO: Make more general version of this method?
        return ColumnSchema(self.col_name, self.col_type, col_constraint, self.details)


# TODO: Fix comparisons of tables not being equal due to this class!
#       --> Fixed?
class ColumnTypes(Enum):
    null = 'NULL'
    integer = 'INT'
    real = 'REAL'
    text = 'TEXT'
    blob = 'BLOB'

    def __eq__(self, other):
        return have_equal_type_names(self, other) and self.value == other.value

    def __str__(self):
        return self.name


class ColumnDetails(Enum):
    tensor = 'tensor'
    date = 'date'
    image = 'image'

    def __eq__(self, other):
        return have_equal_type_names(self, other) and self.value == other.value

    def __str__(self):
        return self.name


class Columns:
    center_col = ColumnSchema('center', ColumnTypes.blob, col_details=ColumnDetails.tensor)
    cluster_id_col = ColumnSchema('cluster_id', ColumnTypes.integer)
    embedding_col = ColumnSchema('embedding', ColumnTypes.blob, col_details=ColumnDetails.tensor)
    face_id_col = ColumnSchema('face_id', ColumnTypes.integer)
    file_name_col = ColumnSchema('file_name', ColumnTypes.text)
    image_id_col = ColumnSchema('image_id', ColumnTypes.integer)
    label_col = ColumnSchema('label', ColumnTypes.text)
    last_modified_col = ColumnSchema('last_modified', ColumnTypes.text, col_details=ColumnDetails.date)
    thumbnail_col = ColumnSchema('thumbnail', ColumnTypes.blob, col_details=ColumnDetails.image)


class Tables:
    images_table = TableSchema(
        'images',
        [Columns.image_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by faces table
         Columns.file_name_col.with_constraint('NOT NULL'),
         Columns.last_modified_col.with_constraint('NOT NULL')
         ],
        [f'PRIMARY KEY ({Columns.image_id_col})'
         ]
    )

    faces_table = TableSchema(
        'faces',
        [Columns.face_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by embeddings table
         Columns.image_id_col.with_constraint('NOT NULL'),
         Columns.thumbnail_col.with_constraint('NOT NULL')  # TODO: Which constraint should go here?
         ],
        [f'PRIMARY KEY ({Columns.face_id_col})',
         f'FOREIGN KEY ({Columns.image_id_col}) REFERENCES {images_table} ({Columns.image_id_col})'
         + ' ON DELETE CASCADE'
         ]
    )

    local_tables = (images_table, faces_table)

    cluster_attributes_table = TableSchema(
        'cluster_attributes',
        [Columns.cluster_id_col.with_constraint('NOT NULL'),  # also used by cluster attributes table
         Columns.label_col,
         Columns.center_col
         ],
        [f'PRIMARY KEY ({Columns.cluster_id_col})']
    )

    embeddings_table = TableSchema(
        'embeddings',
        [Columns.cluster_id_col.with_constraint('NOT NULL'),  # also used by cluster attributes table
         Columns.face_id_col.with_constraint('UNIQUE NOT NULL'),  # also used by embeddings table
         Columns.embedding_col.with_constraint('NOT NULL')
         ],
        [f'PRIMARY KEY ({Columns.face_id_col})',
         f'FOREIGN KEY ({Columns.cluster_id_col}) REFERENCES {cluster_attributes_table} ({Columns.cluster_id_col})'
         + ' ON DELETE CASCADE'
         ]
    )

    central_tables = (embeddings_table, cluster_attributes_table)

    @classmethod
    def is_local_table(cls, table):
        # TODO: Use _get_true_attr here?
        if isinstance(table, str):
            if table in cls.get_table_names(local=True):
                return True
            elif table in cls.get_table_names(local=False):
                return False
            raise ValueError(f"table '{table}' not found")

        if table in cls.local_tables:
            return True
        elif table in cls.central_tables:
            return False
        raise ValueError(f"table '{table}' not found (its type, {type(table)}, might not be TableSchema)")

    @classmethod
    def get_table_names(cls, local):
        tables = cls.local_tables if local else cls.central_tables
        return map(lambda t: t.name, tables)
