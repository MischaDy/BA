import logging


# ----- OOP -----
def have_equal_attrs(obj1, obj2):
    return obj1.__dict__ == obj2.__dict__


def have_equal_type_names(obj1, obj2):
    return type(obj1).__name__ == type(obj2).__name__


# ----- I/O -----

def log_error(msg):
    logging.error('Error: ' + msg)


def wait_for_any_input(prompt):
    input(prompt + '\n')


# ----- MISC -----
def clean_str(string, to_lower=True):
    clean_string = string.strip()
    if to_lower:
        return clean_string.lower()
    return clean_string.upper()


def get_nth_tuple_elem(iterables, n=0):
    """
    Return nth element (zero-indexed!) in each iterable stored in the iterable.

    Example: _get_nth_tuple_elem(zip(range(3, 7), 'abcdefgh'), n=1) --> ['a', 'b', 'c', 'd']

    iterables: iterable of indexable iterables, each of at least length n-1 (since n is an index).
    n: index of element to return from each stored iterable
    """
    return list(map(lambda iterable: iterable[n], iterables))
