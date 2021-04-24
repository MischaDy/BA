import logging
import operator
import os
from collections import defaultdict
from functools import reduce

from itertools import filterfalse, tee, islice


# ----- OOP -----

def have_equal_attrs(obj1, obj2):
    return obj1.__dict__ == obj2.__dict__


def have_equal_type_names(obj1, obj2):
    return type(obj1).__name__ == type(obj2).__name__


def is_instance_by_type_name(obj, class_):
    return type(obj).__name__ == class_.__name__


# ----- FUNCTIONAL -----

def starfilter(pred, iterable):
    """
    Analogous to starmap from the itertools module, but wrt. filter. Basic functionality only.

    :param pred:
    :param iterable:
    :return:
    """
    if pred is None:
        return filter(None, iterable)

    def new_pred(args):
        return pred(*args)

    return filter(new_pred, iterable)


def starfilterfalse(pred, iterable):  # noqa
    """
    Analogous to starmap and filterfalse from the itertools module. Basic functionality only.

    :param pred:
    :param iterable:
    :return:
    """
    if pred is None:
        return filterfalse(None, iterable)

    def new_pred(args):
        return pred(*args)

    return filterfalse(new_pred, iterable)


# ----- ITERATIONS -----


def partition(pred, iterable):
    """
    Use a predicate to partition entries into false entries and true entries.
    Example: partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9

    Adapted from itertools recipes: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    if isinstance(iterable, list):
        iterator1, iterator2 = iterable, iterable
    else:
        iterator1, iterator2 = tee(iterable)
    return filterfalse(pred, iterator1), filter(pred, iterator2)


# ----- I/O -----

def log_error(error):
    # TODO: Raise errors instead?
    if isinstance(error, BaseException):
        logging.error(f'{error.__class__}, {error.args}')
    else:
        logging.error(f'Error: {error}')


def wait_for_any_input(prompt):
    input(prompt + '\n')


def get_user_decision(prompt, choices_strs=None, valid_choices=None, no_choices_strs=False, allow_empty=True,
                      empty_as_none=True, sep=' / ', prefix='\n', postfix='\n', strip_input_str=True, to_lower=False,
                      to_upper=False):
    """

    :param to_upper:
    :param to_lower:
    :param strip_input_str:
    :param no_choices_strs: If true, valid_choices is also used as choices_strs, the latter being ignored.
    :param empty_as_none:
    :param allow_empty:
    :param prompt:
    :param choices_strs: Choices presented to user.
    :param valid_choices:
    :param sep:
    :param prefix:
    :param postfix:
    :return:
    """
    # TODO: Create Enum of different user decisions and use that for evaluating choice?
    if valid_choices is None:
        valid_choices = ['y', 'n']

    if no_choices_strs:
        choices_strs = valid_choices
    elif choices_strs is None:
        choices_strs = ['[y]es', '[n]o']

    choices_str = sep.join(choices_strs)
    full_prompt = prefix + f'{prompt} ({choices_str})' + postfix
    user_decision = _get_user_decision_worker(full_prompt, valid_choices=valid_choices, allow_empty=allow_empty,
                                              empty_as_none=empty_as_none, strip_input_str=strip_input_str,
                                              to_lower=to_lower, to_upper=to_upper)
    return user_decision


def _get_user_decision_worker(prompt, valid_choices=None, allow_empty=True, empty_as_none=True,
                              print_valid_inputs=False, strip_input_str=True, to_lower=False, to_upper=False):
    def get_processed_input():
        proc_input = get_user_input_of_type(prompt, allow_empty=allow_empty, empty_as_none=empty_as_none,
                                            strip_input_str=strip_input_str, to_lower=to_lower, to_upper=to_upper)
        return proc_input

    def make_error_msg():
        msg = 'Error: invalid choice'
        if print_valid_inputs and valid_choices:
            valid_choices_str = "'" + "', '".join(valid_choices) + "'"
            return msg + f" (valid choices: {valid_choices_str})"
        return msg

    if valid_choices is None:
        valid_choices = []

    user_input = get_processed_input()
    while user_input not in valid_choices:
        print(make_error_msg())
        user_input = get_processed_input()
    return user_input


def get_user_input_of_type(prompt=None, obj_name='object', class_=str, exceptions=None, allow_empty=False,
                           empty_as_none=True, strip_input_str=True, to_lower=False, to_upper=False):
    """

    :param to_upper:
    :param to_lower:
    :param strip_input_str:
    :param prompt:
    :param class_:
    :param obj_name:
    :param exceptions: Iterable of other valid values the user may enter
    :param allow_empty: If true, add empty string to list of exceptions
    :param empty_as_none: If empty is explicitly allowed and
    :return:
    """
    def __make_prompt(obj_name, must_be):
        prompt = "\n" + f"Please enter the {obj_name} ({must_be})." + "\n"
        return prompt

    def __make_error_msg(obj_name, must_be):
        error_msg = f"{obj_name} {must_be}." + "\nPlease try again."
        return error_msg

    def __make_must_be_str(class_, exceptions):
        exceptions_str = "'" + "', '".join(exceptions) + "'"
        cleaned_exceptions_str = exceptions_str.replace("''", "<empty string>")

        must_be = "must be "
        if class_ != str:
            must_be += f"convertible to a(n) {class_.__name__}"

        if len(exceptions) == 1:
            must_be += f" or have the value {cleaned_exceptions_str}"
        elif len(exceptions) > 1:
            must_be += f" or one of: {cleaned_exceptions_str}"
        return must_be

    exceptions = [] if exceptions is None else list(exceptions)
    if allow_empty:
        exceptions.append('')

    must_be_str = __make_must_be_str(class_, exceptions)
    if prompt is None:
        prompt = __make_prompt(obj_name, must_be_str)
    error_msg = __make_error_msg(obj_name, must_be_str)

    user_input = None
    while not isinstance(user_input, class_):
        user_input = input(prompt)
        if user_input in exceptions:
            break
        try:
            user_input = class_(user_input)
        except ValueError:
            log_error(error_msg)
    if empty_as_none and user_input == '':
        return None

    if class_ == str:
        return clean_string(user_input, strip=strip_input_str, to_lower=to_lower, to_upper=to_upper)
    return user_input


# ----- MISC -----

def clean_string(string, strip=True, to_lower=False, to_upper=False):
    if to_lower and to_upper:
        log_error("at most one of 'to_lower' and 'to_upper' may be provided")
        return
    clean_str = string
    if strip:
        clean_str = ' '.join(clean_str.strip().split())

    if to_lower:
        return clean_str.lower()
    elif to_upper:
        return clean_str.upper()

    return clean_str


def get_every_nth_item(iterables, n=0):
    """
    Yield nth element (zero-indexed!) in each iterable stored in the iterable.

    Example: get_every_nth_item(zip(range(3, 7), 'abcdefgh'), n=1) --> ['a', 'b', 'c', 'd']

    :param iterables: iterable of indexable iterables, each of at least length n-1 (since n is an index)
    :param n: index of element to return from each stored iterable
    :return: Generator of nth element in each iterable stored in 'iterables'
    """
    get_nth_item = operator.itemgetter(n)
    return map(get_nth_item, iterables)


def split_items(iterables, use_longest=False, fillvalue=None):
    """
    shortest iterable determines stuff!

    ...
    Return nth element (zero-indexed!) in each iterable stored in the iterable.

    Example: get_every_nth_item(zip(range(3, 7), 'abcdefgh')) --> [[3, 4, 5, 6], ['a', 'b', 'c', 'd']]

    :param fillvalue:
    :param use_longest:
    :param iterables: iterable of indexable iterables, each of at least length n-1 (since n is an index)
    :return: nth element in each iterable stored in 'iterables'
    """
    # TODO: Improve efficiency, fix docstring, refactor(?)
    # return list(starmap(get_every_nth_item, zip(iterables, range())))
    if len(iterables) == 0:
        return []
    len_aggregator = max if use_longest else min
    num_splits = len_aggregator(map(len, iterables))
    splits = [[] for _ in range(num_splits)]

    if use_longest:
        for iterable in iterables:
            for split, item in zip_longest(splits, iterable, fillvalue=fillvalue):
                split.append(item)
    else:
        for iterable in iterables:
            for split, item in zip(splits, iterable):
                split.append(item)
    return splits


def remove_items(iterable, items):
    for item in items:
        try:
            iterable.remove(item)
        except ValueError:
            log_error(f'Item {item} not found, could not be removed')


# def overwrite_list(list_, new_values):
#     list_.clear()
#     list_.extend(new_values)


def overwrite_dict(dict_, other_dict):
    dict_.clear()
    dict_.update(other_dict)


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    Taken from: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


# def open_nested_contexts(func, args=None, kwargs=None, context_managers=None):
#     if args is None:
#         args = []
#     if kwargs is None:
#         kwargs = dict()
#     if context_managers is None:
#         context_managers = []
#     return _open_nested_contexts_worker(func, args, kwargs, iter(context_managers))
#
#
# def _open_nested_contexts_worker(func, args, kwargs, context_managers_iterator):
#     try:
#         context_manager = next(context_managers_iterator)
#     except StopIteration:
#         result = func(*args, **kwargs)
#         return result
#
#     with context_manager:
#         result = _open_nested_contexts_worker(func, args, kwargs, context_managers_iterator)
#     return result


def enumerate_strs(iterable, start=0):
    str_enumeration = ((str(count), item)
                       for count, item in enumerate(iterable, start))
    return str_enumeration


def get_parent_dir_path(obj_path):
    return os.path.split(obj_path)[0]


def ignore_first_n_args_decorator(n=0):
    def ignore_first_n_args(func):
        # TODO: User functools.wraps or the like?
        def wrapped(*args, **kwargs):
            args_subset = islice(args, n, None)
            return func(*args_subset, **kwargs)
        return wrapped
    return ignore_first_n_args


def get_multiple(dict_, keys):
    return map(dict_.get, keys)


def remove_multiple(dict_, keys):
    for key in keys:
        dict_.pop(key)


def get_inverse_dict(dict_):
    keys, values = dict_.keys(), dict_.values()
    return dict(zip(values, keys))


def group_pairs(pairs, ret_dict=True):
    """
    Taken from first example in https://docs.python.org/3/library/collections.html#defaultdict-examples

    :param ret_dict:
    :param pairs:
    :return:
    """
    groups_dict = defaultdict(list)
    for key, val in pairs:
        groups_dict[key].append(val)
    if ret_dict:
        return groups_dict
    groups = sorted(groups_dict.items())
    return groups


def map_dict_vals(dict_, func):
    for key, val in dict_.items():
        dict_[key] = func(val)


def enumerate_descending(iterable):
    size = len(iterable)
    indices = reversed(range(size))
    for ind, item in zip(indices, iterable):
        yield ind, item


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def nth(iterable, n, default=None):
    """
    Returns the nth item or a default value

    Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    return next(islice(iterable, n, None), default)


def add_multiple_to_dict(dict_, items):
    for key, value in items:
        dict_[key] = value


def chain_dicts(*dicts):
    if not dicts:
        return
    elif len(dicts) == 1:
        return dicts[0]

    chained_dict = dicts[0].copy()
    for dict_ in dicts[1:]:
        for key, chained_value in chained_dict.items():
            chained_dict[key] = dict_[chained_value]

    return chained_dict


def take_first(iterable):
    return next(iter(iterable))
