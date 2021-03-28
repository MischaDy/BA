import logging
import operator

from itertools import zip_longest


# ----- OOP -----

def have_equal_attrs(obj1, obj2):
    return obj1.__dict__ == obj2.__dict__


def have_equal_type_names(obj1, obj2):
    return type(obj1).__name__ == type(obj2).__name__


def is_instance_by_type_name(obj, class_):
    return type(obj).__name__ == class_.__name__


# ----- I/O -----

def log_error(msg):
    logging.error(f'Error: {msg}')


def wait_for_any_input(prompt):
    input(prompt + '\n')


def get_user_decision(prompt, choices_strs=('[y]es', '[n]o'), valid_choices=('y', 'n'), sep=' / ', prefix='\n',
                      postfix='\n', should_clean_decision=True):
    # TODO: Create Enum of different user decisions and use that for evaluating choice?
    # TODO: Allow to abort (param what the abort input should look like)

    choices_str = sep.join(choices_strs)
    full_prompt = prefix + f'{prompt} ({choices_str})' + postfix
    user_decision = get_user_input(full_prompt, valid_choices=valid_choices, should_clean_input=should_clean_decision)
    return user_decision


def get_user_input(prompt, valid_choices=None, print_valid_inputs=False, should_clean_input=True):
    if valid_choices is None:
        valid_choices = []

    def get_processed_input():
        if should_clean_input:
            return clean_str(input(prompt))
        return input(prompt)

    def make_error_msg():
        msg = 'Error: invalid choice'
        if print_valid_inputs and valid_choices:
            valid_choices_str = "'" + "', '".join(valid_choices) + "'"
            return msg + f" (valid choices: {valid_choices_str})"
        return msg

    user_input = get_processed_input()
    while user_input not in valid_choices:
        print(make_error_msg())
        user_input = get_processed_input()
    return user_input


# ----- MISC -----

def clean_str(string, to_lower=True):
    clean_string = ' '.join(string.strip().split())
    if to_lower:
        return clean_string.lower()
    return clean_string.upper()


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


def overwrite_list(iterable, new_values):
    iterable.clear()
    iterable.extend(new_values)
