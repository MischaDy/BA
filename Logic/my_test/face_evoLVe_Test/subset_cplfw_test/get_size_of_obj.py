from __future__ import print_function
from sys import getsizeof  # , stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(obj, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate size_of object without __sizeof__

    def size_of(obj):
        if id(obj) in seen:       # do not double count the same object
            return 0
        seen.add(id(obj))
        s = getsizeof(obj, default_size)

        if verbose:
            print(s, type(obj), repr(obj))

        for type_, handler in all_handlers.items():
            if isinstance(obj, type_):
                s += sum(map(size_of, handler(obj)))
                break
        else:
            if not hasattr(obj.__class__, '__slots__'):
                if hasattr(obj, '__dict__'):
                    # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `class_(None)`)
                    # have neither
                    s += size_of(obj.__dict__)
                # else, `o` has no attributes at all, so sys.getsizeof() actually returned the correct value
            else:
                s += sum(size_of(getattr(obj, x)) for x in obj.__class__.__slots__ if hasattr(obj, x))

        return s

    return size_of(obj)


## Example call ##

if __name__ == '__main__':
    d = dict(a=1, b=2, c=3, d=[4, 5, 6, 7], e='a string of chars')
    print(total_size(d, verbose=True))
