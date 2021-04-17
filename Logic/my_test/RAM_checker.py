import psutil


# in bytes
KILOBYTE = 2 ** 10
MEGABYTE = 2 ** 20
GIGABYTE = 2 ** 30

PREC = 3


# https://stackoverflow.com/questions/1456373/two-way-reverse-map
class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


BYTES_2DICT = TwoWayDict([(KILOBYTE, "KB"),
                          (MEGABYTE, "MB"),
                          (GIGABYTE, "GB")])


def available_mem(prec=3):
    """Return available memory in Kilo, Mega-, or Gigabytes, depending on practicality."""

    mem = psutil.virtual_memory().available

    for test_divisor in sorted(BYTES_2DICT.keys(), reverse=True):
        quotient = mem // test_divisor
        if quotient > 0:
            divisor = test_divisor
            break
    return round(mem / divisor, prec), BYTES_2DICT[divisor]


def print_available_mem(prec=3):
    print('{} {} available'.format(*available_mem(prec)))


if __name__ == '__main__':
    print_available_mem(PREC)
