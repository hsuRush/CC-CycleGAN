from argparse import ArgumentTypeError

# Commandline argument parsing
###
def probility_float(arg):
    try:
        value = float(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, float.__name__))

    if 1 > value > 0:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than 0 and lesser than 1 was '
            'required'.format(arg, float.__name__, x))

def positive_int(arg):
    return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)


def positive_float(arg):
    return number_greater_x(arg, float, 0)

def number_greater_x(arg, type_, x):
    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than {} was '
            'required'.format(arg, type_.__name__, x))
