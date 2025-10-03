class Accumulator:
    """
    This class passes through but also accumulates its arguments.

    Example usage:

    acc = Accumulator()
    print(acc("foo"))  # Output: foo
    print(acc(2))      # Output: 2
    print(acc)         # Output: ['foo', 2]
    """
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)
        return value

    def __repr__(self):
        return repr(self.values)