import csv

def read_csv_with_constructors(filename, constructors):
    """
    Reads a CSV file and applies constructors to specified columns by header name.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    constructors : list of (callable, *colnames)
        A list where the first element is a constructor (class or function),
        and the remaining elements are column names to use as arguments.

    Returns
    -------
    tuple of lists
        One list for each constructor, containing constructed objects.
    """
    results = [[] for _ in constructors]

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for idx, (constructor, *colnames) in enumerate(constructors):
                args = [row[col] for col in colnames]
                if len(args) == 1:
                    results[idx].append(constructor(args[0]))
                else:
                    results[idx].append(constructor(*args))

    return tuple(results)