"""Small analysis helpers independent of historical manuscript output folders."""
import csv
import math
import statistics

from scipy.stats import t as student_t


def mean_ci(values):
    values = [float(v) for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    half = (student_t.ppf(.975, len(values)-1)*statistics.stdev(values)/math.sqrt(len(values))
            if len(values) > 1 else float("nan"))
    return statistics.mean(values), float(half)


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows
