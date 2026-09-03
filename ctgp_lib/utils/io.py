"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import os
import csv
import pandas as pd

def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    import csv
    with open(path, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_history_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
