import pandas as pd
from itertools import combinations
from constants import *


def find_functional_dependencies(df):
    """
    Finds one-to-one functional dependencies in a given pandas DataFrame.
    """

    def is_fd(lhs, rhs):
        # Group by the lhs columns and check if the rhs is unique within those groups
        grouped = df.groupby(list(lhs))
        for _, group in grouped:
            if len(group[rhs].unique()) > 1:
                return False
        return True

    fds = []
    columns = df.columns
    for r in range(1, min(len(columns), 2)):
        for lhs in combinations(columns, r):
            remaining_columns = [col for col in columns if col not in lhs]
            for rhs in remaining_columns:
                if is_fd(lhs, rhs):
                    fds.append((lhs[0], rhs))

    return fds


if __name__ == "__main__":
    data_file_path = f"{table_data_folder_path}{hosp_20k_path}"
    df = pd.read_csv(data_file_path)

    fds = find_functional_dependencies(df)
    print(f"# one-to-one fds -> {len(fds)}")
    for fd in fds:
        print(fd)
