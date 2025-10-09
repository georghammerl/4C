# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Script for comparing result of the post_ensight filter with reference data provided in a csv file.

Point-wise quantities are compared by first finding the closest point in the output data set
using a KDTree and then comparing the values of the quantities at that point.
"""

import sys
import pyvista as pv
import argparse
import csv
import scipy.spatial as sp
import numpy as np
import math


def read_results(path):
    # Read output file
    print(f"Reading output file: {path}")
    reader = pv.get_reader(path)

    # Check if the file is time-dependent
    if not hasattr(reader, "time_values") or reader.time_values is None:
        datasets = reader.read()
        return datasets[0]
    else:
        timesteps = reader.time_values
        print(f"Found {len(timesteps)} timesteps: {timesteps}")
        last_time = timesteps[-1]
        print(f"Extracting last timestep: {last_time}")

        # Set the reader to the last timestep
        reader.set_active_time_value(last_time)
        datasets = reader.read()
        return datasets[0]


def iter_reference(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=",")

        keys = None
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
            else:
                assert keys is not None
                yield {k: v for k, v in zip(keys, row)}


def get_num_digits(float_str):
    if "." in float_str:
        return len(float_str.split(".")[-1])
    else:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare results of parallel and serial post processing by the post_ensight filter"
    )
    parser.add_argument(
        "case_file",
        type=str,
        help="Path to the parallel case file (e.g., xxx_PAR_constr3D_MPC_direct.4C.yaml_structure.case)",
    )
    parser.add_argument(
        "ref_file",
        type=str,
        help="Reference csv file",
    )
    args = parser.parse_args()
    data_set = read_results(args.case_file)

    tree = sp.KDTree(data_set.points)

    ignore_keys = ["Points:0", "Points:1", "Points:2"]

    num_error = 0

    for quantities in iter_reference(args.ref_file):
        x = np.array(
            [quantities["Points:0"], quantities["Points:1"], quantities["Points:2"]]
        )
        num_digits_points = min(
            get_num_digits(quantities["Points:0"]),
            get_num_digits(quantities["Points:1"]),
            get_num_digits(quantities["Points:2"]),
        )
        _, i = tree.query(x, distance_upper_bound=10**-num_digits_points)

        if i == len(data_set.points):
            print(
                f"No matching point found within distance bound for reference point {x} (distance_upper_bound={10**-num_digits_points})."
            )
            num_error += 1
            continue

        for key, ref_value in quantities.items():
            if key in ignore_keys:
                continue

            if ":" in key:
                name, component = key.split(":")
                component = int(component)

                if name not in data_set.point_data:
                    print(
                        f"Quantity '{name}' not found in output data. Available quantities are: {data_set.point_data.keys()}"
                    )
                    num_error += 1
                    continue
                value = data_set.point_data[name][i, component]
            else:
                if key not in data_set.point_data:
                    print(
                        f"Quantity '{key}' not found in output data. Available quantities are: {data_set.point_data.keys()}"
                    )
                    num_error += 1
                    continue
                value = data_set.point_data[key][i]

            num_digits = get_num_digits(ref_value)

            if not math.isclose(
                value, float(ref_value), rel_tol=0, abs_tol=10**-num_digits
            ):
                num_error += 1
                print(
                    f"Mismatch at point {x} for quantity {key}: {value} != {float(ref_value)}"
                )

    if num_error == 0:
        print("All quantities match the reference data.")
    else:
        print(f"Found {num_error} mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
