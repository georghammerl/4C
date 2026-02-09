#!/usr/bin/env python3
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that all companion input files are used within an input file.

Current files include:

    - *.mat.4C.yaml
    - *.mat.4C.json
    - *.xml
    - *.csv
    - *.json
    - *.mtx
    - *.py
    - *.txt
    - *.exo
    - *.msh
"""

import sys

from pathlib import Path


def is_input_file(file: Path) -> bool:
    """Check if file is an input file, i.e., has an extension of .4C.yaml or .4C.json.

    Args:
        file: Path to file to check.
    Returns:
        True if file is an input file, False otherwise.
    """

    if file.name.endswith(".4C.yaml") or file.name.endswith(".4C.json"):
        if file.name.endswith(".mat.4C.yaml") or file.name.endswith(".mat.4C.json"):
            return False

        return True

    return False


def is_companion_file(file: Path) -> bool:
    """Check if file is a companion input file, i.e., lives on top level
    of /tests/input_files/ and is not an input file, for example *.json,
    *.xml, *.mat.4C.yaml, ...

    Args:
        file: Path to file to check.

    Returns:
        True if file is a companion input file, False otherwise.
    """

    # exclude subfolders
    if file.parent != Path("tests/input_files"):
        return False

    # exclude input files
    if is_input_file(file):
        return False

    return True


def main():
    """Check that all companion input files are used within an input file."""

    companion_files = []
    unused_companion_files = []

    for file in sys.argv[1:]:
        if is_companion_file(Path(file)):
            companion_files.append(file)

    # retrieve all input files
    # note: this is necessary because pre-commit batches the files into chunks
    # (even with require_serial: true) and thus not all companion files are
    # passed to the script at once, but only a subset of them. Therefore, we
    # need to retrieve all input files from the file system to check if any of
    # the companion files are used in any of the input files.
    input_files = []

    # only get top-level files from /tests/input_files and do not use folders within
    for file in Path("tests/input_files").iterdir():
        if file.is_file() and is_input_file(file):
            input_files.append(file)

    # check if filename of companion file is mentioned in any input file
    for companion_file in companion_files:
        companion_file_name = Path(companion_file).name
        found = False

        for input_file in input_files:
            with open(input_file, "r") as f:
                if companion_file_name in f.read():
                    found = True
                    break

        if not found:
            unused_companion_files.append(companion_file)

    if unused_companion_files:
        print("The following companion input files are not used in any input file:\n")
        for file in unused_companion_files:
            print(f"    {file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
