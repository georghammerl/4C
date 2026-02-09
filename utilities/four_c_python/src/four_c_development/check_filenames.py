# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that C++ files have correct filenames"""

import sys

from pathlib import Path
from four_c_common_utils import common_utils as utils


def check_file_name(file: Path) -> bool:
    """Check that the file has a valid filename.

    A valid filename is prefixed by '4C' and the module name.
    If it is a test file it is postfixed by '_test'.

    Args:
        file: The file to check

    Returns:
        True if the file has a valid filename, False otherwise
    """

    module_name = utils.get_module_name(file)

    if not file.name.startswith("4C_" if module_name is None else f"4C_{module_name}"):
        return False

    if "/tests/" in str(file) or "/unittests/" in str(file):
        if not file.name.split(".")[0].endswith("_test"):
            return False

    return True


def main():
    """Check that C++ files have correct filenames."""

    wrong_file_names = []

    for file in sys.argv[1:]:
        if not check_file_name(Path(file)):
            wrong_file_names.append(file)

    if wrong_file_names:
        print("The following files do not adhere to our file naming convention:\n")
        for wrong_file in wrong_file_names:
            print(f"    {wrong_file}")

        print(
            "\nPlease rename the files to match the naming convention.\n"
            "A valid filename must:\n"
            "   - Be prefixed with '4C_' followed by the module name\n"
            "   - If it is a test file, it should end with '_test'"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
