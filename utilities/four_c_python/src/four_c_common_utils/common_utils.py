# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Utils."""

import sys
from pathlib import Path


def file_contents(filename):
    "Return a file's contents for this transaction."
    with open(filename) as myfile:
        output = myfile.readlines()
    return output


def pretty_print_error_stderr(allerrors):
    _common_write_error(
        allerrors,
        sys.stderr,
        ["Your commit was rejected due to the following reason(s):"],
    )


def pretty_print_error_file(allerrors, errfile):
    with open(errfile, "w") as file:
        _common_write_error(allerrors, file)


def pretty_print_error_report(reason, errors, errfile):
    if len(errors) > 0:
        errors = [
            reason,
            "",
        ] + errors

        if errfile is None:
            pretty_print_error_stderr(errors)
        else:
            pretty_print_error_file(errors, errfile)


def _common_write_error(allerrors, deststream, header=None):
    max_width = max([56] + [len(line) for line in allerrors])

    print_line = lambda line: deststream.write(
        "E " + line + " " * (max_width - len(line)) + " E\n"
    )

    # print header
    deststream.write(
        "\n" + "E" * (max_width + 4) + "\nE" + " " * (max_width + 2) + "E\n"
    )
    if header is not None:
        for line in header:
            print_line(line)
        print_line("")

    # print body
    for line in allerrors:
        print_line(line)

    # footer
    deststream.write("E" + " " * (max_width + 2) + "E\n" + "E" * (max_width + 4) + "\n")


def get_module_name(file: Path) -> str | None:
    """Get the corresponding module name for a file.

    The module name is determined by the parent folder
    name which is on the same level as the first
    .contains_modules file.

    Args:
        file: The file to get the module name from

    Returns:
        The module name or None if no module is found
    """

    current_dir = file.parent.absolute()

    while True:
        if current_dir == Path("/"):
            return None
        if (current_dir / ".contains_modules").is_file():
            return file.absolute().relative_to(current_dir).parts[0]
        current_dir = current_dir.parent
