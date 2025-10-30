# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import yaml
import pathlib
import shutil
import jinja2
import argparse
import re

PATH_TO_TESTS = ""
RECOGNIZED_TYPES = ["rst", "md"]


def load_input_file(yaml_file):
    """Returns the dictionary of parameters for the given yaml file."""
    data = yaml.safe_load((PATH_TO_TESTS / pathlib.Path(yaml_file)).read_text())
    return data


def add_raw_text_from_file(text_file, indent=0):
    """Returns the dictionary of parameters for the given yaml file."""
    with open(PATH_TO_TESTS / pathlib.Path(text_file), "r") as file:
        input = file.readlines()
    if indent > 0:
        indent_string = " " * indent
        input = "".join([indent_string + line for line in input])
    else:
        input = "".join(input)
    return input


def load_meta_data():
    """Returns a dictionary of sections and their parameters from the meta file 4C_metadata.yaml."""
    metafile_data = yaml.safe_load(
        (PATH_TO_TESTS / ("../.." / pathlib.Path("4C_metadata.yaml"))).read_text()
    )
    sections = [section["name"] for section in metafile_data["sections"]["specs"]]
    sections += metafile_data["legacy_string_sections"]

    return sections


def yaml_dump(data, filetype="rst"):
    """Returns a string of the yaml file in the given filetype.
    As of now I can return markdown and restructuredText code blocks.
    """
    if filetype == "rst":
        rststring = ""
        yaml_data = yaml.safe_dump(data, sort_keys=False).split("\n")
        for line in yaml_data:
            rststring += "    " + line + "\n"
        rststring += "\n"
        return ".. code-block:: yaml\n\n" + rststring + "\n"
    elif filetype == "md":
        return "```yaml\n" + yaml.safe_dump(data, sort_keys=False) + "```\n"
    else:
        raise TypeError(f"Filetype {filetype} for yaml_dump cannot be recognized yet.")


def section_dump(input_file_section, section_names, filetype="rst"):
    """Returns a string of the given sections from the given dictionary of sections.
    The parameter section_names can be either a string (one section name) or a list of section names
    """
    if isinstance(section_names, str):
        section_names = [section_names]
    yaml_dict = {}
    for section in section_names:
        yaml_dict[section] = input_file_section[section]

    return yaml_dump(yaml_dict, filetype)


def find_sections_in_meta(
    meta_file_data,
    section_name_expressions,
    filetype="rst",
):
    """Returns a string of the given section name expressions from the given dictionary of sections.
    It can take either one regular expression for a section name or a list of those
    """
    if isinstance(section_name_expressions, str):
        section_name_expressions = [section_name_expressions]
    section_names = []
    for section_name_expression in section_name_expressions:
        reg_expression = re.compile(section_name_expression)

        section_names += filter(reg_expression.match, meta_file_data)
    if len(section_names) == 0:
        exit("No sections found for the given regular expressions.")
    return yaml_dump(section_names, filetype)


def convert(template_path, rendering_path, input_file_path):
    global PATH_TO_TESTS
    PATH_TO_TESTS = input_file_path
    target_dir = pathlib.Path(rendering_path)
    template_list = []
    for suffix in ["*.j2", "*.rst", "*.md"]:
        template_list += list(pathlib.Path(template_path).glob(suffix))
    for template_file in template_list:

        try:
            template = jinja2.Template(template_file.read_text())
        except jinja2.exceptions.TemplateSyntaxError as e:
            print(f"Warning: Could not read {template_file}: {e}")
            continue
        docfile_name = (
            template_file.stem if template_file.suffix == ".j2" else template_file
        )
        docfile_fullpath = target_dir / docfile_name
        print(f"source: {docfile_name}, target: {docfile_fullpath}")
        docfile_fullpath.write_text(
            template.render(
                section_dump=section_dump,
                load_input_file=load_input_file,
                yaml_dump=yaml_dump,
                find_sections_in_meta=find_sections_in_meta,
                load_meta_data=load_meta_data,
                add_raw_text_from_file=add_raw_text_from_file,
                len=len,
            )
        )
    for filetype in RECOGNIZED_TYPES:
        for file in pathlib.Path(template_path).glob("*." + filetype):
            shutil.copy(file, target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render all tutorials in the given directory."
    )
    parser.add_argument(
        "template_path", type=str, help="Path to the tutorial templates"
    )
    parser.add_argument(
        "rendering_path", type=str, help="Path to the final tutorial files"
    )
    parser.add_argument("input_file_path", type=str, help="Path to the input files")
    # Parse the arguments
    args = parser.parse_args()

    convert(args.template_path, args.rendering_path, args.input_file_path)
