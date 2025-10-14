// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FILE_UTILS_HPP
#define FOUR_C_IO_INPUT_FILE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <ostream>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;

  namespace Nurbs
  {
    class Knotvector;
  }
}  // namespace Core::FE

namespace Core::IO
{
  class InputFile;
  class InputSpec;
}  // namespace Core::IO

namespace Core::IO
{
  void read_parameters_in_section(
      InputFile& input, const std::string& section_name, Teuchos::ParameterList& list);

  /**
   * Read a node-design topology section. This is a collective call that propagates data that
   * may only be available on rank 0 to all ranks.
   *
   * @param input The input file.
   * @param name Name of the topology to read
   * @param dobj_fenode Resulting collection of all nodes that belong to a design.
   * @param get_discretization Callback to return a discretization by name.
   */
  void read_design(InputFile& input, const std::string& name,
      std::vector<std::vector<int>>& dobj_fenode,
      const std::function<const Core::FE::Discretization&(const std::string& name)>&
          get_discretization);

  /**
   * @brief Read the knot vector section (for isogeometric analysis)
   *
   * @param  input         (in ): InputFile object
   * @param  name           (in ): Name/type of discretisation
   *
   * @return The Knotvector object read from the input file.
   */
  std::unique_ptr<Core::FE::Nurbs::Knotvector> read_knots(
      InputFile& input, const std::string& name);

}  // namespace Core::IO


FOUR_C_NAMESPACE_CLOSE

#endif
