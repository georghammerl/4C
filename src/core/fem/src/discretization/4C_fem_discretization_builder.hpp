// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DISCRETIZATION_BUILDER_HPP
#define FOUR_C_FEM_DISCRETIZATION_BUILDER_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_io_mesh.hpp"

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /**
   * @brief A builder class to fill Discretization objects
   *
   * The aim of this class is to separate incremental construction of a Discretization from
   * the Discretization class itself.
   */
  template <int dim>
  class DiscretizationBuilder
  {
   public:
    using IndexType = int;

    /**
     * @brief Add a node to the builder
     */
    void add_node(std::span<const double, dim> x, IndexType global_id,
        std::shared_ptr<Core::Nodes::Node> user_node = nullptr);

    /**
     * @brief Add an element to the builder
     */
    void add_element(Core::FE::CellType cell_type, std::span<const IndexType> node_ids,
        IndexType global_id, std::shared_ptr<Core::Elements::Element> user_element = nullptr);

    /**
     * @brief Build a Discretization from the data added to the builder
     */
    void build(Discretization& discretization,
        const Core::Rebalance::RebalanceParameters& rebalance_parameters);

    /**
     * @brief Check whether the data added to the builder is consistent
     *
     * This checks that
     *  - all element node ids refer to nodes that have been added to the builder
     *  - all nodes are used by at least one element
     */
    void assert_consistent() const;

   private:
    struct NodeData
    {
      std::array<double, dim> x;
      IndexType global_id;
      std::shared_ptr<Core::Nodes::Node> user_node;
    };

    struct ElementData
    {
      Core::FE::CellType cell_type;
      std::vector<IndexType> node_ids;
      IndexType global_id;
      std::shared_ptr<Core::Elements::Element> user_element;
    };

    std::map<IndexType, NodeData> nodes_;
    std::map<IndexType, ElementData> elements_;

    std::unordered_set<IndexType> used_node_ids_;
  };
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
