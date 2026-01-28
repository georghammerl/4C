// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_BVH_HPP
#define FOUR_C_GEOMETRIC_SEARCH_BVH_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"

#ifdef FOUR_C_WITH_ARBORX
#include <ArborX.hpp>
#endif

#include "4C_geometric_search_access_traits.hpp"

#include <Kokkos_Core.hpp>

#include <tuple>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  struct BoundingVolume;

  /*! \brief Structure to hold a pair found during a collision search
   */
  struct CollisionSearchResult
  {
    //! Global ID of the predicate
    int gid_predicate;
    //! Global ID of the primitive
    int gid_primitive;
  };

  /*!
   * \brief Wrapper class for the ArborX bounding volume hierarchy.
   */
  class BoundingVolumeHierarchy
  {
   public:
    // Primitives placeholder
    using Primitives = BoundingVolumeVectorPlaceholder<PrimitivesTag>;

#ifndef FOUR_C_WITH_ARBORX
    /*! \brief This class can not be used without ArborX, add empty methods and a controlled error.
     */
    explicit BoundingVolumeHierarchy(Primitives const& values)
    {
      FOUR_C_THROW(
          "The class 'Core::GeometricSearch::BoundingVolumeHierarchy' can only be used with ArborX."
          "To use it, enable ArborX during the configure process.");
    }

    using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
    template <class Predicates>
    std::tuple<Kokkos::View<int*, MemorySpace>, Kokkos::View<int*, MemorySpace>> query(
        Predicates const& predicates) const
    {
      FOUR_C_THROW(
          "The class 'Core::GeometricSearch::BoundingVolumeHierarchy' can only be used with ArborX."
          "To use it, enable ArborX during the configure process.");
    }
#else

    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using ArborXBoundingVolumeHierarchy =
        ArborX::BoundingVolumeHierarchy<typename ExecutionSpace::memory_space,
            ArborX::Details::AccessValues<Primitives>::value_type>;
    using MemorySpace = typename ArborXBoundingVolumeHierarchy::memory_space;

    /*
     * @brief Generate the bounding volume hierarchy from a vector of primitives.
     */
    explicit BoundingVolumeHierarchy(const Primitives& values)
        : execution_{}, arborx_bounding_volume_hierarchy_(execution_, values)
    {
    }

    /*! \brief Query the bounding volume hierarchy with a given vector of predicates and record
     * results in {indices, offsets}.
     *
     * (From ArborX documentation)
     * Finds all primitives meeting the predicates and record results in {indices, offsets}. indices
     * stores the indices of the objects that satisfy the predicates. offsets stores the locations
     * in indices that start a predicate, that is, predicates(i) is satisfied by
     * primitives(indices(j)) for offsets(i) <= j < offsets(i+1). Following the usual convention,
     * offsets(n) == indices.size(), where n is the number of queries that were performed and
     * indices.size() is the total number of collisions. Be aware, that this function will return
     * all in primitives vs. all in predicates!
     *
     * @param predicates Bounding volumes to intersect with
     * @return {indices, offsets} See description above
     *
     * D. Lebrun-Grandie, A. Prokopenko, B. Turcksin, and S. R. Slattery. 2020.
     * ArborX: A Performance Portable Geometric Search Library. ACM Trans. Math. Softw. 47, 1,
     * Article 2 (2021), https://doi.org/10.1145/3412558
     */
    template <class Predicates>
    auto query(Predicates const& predicates) const
    {
      Kokkos::View<int*, MemorySpace> indices("indices_full", 0);
      Kokkos::View<int*, MemorySpace> offsets("offset_full", 0);

      auto get_indices_callback =
          KOKKOS_LAMBDA(const auto predicate, const auto& value, const auto& out)->void
      {
        out(value.index);
      };

      // Perform the collision check.
      arborx_bounding_volume_hierarchy_.query(execution_,
          BoundingVolumeVectorPlaceholder<PredicatesTag>{predicates}, get_indices_callback, indices,
          offsets);

      return std::make_tuple(indices, offsets);
    }

   private:
    ExecutionSpace execution_;
    ArborXBoundingVolumeHierarchy arborx_bounding_volume_hierarchy_;
#endif
  };

  /*!
   * @brief Find all primitives meeting the predicates and record all matching pairs in a vector.
   *
   * @param primitives Bounding volumes to search for intersections
   * @param predicates Bounding volumes to intersect with
   * @return pairs Vector of the found interaction pairs
   */
  std::vector<CollisionSearchResult> collision_search(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates);

  /*! \brief Find all primitives meeting the predicates and record all matching pairs in a vector,
   * also, this function prints the results.
   *
   * @param primitives Bounding volumes to search for intersections
   * @param predicates Bounding volumes to intersect with
   * @param comm Communicator object of the discretization
   * @param verbosity Enabling printout of the geometric search information
   * @return pairs Vector of the found interaction pairs
   */
  std::vector<CollisionSearchResult> collision_search_print_results(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates, const MPI_Comm comm,
      const Core::IO::Verbositylevel verbosity);

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
