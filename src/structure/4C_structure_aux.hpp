// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_AUX_HPP
#define FOUR_C_STRUCTURE_AUX_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_linalg_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Solid
{
  /// Determine norm of force residual
  double calculate_vector_norm(const enum Inpar::Solid::VectorNorm norm,  ///< type of norm to use
      const Core::LinAlg::Vector<double>& vect,  ///< the vector of interest
      const int numneglect =
          0  ///< number of DOFs that have to be neglected for possible length scaling
  );

  /// specific MultiMapExtractor to handle the structure field
  class MapExtractor : public Core::LinAlg::MultiMapExtractor
  {
   public:
    enum
    {
      cond_other = 0,
      cond_fsi = 1,
      cond_lung_asi = 2,
      cond_bio_gr = 3,
      cond_ale_wear = 4,
      cond_fpsi = 5,
      cond_immersed = 6,
      cond_pasi = 7
    };

    /// setup the whole thing
    void setup(const Core::FE::Discretization& dis, const Core::LinAlg::Map& fullmap,
        bool overlapping = false);

    /// get all element gids those nodes are touched by any condition
    std::shared_ptr<std::set<int>> conditioned_element_map(
        const Core::FE::Discretization& dis) const;

    MAP_EXTRACTOR_VECTOR_METHODS(other, cond_other)
    MAP_EXTRACTOR_VECTOR_METHODS(fsi_cond, cond_fsi)
    MAP_EXTRACTOR_VECTOR_METHODS(lung_asi_cond, cond_lung_asi)
    MAP_EXTRACTOR_VECTOR_METHODS(ale_wear_cond, cond_ale_wear)
    MAP_EXTRACTOR_VECTOR_METHODS(fpsi_cond, cond_fpsi)
    MAP_EXTRACTOR_VECTOR_METHODS(immersed_cond, cond_immersed)
    MAP_EXTRACTOR_VECTOR_METHODS(pasi_cond, cond_pasi)
  };
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
