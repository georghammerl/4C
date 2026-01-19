// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_BINNING_HPP
#define FOUR_C_PARTICLE_ENGINE_BINNING_HPP

#include "4C_config.hpp"

#include "4C_binstrategy.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Particle
{
  /*!
   * \brief Struct containing the relevant binning variables
   */
  struct Binning
  {
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy_;

    //! distribution of row bins
    std::shared_ptr<Core::LinAlg::Map> binrowmap_;

    //! distribution of column bins
    std::shared_ptr<Core::LinAlg::Map> bincolmap_;

    //! minimum relevant bin size
    double minbinsize_{0.0};

    //! vector of bin center coordinates
    std::shared_ptr<Core::LinAlg::MultiVector<double>> bincenters_;

    //! vector of bin weights
    std::shared_ptr<Core::LinAlg::MultiVector<double>> binweights_;

    //! relate half surrounding neighboring bins (including owned bin itself) to owned bins
    std::vector<std::set<int>> halfneighboringbinstobins_;

    //! flag denoting valid relation of half surrounding neighboring bins to owned bins
    bool validhalfneighboringbins_{false};

    //! owned bins at an open boundary or a periodic boundary
    std::set<int> boundarybins_;

    //! owned bins touched by other processors
    std::set<int> touchedbins_;

    //! maps bins of surrounding first layer to owning processors
    std::map<int, int> firstlayerbinsownedby_;

    //! bins being ghosted on this processor
    std::set<int> ghostedbins_;

    //! maps bins on this processor to processors ghosting that bins
    std::map<int, std::set<int>> thisbinsghostedby_;
  };

}  // namespace Particle

FOUR_C_NAMESPACE_CLOSE
#endif
