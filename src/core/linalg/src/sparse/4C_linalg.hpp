// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_HPP
#define FOUR_C_LINALG_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*! \enum Core::LinAlg::DataAccess
   *  \brief Handling of data access (Copy or View)
   *
   *  If set to Core::LinAlg::DataAccess::Copy, user data will be copied at construction.
   *  If set to Core::LinAlg::DataAccess::Share, user data will be shared.
   *
   *  \note A separate Core::LinAlg::DataAccess is necessary in order to resolve
   *  possible ambiguity conflicts with the Epetra_DataAccess.
   *
   *  Use Core::LinAlg::DataAccess for construction of any Core::LINALG matrix object.
   *  Use plain 'Copy' or 'View' for construction of any Epetra matrix object.
   *
   */
  enum class DataAccess
  {
    Copy,  ///< deep copy
    Share  ///< Shared ownership to original data
  };
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
