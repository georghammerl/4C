// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_EXCEPTIONS_HPP
#define FOUR_C_LINALG_UTILS_EXCEPTIONS_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

/**
 * Epetra has a C-style interface returning integer error codes. This macro is meant to be wrapped
 * around a call to an Epetra function and asserts that the integer return value is equal to zero.
 * Otherwise, an exception is thrown.
 *
 * @code
 *    ASSERT_EPETRA_CALL(graph_->insert_global_value(...));
 * @endcode
 */
#define ASSERT_EPETRA_CALL(expr)                             \
  do                                                         \
  {                                                          \
    [[maybe_unused]] int err = (expr);                       \
    FOUR_C_ASSERT(err == 0, "Epetra error (code {}).", err); \
  } while (0)

#endif
