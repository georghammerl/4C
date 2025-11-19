// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_GRAVITY_HPP
#define FOUR_C_PARTICLE_ALGORITHM_GRAVITY_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief gravity acceleration handler for particle simulations
   *
   */
  class GravityHandler
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] gravity the gravity vector
     * \param[in] gravityrampfctnumber number of function for ramping the gravity
     */
    explicit GravityHandler(const std::vector<double>& gravity, int gravityrampfctnumber);

    /*!
     * \brief get gravity acceleration
     *
     * Evaluate the gravity ramp function at the given time to get the scaled gravity acceleration.
     *
     *
     * \param[in]  time           evaluation time
     * \param[out] scaled_gravity scaled gravity acceleration
     */
    void get_gravity_acceleration(const double time, std::vector<double>& scaled_gravity);

   protected:
    //! gravity acceleration vector
    std::vector<double> gravity_;

    //! gravity ramp function number
    const int gravityrampfctnumber_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
