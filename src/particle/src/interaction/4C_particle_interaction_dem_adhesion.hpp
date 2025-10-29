// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
}  // namespace Particle

namespace Particle
{
  class WallHandlerInterface;
}

namespace Particle
{
  class InteractionWriter;
  class DEMNeighborPairs;
  class DEMHistoryPairs;
  class DEMAdhesionLawBase;
  class DEMAdhesionSurfaceEnergyBase;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMAdhesion final
  {
   public:
    //! constructor
    explicit DEMAdhesion(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~DEMAdhesion();

    //! init contact handler
    void init();

    //! setup contact handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs,
        const std::shared_ptr<Particle::DEMHistoryPairs> historypairs, const double& k_normal);

    //! get adhesion distance
    inline double get_adhesion_distance() const { return adhesion_distance_; };

    //! add adhesion contribution to force field
    void add_force_contribution();

   private:
    //! init adhesion law handler
    void init_adhesion_law_handler();

    //! init adhesion surface energy handler
    void init_adhesion_surface_energy_handler();

    //! setup particle interaction writer
    void setup_particle_interaction_writer();

    //! evaluate particle adhesion contribution
    void evaluate_particle_adhesion();

    //! evaluate particle-wall adhesion contribution
    void evaluate_particle_wall_adhesion();

    //! discrete element method specific parameter list
    const Teuchos::ParameterList& params_dem_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! particle interaction writer
    std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter_;

    //! neighbor pair handler
    std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_;

    //! history pair handler
    std::shared_ptr<Particle::DEMHistoryPairs> historypairs_;

    //! adhesion law handler
    std::unique_ptr<Particle::DEMAdhesionLawBase> adhesionlaw_;

    //! adhesion surface energy handler
    std::unique_ptr<Particle::DEMAdhesionSurfaceEnergyBase> adhesionsurfaceenergy_;

    //! adhesion distance
    const double adhesion_distance_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
