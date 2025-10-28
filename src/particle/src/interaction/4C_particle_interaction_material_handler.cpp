// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_material_handler.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::MaterialHandler::MaterialHandler(const Teuchos::ParameterList& params) : params_(params)
{
  // empty constructor
}

void Particle::MaterialHandler::init()
{
  // init map relating particle types to material ids
  std::map<Particle::TypeEnum, int> typetomatidmap;

  // read parameters relating particle types to values
  Particle::Utils::read_params_types_related_to_values(
      params_, "PHASE_TO_MATERIAL_ID", typetomatidmap);

  // determine size of vector indexed by particle types
  const int typevectorsize = ((--typetomatidmap.end())->first) + 1;

  // allocate memory to hold particle types
  phasetypetoparticlematpar_.resize(typevectorsize);

  // relate particle types to particle material parameters
  for (auto& typeIt : typetomatidmap)
  {
    // get type of particle
    Particle::TypeEnum type_i = typeIt.first;

    // add to set of particle types of stored particle material parameters
    storedtypes_.insert(type_i);

    // get material parameters and cast to particle material parameter
    const Core::Mat::PAR::Parameter* matparameter =
        Global::Problem::instance()->materials()->parameter_by_id(typeIt.second);
    const Mat::PAR::ParticleMaterialBase* particlematparameter =
        dynamic_cast<const Mat::PAR::ParticleMaterialBase*>(matparameter);

    // safety check
    if (particlematparameter == nullptr) FOUR_C_THROW("cast to specific particle material failed!");

    // relate particle types to particle material parameters
    phasetypetoparticlematpar_[type_i] = particlematparameter;
  }
}

void Particle::MaterialHandler::setup()
{
  // nothing to do
}

FOUR_C_NAMESPACE_CLOSE
