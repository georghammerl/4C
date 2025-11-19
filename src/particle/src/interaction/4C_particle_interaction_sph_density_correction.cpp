// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_density_correction.hpp"

#include "4C_particle_input.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHDensityCorrectionBase::SPHDensityCorrectionBase()
{
  // empty constructor
}

void Particle::SPHDensityCorrectionBase::corrected_density_interior(
    const double* denssum, double* dens) const
{
  dens[0] = denssum[0];
}

Particle::SPHDensityCorrectionInterior::SPHDensityCorrectionInterior()
    : Particle::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool Particle::SPHDensityCorrectionInterior::compute_density_bc() const { return false; }

void Particle::SPHDensityCorrectionInterior::corrected_density_free_surface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  // density of free surface particles is not corrected
}

Particle::SPHDensityCorrectionNormalized::SPHDensityCorrectionNormalized()
    : Particle::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool Particle::SPHDensityCorrectionNormalized::compute_density_bc() const { return false; }

void Particle::SPHDensityCorrectionNormalized::corrected_density_free_surface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  dens[0] = denssum[0] / colorfield[0];
}

Particle::SPHDensityCorrectionRandles::SPHDensityCorrectionRandles()
    : Particle::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool Particle::SPHDensityCorrectionRandles::compute_density_bc() const { return true; }

void Particle::SPHDensityCorrectionRandles::corrected_density_free_surface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  dens[0] = denssum[0] + dens_bc[0] * (1.0 - colorfield[0]);
}

FOUR_C_NAMESPACE_CLOSE
