// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_ele_factory.hpp"

#include "4C_solid_poro_ele_calc_lib.hpp"

FOUR_C_NAMESPACE_OPEN



template <unsigned dim>
Discret::Elements::SolidPoroPressureBasedCalcVariant<dim>
Discret::Elements::create_solid_poro_pressure_based_calculation_interface(
    const Discret::Elements::SolidElementProperties<dim>& element_properties,
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<
      Discret::Elements::Internal::ImplementedSolidPoroCellTypes<dim>>(celltype,
      [&](auto celltype_t)
      {
        return create_solid_poro_pressure_based_calculation_interface<celltype_t()>(
            element_properties);
      });
}

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureBasedCalcVariant<Core::FE::dim<celltype>>
Discret::Elements::create_solid_poro_pressure_based_calculation_interface(
    const Discret::Elements::SolidElementProperties<Core::FE::dim<celltype>>& element_properties)
{
  if constexpr (Core::FE::dim<celltype> == 2)
  {
    return SolidPoroPressureBasedEleCalc<celltype>(
        element_properties.reference_thickness, element_properties.plane_assumption);
  }
  else
  {
    return SolidPoroPressureBasedEleCalc<celltype>();
  }
}

template <unsigned dim>
Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant<dim>
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface(
    const Discret::Elements::SolidElementProperties<dim>& element_properties,
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<
      Discret::Elements::Internal::ImplementedSolidPoroCellTypes<dim>>(celltype,
      [&](auto celltype_t)
      {
        return create_solid_poro_pressure_velocity_based_calculation_interface<celltype_t()>(
            element_properties);
      });
}

template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant<Core::FE::dim<celltype>>
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface(
    const Discret::Elements::SolidElementProperties<Core::FE::dim<celltype>>& element_properties)
{
  if constexpr (Core::FE::dim<celltype> == 2)
  {
    return SolidPoroPressureVelocityBasedEleCalc<celltype, PorosityFormulation::from_material_law>(
        element_properties.reference_thickness, element_properties.plane_assumption);
  }
  else
  {
    return SolidPoroPressureVelocityBasedEleCalc<celltype,
        PorosityFormulation::from_material_law>();
  }
}

template <unsigned dim>
Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant<dim>
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface(
    const Discret::Elements::SolidElementProperties<dim>& element_properties,
    Core::FE::CellType celltype)
{
  return Core::FE::cell_type_switch<
      Discret::Elements::Internal::ImplementedSolidPoroCellTypes<dim>>(celltype,
      [&](auto celltype_t)
      {
        return create_solid_poro_pressure_velocity_based_p1_calculation_interface<celltype_t()>(
            element_properties);
      });
}


template <Core::FE::CellType celltype>
Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant<Core::FE::dim<celltype>>
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface(
    const Discret::Elements::SolidElementProperties<Core::FE::dim<celltype>>& element_properties)
{
  if constexpr (Core::FE::dim<celltype> == 2)
  {
    return SolidPoroPressureVelocityBasedEleCalc<celltype,
        PorosityFormulation::as_primary_variable>(
        element_properties.reference_thickness, element_properties.plane_assumption);
  }
  else
  {
    return SolidPoroPressureVelocityBasedEleCalc<celltype,
        PorosityFormulation::as_primary_variable>();
  }
}

template Discret::Elements::SolidPoroPressureBasedCalcVariant<2>
Discret::Elements::create_solid_poro_pressure_based_calculation_interface<2>(
    const Discret::Elements::SolidElementProperties<2>&, Core::FE::CellType celltype);
template Discret::Elements::SolidPoroPressureBasedCalcVariant<3>
Discret::Elements::create_solid_poro_pressure_based_calculation_interface<3>(
    const Discret::Elements::SolidElementProperties<3>&, Core::FE::CellType celltype);

template Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant<2>
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface<2>(
    const Discret::Elements::SolidElementProperties<2>&, Core::FE::CellType celltype);
template Discret::Elements::SolidPoroPressureVelocityBasedCalcVariant<3>
Discret::Elements::create_solid_poro_pressure_velocity_based_calculation_interface<3>(
    const Discret::Elements::SolidElementProperties<3>&, Core::FE::CellType celltype);

template Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant<2>
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface<2>(
    const Discret::Elements::SolidElementProperties<2>&, Core::FE::CellType celltype);
template Discret::Elements::SolidPoroPressureVelocityBasedP1CalcVariant<3>
Discret::Elements::create_solid_poro_pressure_velocity_based_p1_calculation_interface<3>(
    const Discret::Elements::SolidElementProperties<3>&, Core::FE::CellType celltype);

FOUR_C_NAMESPACE_CLOSE
