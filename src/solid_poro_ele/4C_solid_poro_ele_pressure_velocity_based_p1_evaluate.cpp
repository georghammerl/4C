// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_interpolation.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_solid_ele_calc_lib.hpp"
#include "4C_solid_ele_factory.hpp"
#include "4C_solid_poro_ele_calc_lib.hpp"
#include "4C_solid_poro_ele_calc_lib_p1.hpp"
#include "4C_solid_poro_ele_pressure_velocity_based_p1.hpp"
#include "4C_solid_scatra_ele_factory.hpp"
#include "4C_utils_exceptions.hpp"

#include <type_traits>
#include <utility>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace
{

  template <typename T, typename Variant>
  struct IsMemberOfVariant;

  template <typename T, typename... Types>
  struct IsMemberOfVariant<T, std::variant<Types...>>
      : public std::disjunction<std::is_same<T, Types>...>
  {
  };

  template <typename T>
  concept IsSolidInterface =
      IsMemberOfVariant<std::remove_cvref_t<T>, Discret::Elements::SolidCalcVariant<2>>::value ||
      IsMemberOfVariant<std::remove_cvref_t<T>, Discret::Elements::SolidCalcVariant<3>>::value;

  template <typename T>
  concept IsSolidScatraInterface = IsMemberOfVariant<std::remove_cvref_t<T>,
                                       Discret::Elements::SolidScatraCalcVariant<2>>::value ||
                                   IsMemberOfVariant<std::remove_cvref_t<T>,
                                       Discret::Elements::SolidScatraCalcVariant<3>>::value;

  template <IsSolidScatraInterface SolidInterface>
  const auto& get_location_array(const Core::Elements::LocationArray& la)
  {
    // The solid-scatra interface expects all location arrays!
    return la;
  }

  template <IsSolidInterface SolidInterface>
  const auto& get_location_array(const Core::Elements::LocationArray& la)
  {
    // The solid interface only expects the first location array!
    return la[0].lm_;
  }

  template <unsigned dim>
  void add_element_averaged_scalars_to_params(
      Discret::Elements::SolidPoroPressureVelocityBasedP1<dim>& element,
      Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
      const std::vector<int>& lm)
  {
    FOUR_C_ASSERT(discretization.has_state(2, "scalar"),
        "The solid-poro-scatra element expects a scalar state to be defined in the "
        "discretization!");

    // check if you can get the scalar state
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
        discretization.get_state(2, "scalar");
    FOUR_C_ASSERT(scalarnp != nullptr, "The scalar state pointer is null in the discretization!");

    // extract local values of the global vectors
    std::vector<double> myscalar = Core::FE::extract_values(*scalarnp, lm);

    FOUR_C_ASSERT_ALWAYS(
        element.num_material() >= 3, "No third material defined for Solid-poro-scatra element!");
    std::shared_ptr<Core::Mat::Material> scatramat = element.material(2);

    int numscal = 1;
    if (scatramat->material_type() == Core::Materials::m_matlist or
        scatramat->material_type() == Core::Materials::m_matlist_reactions)
    {
      std::shared_ptr<Mat::MatList> matlist = std::dynamic_pointer_cast<Mat::MatList>(scatramat);
      numscal = matlist->num_mat();
    }

    const int num_nodes = Core::FE::num_nodes(element.shape());
    std::shared_ptr<std::vector<double>> scalar =
        std::make_shared<std::vector<double>>(numscal, 0.0);
    FOUR_C_ASSERT_ALWAYS(std::cmp_equal(myscalar.size(), numscal * num_nodes),
        "Sizes of number of scalars does not match!");

    for (int i = 0; i < num_nodes; i++)
    {
      for (int j = 0; j < numscal; j++)
      {
        scalar->at(j) += myscalar[numscal * i + j] / num_nodes;
      }
    }

    params.set("scalars", scalar);
  }

}  // namespace

template <unsigned dim>
int Discret::Elements::SolidPoroPressureVelocityBasedP1<dim>::evaluate(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}.", id());
  FOUR_C_ASSERT(solidporo_press_vel_based_calc_variant_.has_value(),
      "The solidporo calculation interface is not initialized for element id {}. The element needs "
      "to be fully setup before packing.",
      id());
  if (!material_post_setup_)
  {
    std::visit([&](auto& interface)
        { interface->material_post_setup(*this, struct_poro_material()); }, *solid_calc_variant_);
    material_post_setup_ = true;
  }

  if (discretization.has_state(0, "displacement") and (!initial_porosity_))
  {
    initial_porosity_ = Core::FE::extract_values(*discretization.get_state(0, "displacement"),
        get_reduced_porosity_location_array(
            la[0].lm_, Core::FE::get_number_of_element_volumes(shape()), dim));
  }

  // get ptr to interface to time integration
  set_params_interface_ptr(params);

  const Core::Elements::ActionType action = std::invoke(
      [&]()
      {
        if (is_params_interface())
          return params_interface().get_action_type();
        else
          return Core::Elements::string_to_action_type(params.get<std::string>("action", "none"));
      });

  const int num_nodes = Core::FE::get_number_of_element_nodes(shape());
  int num_displ_dofs = num_nodes * dim;

  switch (action)
  {
    case Core::Elements::struct_calc_nlnstiff:
    {
      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix K_dd(num_displ_dofs, num_displ_dofs);

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params, &force_displ, &K_dd, nullptr);
          },
          *solid_calc_variant_);

      if (la.size() > 1)
      {
        const bool with_scatra =
            poro_ele_property_.impltype != Inpar::ScaTra::ImplType::impltype_undefined;

        if (with_scatra)
        {
          FOUR_C_ASSERT(la.size() > 2, "For solid-poro-scatra, we expect 3 location arrays!");
          // We also have a scalar
          add_element_averaged_scalars_to_params(*this, params, discretization, la[2].lm_);
        }

        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        Core::LinAlg::SerialDenseMatrix K_dp(num_displ_dofs, num_nodes);
        Core::LinAlg::SerialDenseMatrix K_pd(num_nodes, num_displ_dofs);
        Core::LinAlg::SerialDenseMatrix K_pp(num_nodes, num_nodes);

        Core::LinAlg::SerialDenseMatrix reaction_matrix(num_displ_dofs, num_displ_dofs);

        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);

        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
                .K_displacement_displacement = &K_dd,
                .K_displacement_porosity = &K_dp,
                .K_porosity_displacement = &K_pd,
                .K_porosity_porosity = &K_pp,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, &reaction_matrix);
            },
            *solidporo_press_vel_based_calc_variant_);


        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);

        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
            SolidPoroDofType::porosity, dim>(elemat1, K_dp);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::displacement, dim>(elemat1, K_pd);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::porosity, dim>(elemat1, K_pp);

        if (elemat2.num_cols() > 0)
          assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
              SolidPoroDofType::displacement, dim>(elemat2, reaction_matrix);
      }

      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat1, K_dd);
      return 0;
    }
    case Core::Elements::struct_calc_internalforce:
    {
      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params, &force_displ, nullptr, nullptr);
          },
          *solid_calc_variant_);

      if (la.size() > 1)
      {
        const bool with_scatra =
            poro_ele_property_.impltype != Inpar::ScaTra::ImplType::impltype_undefined;

        if (with_scatra)
        {
          FOUR_C_ASSERT(la.size() > 2, "For solid-poro-scatra, we expect 3 location arrays!");
          // We also have a scalar
          add_element_averaged_scalars_to_params(*this, params, discretization, la[2].lm_);
        }

        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, nullptr);
            },
            *solidporo_press_vel_based_calc_variant_);

        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);
      }

      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      return 0;
    }
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      const bool with_scatra =
          poro_ele_property_.impltype != Inpar::ScaTra::ImplType::impltype_undefined;

      if (with_scatra)
      {
        FOUR_C_ASSERT(la.size() > 2, "For solid-poro-scatra, we expect 3 location arrays!");
        // We also have a scalar
        add_element_averaged_scalars_to_params(*this, params, discretization, la[2].lm_);
      }

      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix K_dd(num_displ_dofs, num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix M_dd(num_displ_dofs, num_displ_dofs);

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params, &force_displ, &K_dd, &M_dd);
          },
          *solid_calc_variant_);


      if (la.size() > 1 and this->num_material() > 1)
      {
        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        Core::LinAlg::SerialDenseMatrix K_dp(num_displ_dofs, num_nodes);
        Core::LinAlg::SerialDenseMatrix K_pd(num_nodes, num_displ_dofs);
        Core::LinAlg::SerialDenseMatrix K_pp(num_nodes, num_nodes);

        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
                .K_displacement_displacement = &K_dd,
                .K_displacement_porosity = &K_dp,
                .K_porosity_displacement = &K_pd,
                .K_porosity_porosity = &K_pp,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, nullptr);
            },
            *solidporo_press_vel_based_calc_variant_);

        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);

        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
            SolidPoroDofType::porosity, dim>(elemat1, K_dp);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::displacement, dim>(elemat1, K_pd);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::porosity, dim>(elemat1, K_pp);
      }


      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat1, K_dd);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat2, M_dd);

      return 0;
    }
    case Core::Elements::struct_poro_calc_scatracoupling:
    {
      // no coupling -> return
      return 0;
    }
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      const bool with_scatra =
          poro_ele_property_.impltype != Inpar::ScaTra::ImplType::impltype_undefined;

      if (with_scatra)
      {
        FOUR_C_ASSERT(la.size() > 2, "For solid-poro-scatra, we expect 3 location arrays!");
        // We also have a scalar
        add_element_averaged_scalars_to_params(*this, params, discretization, la[2].lm_);
      }

      Core::LinAlg::SerialDenseMatrix K_displ_fluid(num_displ_dofs, (dim + 1) * num_nodes);
      Core::LinAlg::SerialDenseMatrix K_por_pres(num_nodes, num_nodes);

      const SolidPoroPrimaryVariables primary_variables =
          extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


      SolidPoroOffDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
          off_diagonal_block_matrices{
              .K_displacement_fluid_dofs = &K_displ_fluid,
              .K_porosity_pressure = &K_por_pres,
          };

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_od(*this, this->struct_poro_material(),
                this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                this->kinematic_type(), discretization, primary_variables, params,
                off_diagonal_block_matrices);
          },
          *solidporo_press_vel_based_calc_variant_);

      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::none, dim>(elemat1, K_displ_fluid);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
          SolidPoroDofType::porosity, dim>(
          elemat1, K_por_pres);  // pressure is equally arranged as porosity

      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->update(*this, solid_poro_material(), discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params);
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->recover(*this, discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params);
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_stress:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->calculate_stress(*this, this->struct_poro_material(),
                StressIO{.type = get_io_stress_type(*this, params),
                    .mutable_data = get_stress_data(*this, params)},
                StrainIO{.type = get_io_strain_type(*this, params),
                    .mutable_data = get_strain_data(*this, params)},
                discretization,
                get_reduced_displacement_location_array(
                    get_location_array<decltype(interface)>(la), num_nodes, dim),
                params);
          },
          *solid_calc_variant_);

      const SolidPoroPrimaryVariables primary_variables =
          extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);

      return 0;
    }
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->initialize_gauss_point_data_output(*this, solid_poro_material(),
                *get_solid_params_interface().gauss_point_data_output_manager_ptr());
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_gauss_point_data_output(*this, solid_poro_material(),
                *get_solid_params_interface().gauss_point_data_output_manager_ptr());
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_predict:
    case Core::Elements::none:
    {
      // do nothing for now
      return 0;
    }
    default:
      FOUR_C_THROW("The element action {} is not yet implemented for the new solid elements",
          action_type_to_string(action));
      // do nothing (no error because there are some actions the poro element is supposed to
      // ignore)
      return 0;
  }
}


template <unsigned dim>
int Discret::Elements::SolidPoroPressureVelocityBasedP1<dim>::evaluate_neumann(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const Core::Conditions::Condition& condition, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW(
      "Cannot yet evaluate volume neumann forces within the pressure-velocity based poro p1 "
      "implementation.");
}


template int Discret::Elements::SolidPoroPressureVelocityBasedP1<2>::evaluate(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3);
template int Discret::Elements::SolidPoroPressureVelocityBasedP1<2>::evaluate_neumann(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const Core::Conditions::Condition& condition, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1);

template int Discret::Elements::SolidPoroPressureVelocityBasedP1<3>::evaluate(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3);
template int Discret::Elements::SolidPoroPressureVelocityBasedP1<3>::evaluate_neumann(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const Core::Conditions::Condition& condition, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1);

FOUR_C_NAMESPACE_CLOSE
