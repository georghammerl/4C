// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_PROXY_TYPES_HPP
#define FOUR_C_IO_PROXY_TYPES_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"

#include <concepts>
#include <sstream>


FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * A specialized type @p T that can be described as using another type.
   *
   * This is useful to describe that the IO-rules of a type can be described with the rules of other
   * primitive types. For example, a LinAlg::Tensor can be described as a nested array of its value
   * type.
   */
  template <typename T>
  struct ProxyType;

  /**
   * A proxy-type for LinAlg::Tensor that uses nested std::array as the underlying type.
   */
  template <typename T, std::size_t... n>
  struct ProxyType<Core::LinAlg::Tensor<T, n...>>
  {
    template <std::size_t... m>
    struct NestedArrayType;

    template <std::size_t m1, std::size_t... m>
    struct NestedArrayType<m1, m...>
    {
      using type = std::array<typename NestedArrayType<m...>::type, m1>;
    };

    template <std::size_t m>
    struct NestedArrayType<m>
    {
      using type = std::array<T, m>;
    };

    using OriginalType = Core::LinAlg::Tensor<T, n...>;
    using type = NestedArrayType<n...>::type;

    static inline std::string pretty_name()
    {
      std::ostringstream oss;
      LinAlg::print_pretty_tensor_name<T, n...>(oss, "Tensor");
      return oss.str();
    }

    static constexpr inline OriginalType to_value(const type& v)
    {
      return Core::LinAlg::make_tensor_from_nested_array<T, n...>(v);
    }

    static constexpr inline type from_value(const OriginalType& v)
    {
      return make_nested_array_from_tensor(v);
    }
  };

  /**
   * A proxy-type for LinAlg::SymmetricTensor that is based on the LinAlg::Tensor type.
   */
  template <typename T, std::size_t... n>
  struct ProxyType<Core::LinAlg::SymmetricTensor<T, n...>>
  {
    using OriginalType = Core::LinAlg::SymmetricTensor<T, n...>;
    using type = Core::LinAlg::Tensor<T, n...>;

    static inline std::string pretty_name()
    {
      std::ostringstream oss;
      LinAlg::print_pretty_tensor_name<T, n...>(oss, "SymmetricTensor");
      return oss.str();
    }

    static constexpr inline OriginalType to_value(const type& v)
    {
      // validate whether the tensor is symmetric
      FOUR_C_ASSERT_ALWAYS(Core::LinAlg::is_symmetric(v),
          "Requiring a symmetric tensor, but given tensor {} is not symmetric.",
          [&]()
          {
            std::ostringstream oss;
            oss << v;
            return oss.str();
          }());

      return Core::LinAlg::assume_symmetry(v);
    }

    static constexpr inline type from_value(const OriginalType& v)
    {
      return Core::LinAlg::get_full(v);
    }
  };

  /**
   * A concept that checks whether a type is a proxy-type of another type.
   */
  template <typename T>
  concept ProxyTypeConcept = requires() {
    { ProxyType<T>::to_value(std::declval<typename ProxyType<T>::type>()) } -> std::same_as<T>;
    { ProxyType<T>::from_value(std::declval<T>()) } -> std::same_as<typename ProxyType<T>::type>;
    { ProxyType<T>::pretty_name() } -> std::convertible_to<std::string>;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
