// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <type_traits>

namespace smooth::feedback::traits {

template<typename T, template<typename...> class Z>
struct is_specialization_of : std::false_type
{};

template<typename... Args, template<typename...> class Z>
struct is_specialization_of<Z<Args...>, Z> : std::true_type
{};

template<typename T, template<typename...> class Z>
static constexpr bool is_specialization_of_v = is_specialization_of<T, Z>::value;

template<typename T, template<std::size_t...> class Z>
struct is_specialization_of_sizet : std::false_type
{};

template<std::size_t... Args, template<std::size_t...> class Z>
struct is_specialization_of_sizet<Z<Args...>, Z> : std::true_type
{};

template<typename T, template<std::size_t...> class Z>
static constexpr bool is_specialization_of_sizet_v = is_specialization_of_sizet<T, Z>::value;

}  // namespace smooth::feedback::traits
