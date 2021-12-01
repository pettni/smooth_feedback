// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__FEEDBACK__TRAITS_HPP_
#define SMOOTH__FEEDBACK__TRAITS_HPP_

#include <type_traits>

namespace smooth::feedback::traits {

template<typename T, template<typename...> class Z>
struct is_specialization_of : std::false_type
{};

template<typename... Args, template<typename...> class Z>
struct is_specialization_of<Z<Args...>, Z> : std::true_type
{};

template<typename T, template<typename...> class Z>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Z>::value;

template<typename T, template<std::size_t...> class Z>
struct is_specialization_of_sizet : std::false_type
{};

template<std::size_t... Args, template<std::size_t...> class Z>
struct is_specialization_of_sizet<Z<Args...>, Z> : std::true_type
{};

template<typename T, template<std::size_t...> class Z>
inline constexpr bool is_specialization_of_sizet_v = is_specialization_of_sizet<T, Z>::value;

}  // namespace smooth::feedback::traits

#endif  // SMOOTH__FEEDBACK__TRAITS_HPP_
