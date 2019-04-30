/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "HostDevice.hpp"
#include <iostream> // std::numpunct
#include <string>   // std::string

namespace xlib {

struct myseps : std::numpunct<char> {
private:
    char do_thousands_sep()   const noexcept final;
    std::string do_grouping() const noexcept final;
};

class ThousandSep {
public:
    ThousandSep();
    ~ThousandSep();

    ThousandSep(const ThousandSep&)    = delete;
    void operator=(const ThousandSep&) = delete;
private:
    myseps* sep { nullptr };
};

template<typename T>
std::string format(T num, unsigned precision = 1) noexcept;

std::string human_readable(size_t size) noexcept;

void fixed_float() noexcept;
void scientific_float() noexcept;

class IosFlagSaver {
public:
    IosFlagSaver()  noexcept;
    ~IosFlagSaver() noexcept;
    IosFlagSaver(const IosFlagSaver &rhs)             = delete;
    IosFlagSaver& operator= (const IosFlagSaver& rhs) = delete;
private:
    std::ios::fmtflags _flags;
    std::streamsize    _precision;
};
//------------------------------------------------------------------------------

void char_sequence(char c, int sequence_length = 80) noexcept;

void printTitle(const std::string& title, char c = '-',
                int sequence_length = 80) noexcept;
//------------------------------------------------------------------------------

/**
 * @brief
 */
template<typename T, int SIZE>
void printArray(T (&array)[SIZE], const std::string& title = "",
                const std::string& sep = " ") noexcept;

/**
 * @brief
 */
template<typename T>
void printArray(const T* array, size_t size, const std::string& title = "",
                const std::string& sep = " ") noexcept;

/**
 * @brief
 * @deprecated
 */
template<typename T>
[[deprecated("pointer of pointer")]]
void printMatrix(T* const* matrix, size_t rows, size_t cols,
                 const std::string& title = "") noexcept;

/**
 * @brief row-major
 */
template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols,
                 const std::string& title = "") noexcept;

/**
 * @brief row-major
 */
template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols, size_t ld,
                 const std::string& title = "") noexcept;

/**
 * @brief column-major (blas and lapack compatibility)
 */
template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols,
                   const std::string& title = "") noexcept;

/**
 * @brief column-major (blas and lapack compatibility)
 */
template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols, size_t ld,
                   const std::string& title = "") noexcept;

//------------------------------------------------------------------------------

/**
 * @brief left to right : char v = 1 -> 10000000
 */
template<typename T>
HOST_DEVICE void
printBits(T* array, int size);

template<typename T>
HOST_DEVICE void
printBits(const T& value);

} // namespace xlib

#include "impl/PrintExt.i.hpp"
