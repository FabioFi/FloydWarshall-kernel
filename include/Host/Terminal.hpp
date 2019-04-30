/**
 * @copyright Copyright © 2018 XLib. All rights reserved.
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
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date March, 2018
 * @version v1.5
 *
 * @file
 */
#pragma once

#include <ostream>

/**
 * @brief change the color of the output stream
 */
enum class Color {
    /** <table border="0"><tr><td><div> Red </div></td><td><div
       style="background:#FF0000;width:20px;height:20px;border:1px solid
       #000"></div></td></tr></table> */
    FG_RED = 31,
    /** <table border="0"><tr><td><div> Green </div></td><td><div
        style="background:#008000;width:20px;height:20px;border:1px
        solid #000"></div></td></tr></table> */
    FG_GREEN = 32,
    /** <table border="0"><tr><td><div> Yellow </div></td><td><div
        style="background:#FFFF00;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_YELLOW = 33,
    /** <table border="0"><tr><td><div> Blue </div></td><td><div
       style="background:#0000FF;width:20px;height:20px;border:1px
       solid #000"></div></td></tr></table> */
    FG_BLUE = 34,
    /** <table border="0"><tr><td><div> Magenta </div></td><td><div
        style="background:#FF00FF;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_MAGENTA = 35,
    /** <table border="0"><tr><td><div> Cyan </div></td><td><div
        style="background:#00FFFF;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_CYAN = 36,
    /** <table border="0"><tr><td><div> Light Gray </div></td><td><div
        style="background:#D3D3D3;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_GRAY = 37,
    /** <table border="0"><tr><td><div> Dark Gray </div></td><td><div
        style="background:#A9A9A9;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_D_GREY = 90,
    /** <table border="0"><tr><td><div> Light Red </div></td><td><div
        style="background:#DC143C;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_RED = 91,
    /** <table border="0"><tr><td><div> Light Green </div></td><td><div
        style="background:#90EE90;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_GREEN = 92,
    /** <table border="0"><tr><td><div> Light Yellow </div></td><td><div
        style="background:#FFFFE0;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_YELLOW = 93,
    /** <table border="0"><tr><td><div> Light Blue </div></td><td><div
        style="background:#ADD8E6;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_BLUE = 94,
    /** <table border="0"><tr><td><div> Light Magenta
       </div></td><td><div
       style="background:#EE82EE;width:20px;height:20px;border:1px
       solid #000"></div></td></tr></table> */
    FG_L_MAGENTA = 95,
    /** <table border="0"><tr><td><div> Light Cyan </div></td><td><div
        style="background:#E0FFFF;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_L_CYAN = 96,
    /** <table border="0"><tr><td><div> White </div></td><td><div
        style="background:#FFFFFF;width:20px;height:20px;border:1px solid
        #000"></div></td></tr></table> */
    FG_WHITE   = 97,
    /** Default */
    FG_DEFAULT = 39
};

/**
 * @brief Emph
 */
enum class Emph {
    SET_BOLD      = 1,
    SET_DIM       = 2,
    SET_UNDERLINE = 4,
    SET_RESET     = 0,
};

/// @cond
std::ostream& operator<<(std::ostream& os, Color mod);
std::ostream& operator<<(std::ostream& os, Emph mod);
/// @endcond

#include "impl/Terminal.i.hpp"
