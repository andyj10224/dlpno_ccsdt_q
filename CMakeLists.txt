#
# @BEGIN LICENSE
#
# dlpno_ccsdt_q by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(dlpno_ccsdt_q CXX)

set(Einsums_DIR /home/aj48384/github/EinsumsInCpp/install/share/cmake/Einsums)
set(psi4_DIR /home/aj48384/github/psi4/objdir_p4sums/stage/share/cmake/psi4)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(psi4 1.1 REQUIRED)
find_package(Einsums REQUIRED)

add_psi4_plugin(dlpno_ccsdt_q plugin.cc)
#target_compile_definitions(dlpno_ccsdt_q PUBLIC EINSUMS_TEST_NANS EINSUMS_ONLY_USE_GENERIC_ALGORITHM)
target_link_libraries(dlpno_ccsdt_q PUBLIC Einsums::einsums)
