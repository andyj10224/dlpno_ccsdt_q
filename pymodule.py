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

import psi4
from psi4 import core
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

def run_dlpno_ccsdt_q(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    dlpno_ccsdt_q can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('dlpno_ccsdt_q')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.core.set_local_option('MYPLUGIN', 'PRINT', 1)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Ensure the DF basis set is loaded before calling the plugin
    aux_basis = core.BasisSet.build(ref_wfn.molecule(), "DF_BASIS_MP2",
                                    core.get_option("DFMP2", "DF_BASIS_MP2"),
                                    "RIFIT", core.get_global_option("BASIS"))
    ref_wfn.set_basisset("DF_BASIS_MP2", aux_basis)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    dlpno_ccsdt_q_wfn = psi4.core.plugin('dlpno_ccsdt_q.so', ref_wfn)

    return dlpno_ccsdt_q_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['dlpno_ccsdt_q'] = run_dlpno_ccsdt_q


def exampleFN():
    # Your Python code goes here
    pass