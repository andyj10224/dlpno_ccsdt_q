/*
 * @BEGIN LICENSE
 *
 * dlpno_ccsdt_q by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/molecule.h"
#include "psi4/lib3index/dfhelper.h"

#include "/home/aj48384/github/psi4/psi4/src/psi4/dlpno/dlpno.h"
#include "/home/aj48384/github/psi4/psi4/src/psi4/dlpno/sparse.h"

#include <memory>

#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Print.hpp"

using namespace psi;
using namespace psi::dlpno;

using namespace einsums;
using namespace einsums::linear_algebra;
using namespace einsums::tensor_algebra;
using namespace einsums::tensor_algebra::index;

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi{ 
namespace dlpno{
namespace dlpno_ccsdt_q {

class DLPNOCCSDT : public DLPNOCCSD_T {
   protected:
    // K_ooov integrals
    std::vector<Tensor<double, 2>> K_iojv_;
    std::vector<Tensor<double, 2>> K_joiv_;
    std::vector<Tensor<double, 2>> K_jokv_;
    std::vector<Tensor<double, 2>> K_kojv_;
    std::vector<Tensor<double, 2>> K_iokv_;
    std::vector<Tensor<double, 2>> K_koiv_;

    // K_ovov integrals
    std::vector<Tensor<double, 2>> K_ivjv_;
    std::vector<Tensor<double, 2>> K_jvkv_;
    std::vector<Tensor<double, 2>> K_ivkv_;

    std::vector<Tensor<double, 3>> K_ivov_;
    std::vector<Tensor<double, 3>> K_jvov_;
    std::vector<Tensor<double, 3>> K_kvov_;

    // K_ovvv integrals
    std::vector<Tensor<double, 3>> K_ivvv_;
    std::vector<Tensor<double, 3>> K_jvvv_;
    std::vector<Tensor<double, 3>> K_kvvv_;

    // DF integrals in the domain of triplet ijk
    std::vector<Tensor<double, 2>> q_io_;
    std::vector<Tensor<double, 2>> q_jo_;
    std::vector<Tensor<double, 2>> q_ko_;

    std::vector<Tensor<double, 2>> q_iv_;
    std::vector<Tensor<double, 2>> q_jv_;
    std::vector<Tensor<double, 2>> q_kv_;

    std::vector<Tensor<double, 3>> q_ov_;
    std::vector<Tensor<double, 3>> q_vv_;

    // Singles Amplitudes
    std::vector<Tensor<double, 2>> T_n_ijk_;
    // Contravariant Triples Amplitudes (from Koch 2020)
    std::vector<Tensor<double, 3>> T_iajbkc_tilde_;

    // Number of threads
    int nthread_;
    // Energy expression
    double e_lccsdt_;
    
    /// computes singles residuals in LCCSDT equations
    void compute_R_ia_triples(std::vector<SharedMatrix>& R_ia, std::vector<std::vector<SharedMatrix>>& R_ia_buffer);
    /// compute doubles residuals in LCCSDT equations
    void compute_R_iajb_triples(std::vector<SharedMatrix>& R_iajb, std::vector<SharedMatrix>& Rn_iajb, std::vector<std::vector<SharedMatrix>>& R_iajb_buffer);
    /// computes triples residuals in LCCSDT equations
    void compute_R_iajbkc(std::vector<SharedMatrix>& R_iajbkc);

    void print_header();
    void estimate_memory();
    void compute_integrals();
    void lccsdt_iterations();
    void print_results();

   public:
    DLPNOCCSDT(SharedWavefunction ref_wfn, Options& options);
    ~DLPNOCCSDT() override;

    double compute_energy() override;
};

DLPNOCCSDT::DLPNOCCSDT(SharedWavefunction ref_wfn, Options& options) : DLPNOCCSD_T(ref_wfn, options) {}
DLPNOCCSDT::~DLPNOCCSDT() {}

void DLPNOCCSDT::print_header() {
    double t_cut_tno = options_.get_double("T_CUT_TNO");
    double t_cut_tno_strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    double t_cut_tno_weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");

    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                    DLPNO-CCSDT                \n");
    outfile->Printf("                   by Andy Jiang               \n");
    outfile->Printf("   --------------------------------------------\n\n");
    outfile->Printf("  DLPNO convergence set to %s.\n\n", options_.get_str("PNO_CONVERGENCE").c_str());
    outfile->Printf("  Detailed DLPNO thresholds and cutoffs:\n");
    outfile->Printf("    T_CUT_TNO_STRONG           = %6.3e \n", t_cut_tno * t_cut_tno_strong_scale);
    outfile->Printf("    T_CUT_TNO_WEAK             = %6.3e \n", t_cut_tno * t_cut_tno_weak_scale);
    outfile->Printf("    T_CUT_DO_TRIPLES           = %6.3e \n", options_.get_double("T_CUT_DO_TRIPLES"));
    outfile->Printf("    T_CUT_MKN_TRIPLES          = %6.3e \n", options_.get_double("T_CUT_MKN_TRIPLES"));
    outfile->Printf("    F_CUT_T                    = %6.3e \n", options_.get_double("F_CUT_T"));
}

void DLPNOCCSDT::estimate_memory() {
    outfile->Printf("\n ==> DLPNO-CCSDT Memory Estimate <== \n\n");

    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    size_t K_iojv_memory = 0;
    size_t K_ivov_memory = 0;
    size_t K_ivvv_memory = 0;
    size_t qij_memory = 0;
    size_t qia_memory = 0;
    size_t qab_memory = 0;

#pragma omp parallel for reduction(+ : K_iojv_memory, K_ivov_memory, K_ivvv_memory, qij_memory, qia_memory, qab_memory)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {

        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        int npao_ijk = lmotriplet_to_paos_[ijk].size();
        int ntno_ijk = n_tno_[ijk];

        K_iojv_memory += 6 * n_lmo_triplets * nlmo_ijk * ntno_ijk;
        K_ivov_memory += 3 * n_lmo_triplets * nlmo_ijk * ntno_ijk * ntno_ijk;
        K_ivvv_memory += 3 * n_lmo_triplets * ntno_ijk * ntno_ijk * ntno_ijk;
        qij_memory += 3 * n_lmo_triplets * naux_ijk * nlmo_ijk;
        qia_memory += 3 * n_lmo_triplets * naux_ijk * nlmo_ijk * ntno_ijk;
        qab_memory += n_lmo_triplets * naux_ijk * ntno_ijk * ntno_ijk;
    }

    size_t total_memory = K_iojv_memory + K_ivov_memory + K_ivvv_memory + qij_memory + qia_memory + qab_memory;

    outfile->Printf("    (i j | l_{ijk} a_{ijk})        : %.3f [GiB]\n", K_iojv_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (i a_{ijk} | l_{ijk} b_{ijk})  : %.3f [GiB]\n", K_ivov_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (i a_{ijk} | b_{ijk} c_{ijk})  : %.3f [GiB]\n", K_ivvv_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | l_{ijk} [i,j,k])    : %.3f [GiB]\n", qij_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | l_{ijk} a_{ijk})    : %.3f [GiB]\n", qia_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | a_{ijk} b_{ijk})    : %.3f [GiB]\n", qab_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    Total Memory Required          : %.3f [GiB]\n\n", total_memory * pow(2.0, -30) * sizeof(double));
}

void DLPNOCCSDT::compute_integrals() {

    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    // Four-center quantities (built from three-center)
    K_iojv_.resize(n_lmo_triplets);
    K_joiv_.resize(n_lmo_triplets);
    K_jokv_.resize(n_lmo_triplets);
    K_kojv_.resize(n_lmo_triplets);
    K_iokv_.resize(n_lmo_triplets);
    K_koiv_.resize(n_lmo_triplets);

    K_ivjv_.resize(n_lmo_triplets);
    K_jvkv_.resize(n_lmo_triplets);
    K_ivkv_.resize(n_lmo_triplets);

    K_ivov_.resize(n_lmo_triplets);
    K_jvov_.resize(n_lmo_triplets);
    K_kvov_.resize(n_lmo_triplets);

    K_ivvv_.resize(n_lmo_triplets);
    K_jvvv_.resize(n_lmo_triplets);
    K_kvvv_.resize(n_lmo_triplets);

    // Three-center quantities
    q_io_.resize(n_lmo_triplets);
    q_jo_.resize(n_lmo_triplets);
    q_ko_.resize(n_lmo_triplets);

    q_iv_.resize(n_lmo_triplets);
    q_jv_.resize(n_lmo_triplets);
    q_kv_.resize(n_lmo_triplets);

    q_ov_.resize(n_lmo_triplets);
    q_vv_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        int ntno_ijk = n_tno_[ijk];

        if (ntno_ijk == 0) continue;

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // => Compute all necessary integrals <= //

        // number of auxiliary functions in the triplet domain
        const int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        // number of LMOs in the triplet domain
        const int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        // number of PAOs in the triplet domain (before removing linear dependencies)
        const int npao_ijk = lmotriplet_to_paos_[ijk].size();

        auto q_io = std::make_shared<Matrix>("(Q_ijk | m i)", naux_ijk, nlmo_ijk);
        auto q_jo = std::make_shared<Matrix>("(Q_ijk | m j)", naux_ijk, nlmo_ijk);
        auto q_ko = std::make_shared<Matrix>("(Q_ijk | m k)", naux_ijk, nlmo_ijk);

        auto q_iv = std::make_shared<Matrix>("(Q_ijk | i a)", naux_ijk, npao_ijk);
        auto q_jv = std::make_shared<Matrix>("(Q_ijk | j b)", naux_ijk, npao_ijk);
        auto q_kv = std::make_shared<Matrix>("(Q_ijk | k c)", naux_ijk, npao_ijk);

        auto q_ov = std::make_shared<Matrix>("(Q_ijk | m a)", naux_ijk, nlmo_ijk * ntno_ijk);
        auto q_vv = std::make_shared<Matrix>("(Q_ijk | a b)", naux_ijk, ntno_ijk * ntno_ijk);

        for (int q_ijk = 0; q_ijk < naux_ijk; q_ijk++) {
            const int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            const int centerq = ribasis_->function_to_center(q);

            // Cheaper Integrals
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                (*q_io)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_lmos_ext_dense_[centerq][l]);
                (*q_jo)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_lmos_ext_dense_[centerq][l]);
                (*q_ko)(q_ijk, l_ijk) = (*qij_[q])(riatom_to_lmos_ext_dense_[centerq][k], riatom_to_lmos_ext_dense_[centerq][l]);
            }


            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                int u = lmotriplet_to_paos_[ijk][u_ijk];
                (*q_iv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_paos_ext_dense_[centerq][u]);
                (*q_jv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_paos_ext_dense_[centerq][u]);
                (*q_kv)(q_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][k], riatom_to_paos_ext_dense_[centerq][u]);
            }

            // More expensive integrals
            auto q_ov_tmp = std::make_shared<Matrix>(nlmo_ijk, npao_ijk);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                    int u = lmotriplet_to_paos_[ijk][u_ijk];
                    (*q_ov_tmp)(l_ijk, u_ijk) = (*qia_[q])(riatom_to_lmos_ext_dense_[centerq][l], riatom_to_paos_ext_dense_[centerq][u]);
                }
            }
            q_ov_tmp = linalg::doublet(q_ov_tmp, X_tno_[ijk], false, false);
            ::memcpy(&(*q_ov)(q_ijk, 0), &(*q_ov_tmp)(0, 0), nlmo_ijk * ntno_ijk * sizeof(double));

            auto q_vv_tmp = std::make_shared<Matrix>(npao_ijk, npao_ijk);

            for (int u_ijk = 0; u_ijk < npao_ijk; ++u_ijk) {
                int u = lmotriplet_to_paos_[ijk][u_ijk];
                for (int v_ijk = 0; v_ijk < npao_ijk; ++v_ijk) {
                    int v = lmotriplet_to_paos_[ijk][v_ijk];
                    int uv_idx = riatom_to_pao_pairs_dense_[centerq][u][v];
                    if (uv_idx == -1) continue;
                    (*q_vv_tmp)(u_ijk, v_ijk) = (*qab_[q])(uv_idx, 0);
                } // end v_ijk
            } // end u_ijk
            q_vv_tmp = linalg::triplet(X_tno_[ijk], q_vv_tmp, X_tno_[ijk], true, false, false);
            ::memcpy(&(*q_vv)(q_ijk, 0), &(*q_vv_tmp)(0, 0), ntno_ijk * ntno_ijk * sizeof(double));

        } // end q_ijk

        // Contract Intermediates
        q_iv = linalg::doublet(q_iv, X_tno_[ijk]);
        q_jv = linalg::doublet(q_jv, X_tno_[ijk]);
        q_kv = linalg::doublet(q_kv, X_tno_[ijk]);
        
        // Multiply by (P|Q)^{-1/2}
        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_[ijk], lmotriplet_to_ribfs_[ijk]);
        A_solve->power(0.5, 1.0e-14);

        C_DGESV_wrapper(A_solve->clone(), q_io);
        C_DGESV_wrapper(A_solve->clone(), q_jo);
        C_DGESV_wrapper(A_solve->clone(), q_ko);
        C_DGESV_wrapper(A_solve->clone(), q_iv);
        C_DGESV_wrapper(A_solve->clone(), q_jv);
        C_DGESV_wrapper(A_solve->clone(), q_kv);
        C_DGESV_wrapper(A_solve->clone(), q_ov);
        C_DGESV_wrapper(A_solve->clone(), q_vv);

        q_io_[ijk] = Tensor<double, 2>("(Q_ijk | m i)", naux_ijk, nlmo_ijk);
        q_jo_[ijk] = Tensor<double, 2>("(Q_ijk | m j)", naux_ijk, nlmo_ijk);
        q_ko_[ijk] = Tensor<double, 2>("(Q_ijk | m k)", naux_ijk, nlmo_ijk);

        q_iv_[ijk] = Tensor<double, 2>("(Q_ijk | i a)", naux_ijk, ntno_ijk);
        q_jv_[ijk] = Tensor<double, 2>("(Q_ijk | j b)", naux_ijk, ntno_ijk);
        q_kv_[ijk] = Tensor<double, 2>("(Q_ijk | k c)", naux_ijk, ntno_ijk);

        q_ov_[ijk] = Tensor<double, 3>("(Q_ijk | m a)", naux_ijk, nlmo_ijk, ntno_ijk);
        q_vv_[ijk] = Tensor<double, 3>("(Q_ijk | a b)", naux_ijk, ntno_ijk, ntno_ijk);

        ::memcpy(q_io_[ijk].data(), q_io->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        ::memcpy(q_jo_[ijk].data(), q_jo->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        ::memcpy(q_ko_[ijk].data(), q_ko->get_pointer(), naux_ijk * nlmo_ijk * sizeof(double));
        ::memcpy(q_iv_[ijk].data(), q_iv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        ::memcpy(q_jv_[ijk].data(), q_jv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        ::memcpy(q_kv_[ijk].data(), q_kv->get_pointer(), naux_ijk * ntno_ijk * sizeof(double));
        ::memcpy(q_ov_[ijk].data(), q_ov->get_pointer(), naux_ijk * nlmo_ijk * ntno_ijk * sizeof(double));
        ::memcpy(q_vv_[ijk].data(), q_vv->get_pointer(), naux_ijk * ntno_ijk * ntno_ijk * sizeof(double));

        K_iojv_[ijk] = Tensor<double, 2>("K_iojv", nlmo_ijk, ntno_ijk);
        K_joiv_[ijk] = Tensor<double, 2>("K_joiv", nlmo_ijk, ntno_ijk);
        K_jokv_[ijk] = Tensor<double, 2>("K_jokv", nlmo_ijk, ntno_ijk);
        K_kojv_[ijk] = Tensor<double, 2>("K_kojv", nlmo_ijk, ntno_ijk);
        K_iokv_[ijk] = Tensor<double, 2>("K_iokv", nlmo_ijk, ntno_ijk);
        K_koiv_[ijk] = Tensor<double, 2>("K_koiv", nlmo_ijk, ntno_ijk);

        einsum(0.0, Indices{index::l, index::a}, &K_iojv_[ijk], 1.0, Indices{index::Q, index::l}, q_io_[ijk], Indices{index::Q, index::a}, q_jv_[ijk]);
        einsum(0.0, Indices{index::l, index::a}, &K_joiv_[ijk], 1.0, Indices{index::Q, index::l}, q_jo_[ijk], Indices{index::Q, index::a}, q_iv_[ijk]);
        einsum(0.0, Indices{index::l, index::a}, &K_jokv_[ijk], 1.0, Indices{index::Q, index::l}, q_jo_[ijk], Indices{index::Q, index::a}, q_kv_[ijk]);
        einsum(0.0, Indices{index::l, index::a}, &K_kojv_[ijk], 1.0, Indices{index::Q, index::l}, q_ko_[ijk], Indices{index::Q, index::a}, q_jv_[ijk]);
        einsum(0.0, Indices{index::l, index::a}, &K_iokv_[ijk], 1.0, Indices{index::Q, index::l}, q_io_[ijk], Indices{index::Q, index::a}, q_kv_[ijk]);
        einsum(0.0, Indices{index::l, index::a}, &K_koiv_[ijk], 1.0, Indices{index::Q, index::l}, q_ko_[ijk], Indices{index::Q, index::a}, q_iv_[ijk]);

        K_ivjv_[ijk] = Tensor<double, 2>("K_ivjv", ntno_ijk, ntno_ijk);
        K_jvkv_[ijk] = Tensor<double, 2>("K_jvkv", ntno_ijk, ntno_ijk);
        K_ivkv_[ijk] = Tensor<double, 2>("K_ivkv", ntno_ijk, ntno_ijk);

        einsum(0.0, Indices{index::a, index::b}, &K_ivjv_[ijk], 1.0, Indices{index::Q, index::a}, q_iv_[ijk], Indices{index::Q, index::b}, q_jv_[ijk]);
        einsum(0.0, Indices{index::a, index::b}, &K_jvkv_[ijk], 1.0, Indices{index::Q, index::a}, q_jv_[ijk], Indices{index::Q, index::b}, q_kv_[ijk]);
        einsum(0.0, Indices{index::a, index::b}, &K_ivkv_[ijk], 1.0, Indices{index::Q, index::a}, q_iv_[ijk], Indices{index::Q, index::b}, q_kv_[ijk]);

        K_ivov_[ijk] = Tensor<double, 3>("K_ivov", nlmo_ijk, ntno_ijk, ntno_ijk);
        K_jvov_[ijk] = Tensor<double, 3>("K_jvov", nlmo_ijk, ntno_ijk, ntno_ijk);
        K_kvov_[ijk] = Tensor<double, 3>("K_kvov", nlmo_ijk, ntno_ijk, ntno_ijk);

        einsum(0.0, Indices{index::m, index::a, index::b}, &K_ivov_[ijk], 1.0, Indices{index::Q, index::m, index::a}, q_ov_[ijk], 
                    Indices{index::Q, index::b}, q_iv_[ijk]);
        einsum(0.0, Indices{index::m, index::a, index::b}, &K_jvov_[ijk], 1.0, Indices{index::Q, index::m, index::a}, q_ov_[ijk],
                    Indices{index::Q, index::b}, q_jv_[ijk]);
        einsum(0.0, Indices{index::m, index::a, index::b}, &K_kvov_[ijk], 1.0, Indices{index::Q, index::m, index::a}, q_ov_[ijk],
                    Indices{index::Q, index::b}, q_kv_[ijk]);

        K_ivvv_[ijk] = Tensor<double, 3>("K_ivvv", ntno_ijk, ntno_ijk, ntno_ijk);
        K_jvvv_[ijk] = Tensor<double, 3>("K_jvvv", ntno_ijk, ntno_ijk, ntno_ijk);
        K_kvvv_[ijk] = Tensor<double, 3>("K_kvvv", ntno_ijk, ntno_ijk, ntno_ijk);

        einsum(0.0, Indices{index::a, index::b, index::c}, &K_ivvv_[ijk], 1.0, Indices{index::Q, index::c}, q_iv_[ijk], 
                    Indices{index::Q, index::a, index::b}, q_vv_[ijk]);
        einsum(0.0, Indices{index::a, index::b, index::c}, &K_jvvv_[ijk], 1.0, Indices{index::Q, index::c}, q_jv_[ijk],
                    Indices{index::Q, index::a, index::b}, q_vv_[ijk]);
        einsum(0.0, Indices{index::a, index::b, index::c}, &K_kvvv_[ijk], 1.0, Indices{index::Q, index::c}, q_kv_[ijk],
                    Indices{index::Q, index::a, index::b}, q_vv_[ijk]);
    } // end ijk
}

void DLPNOCCSDT::compute_R_ia_triples(std::vector<SharedMatrix>& R_ia, std::vector<std::vector<SharedMatrix>>& R_ia_buffer) {
    
    size_t naocc = i_j_to_ij_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    // Compute residual from LCCSD
    DLPNOCCSD::compute_R_ia(R_ia, R_ia_buffer);

    // Clean buffers
#pragma omp parallel for
    for (int thread = 0; thread < nthread_; ++thread) {
        for (int i = 0; i < naocc; ++i) {
            R_ia_buffer[thread][i]->zero();
        } // end thread
    } // end i

    // Looped over unique triplets (Koch Algorithm 1)
#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], kk = i_j_to_ij_[k][k];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        double prefactor = 1.0;
        if (i == j || j == k) {
            prefactor = 0.5;
        }

        std::vector<std::tuple<int, int, int>> P_S = {std::make_tuple(i, j, k), std::make_tuple(j, i, k), std::make_tuple(k, i, j)};
        std::vector<Tensor<double, 2>> K_ovov_list = {K_jvkv_[ijk], K_ivkv_[ijk], K_ivjv_[ijk]};

        for (int perm_idx = 0; perm_idx < P_S.size(); ++perm_idx) {
            auto &[i, j, k] = P_S[perm_idx];
            int ii = i_j_to_ij_[i][i];
            
            Tensor<double, 3> T_ijk_tilde = T_iajbkc_tilde_[ijk];
            if (perm_idx == 1) {
                sort(Indices{index::b, index::a, index::c}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (perm_idx == 2) {
                sort(Indices{index::c, index::a, index::b}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            }

            Tensor<double, 1> R_ia_cont("R_ia_cont", n_tno_[ijk]);
            einsum(0.0, Indices{index::a}, &R_ia_cont, prefactor, Indices{index::a, index::b, index::c}, T_ijk_tilde, Indices{index::b, index::c}, K_ovov_list[perm_idx]);

            auto R_ia_psi = std::make_shared<Matrix>(n_tno_[ijk], 1);
            ::memcpy(R_ia_psi->get_pointer(), &(R_ia_cont)(0, 0), n_tno_[ijk] * sizeof(double));

            auto S_ii_ijk = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ii], lmotriplet_to_paos_[ijk]);
            S_ii_ijk = linalg::triplet(X_pno_[ii], S_ii_ijk, X_tno_[ijk], true, false, false);

            R_ia_buffer[thread][i]->add(linalg::doublet(S_ii_ijk, R_ia_psi, false, false));
        }
        
    } // end ijk

    // Flush buffers
#pragma omp parallel for
    for (int i = 0; i < naocc; ++i) {
        for (int thread = 0; thread < nthread_; ++thread) {
            R_ia[i]->add(R_ia_buffer[thread][i]);
        } // end thread
    } // end i
}

void DLPNOCCSDT::compute_R_iajb_triples(std::vector<SharedMatrix>& R_iajb, std::vector<SharedMatrix>& Rn_iajb,
                                        std::vector<std::vector<SharedMatrix>>& R_iajb_buffer) {

    size_t n_lmo_pairs = ij_to_i_j_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    // Compute residual from LCCSD
    DLPNOCCSD::compute_R_iajb(R_iajb, Rn_iajb);

    // Clean buffers
#pragma omp parallel for
    for (int thread = 0; thread < nthread_; ++thread) {
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            R_iajb_buffer[thread][ij]->zero();
        } // end thread
    } // end i

// Looped over unique triplets (Koch Algorithm 1)
#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        int ntno_ijk = n_tno_[ijk];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // Buffers
        Tensor<double, 2> R_iajb_cont_a("R_iajb_cont_a", n_tno_[ijk], n_tno_[ijk]);
        Tensor<double, 3> R_iajb_cont_b("R_iajb_cont_b", nlmo_ijk, n_tno_[ijk], n_tno_[ijk]);
        SharedMatrix psi_buffer = std::make_shared<Matrix>(n_tno_[ijk], n_tno_[ijk]);

        double prefactor = 1.0;
        if (i == j || j == k) {
            prefactor = 0.5;
        }

        // => Fkc contribution <= //

        std::vector<std::tuple<int, int, int>> P_S = {std::make_tuple(i, j, k), std::make_tuple(i, k, j), std::make_tuple(j, k, i)};
        std::vector<Tensor<double, 3>> K_kvov_list = {K_kvov_[ijk], K_jvov_[ijk], K_ivov_[ijk]};

        for (int idx = 0; idx < P_S.size(); ++idx) {
            auto &[i, j, k] = P_S[idx];
            int ij = i_j_to_ij_[i][j];

            // (T1-dressed Fock Matrix) F_kc = [2.0 * (kc|ld) - (kd|lc)] t_{ld}
            Tensor<double, 1> Fkc("Fkc", ntno_ijk);
            Tensor<double, 3> K_lckd = K_kvov_list[idx];
            sort(Indices{index::l, index::c, index::d}, &K_lckd, Indices{index::l, index::d, index::c}, K_kvov_list[idx]);

            einsum(0.0, Indices{index::c}, &Fkc, 2.0, Indices{index::l, index::d, index::c}, K_kvov_list[idx], Indices{index::l, index::d}, T_n_ijk_[ijk]);
            einsum(1.0, Indices{index::c}, &Fkc, -1.0, Indices{index::l, index::d, index::c}, K_lckd, Indices{index::l, index::d}, T_n_ijk_[ijk]);

            auto T_ijk_tilde = T_iajbkc_tilde_[ijk];
            if (idx == 1) {
                sort(Indices{index::a, index::c, index::b}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (idx == 2) {
                sort(Indices{index::b, index::c, index::a}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            }
            
            einsum(0.0, Indices{index::a, index::b}, &R_iajb_cont_a, prefactor, Indices{index::a, index::b, index::c}, T_ijk_tilde, Indices{index::c}, Fkc);
            ::memcpy(psi_buffer->get_pointer(), R_iajb_cont_a.data(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

            auto S_ijk_ij = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ij]);
            S_ijk_ij = linalg::triplet(X_tno_[ijk], S_ijk_ij, X_pno_[ij], true, false, false);
            R_iajb_buffer[thread][ij]->add(linalg::triplet(S_ijk_ij, psi_buffer, S_ijk_ij, true, false, false));
        }

        // => (db|kc) and (jl|kc) contribution <= //
        std::vector<std::tuple<int, int, int>> perms = {std::make_tuple(i, j, k), std::make_tuple(i, k, j),
                                                        std::make_tuple(j, i, k), std::make_tuple(j, k, i),
                                                        std::make_tuple(k, i, j), std::make_tuple(k, j, i)};

        Tensor<double, 2> K_kvjv = K_jvkv_[ijk];
        sort(Indices{index::b, index::a}, &K_kvjv, Indices{index::a, index::b}, K_jvkv_[ijk]);
        Tensor<double, 2> K_kviv = K_ivkv_[ijk];
        sort(Indices{index::b, index::a}, &K_kviv, Indices{index::a, index::b}, K_ivkv_[ijk]);
        Tensor<double, 2> K_jviv = K_ivjv_[ijk];
        sort(Indices{index::b, index::a}, &K_jviv, Indices{index::a, index::b}, K_ivjv_[ijk]);

        std::vector<Tensor<double, 2>> K_jokv_list = {K_jokv_[ijk], K_kojv_[ijk], K_iokv_[ijk], K_koiv_[ijk], K_iojv_[ijk], K_joiv_[ijk]};
        std::vector<Tensor<double, 2>> K_ovov_list = {K_jvkv_[ijk], K_kvjv, K_ivkv_[ijk], K_kviv, K_ivjv_[ijk], K_jviv};
        std::vector<Tensor<double, 3>> K_kvvv_list = {K_kvvv_[ijk], K_jvvv_[ijk], K_kvvv_[ijk], K_ivvv_[ijk], K_jvvv_[ijk], K_ivvv_[ijk]};
        K_kvov_list = {K_kvov_[ijk], K_jvov_[ijk], K_kvov_[ijk], K_ivov_[ijk], K_jvov_[ijk], K_ivov_[ijk]};

        for (int idx = 0; idx < perms.size(); ++idx) {
            auto &[i, j, k] = perms[idx];
            int ij = i_j_to_ij_[i][j];

            auto T_ijk_tilde = T_iajbkc_tilde_[ijk];
            if (idx == 1) {
                sort(Indices{index::a, index::c, index::b}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (idx == 2) {
                sort(Indices{index::b, index::a, index::c}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (idx == 3) {
                sort(Indices{index::b, index::c, index::a}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (idx == 4) {
                sort(Indices{index::c, index::a, index::b}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            } else if (idx == 5) {
                sort(Indices{index::c, index::b, index::a}, &T_ijk_tilde, Indices{index::a, index::b, index::c}, T_iajbkc_tilde_[ijk]);
            }

            // (T1-dressed integral g_jlkc)
            // (jl|kc)_t1 = (jl|kc) + (jd|kc)t_{l}^{d}
            Tensor<double, 2> g_jlkc = K_jokv_list[idx];
            einsum(1.0, Indices{index::l, index::c}, &g_jlkc, 1.0, Indices{index::d, index::c}, K_ovov_list[idx], Indices{index::l, index::d}, T_n_ijk_[ijk]);

            // (T1-dressed integral g_dbkc)
            // (db|kc)_t1 = (db|kc) - (lb|kc)t_{l}^{d}
            Tensor<double, 3> g_dbkc = K_kvvv_list[idx]; // dbkc
            einsum(1.0, Indices{index::d, index::b, index::c}, &g_dbkc, -1.0, Indices{index::l, index::d}, T_n_ijk_[ijk],
                        Indices{index::l, index::b, index::c}, K_kvov_list[idx]);

            // g_jlkc contribution
            einsum(0.0, Indices{index::l, index::a, index::b}, &R_iajb_cont_b, prefactor, Indices{index::a, index::b, index::c}, T_ijk_tilde, 
                    Indices{index::l, index::c}, g_jlkc);

            // g_dbkc contribution
            einsum(0.0, Indices{index::a, index::d}, &R_iajb_cont_a, prefactor, Indices{index::a, index::b, index::c}, T_ijk_tilde, 
                        Indices{index::d, index::b, index::c}, g_dbkc);

            // Flush buffers (unfortunately we need to copy to Psi for now, this is NOT ideal)
            ::memcpy(psi_buffer->get_pointer(), R_iajb_cont_a.data(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

            auto S_ijk_ij = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ij]);
            S_ijk_ij = linalg::triplet(X_tno_[ijk], S_ijk_ij, X_pno_[ij], true, false, false);
            R_iajb_buffer[thread][ij]->add(linalg::triplet(S_ijk_ij, psi_buffer, S_ijk_ij, true, false, false));
            
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];
                // Flush il buffer
                ::memcpy(psi_buffer->get_pointer(), &(R_iajb_cont_b)(l_ijk, 0, 0), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
                
                auto S_ijk_il = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[il]);
                S_ijk_il = linalg::triplet(X_tno_[ijk], S_ijk_il, X_pno_[il], true, false, false);
                R_iajb_buffer[thread][il]->subtract(linalg::triplet(S_ijk_il, psi_buffer, S_ijk_il, true, false, false));
            } // end l_ijk
        } // end perms

    } // end ijk

    // Flush buffers
#pragma omp parallel for
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int ji = ij_to_ji_[ij];
        for (int thread = 0; thread < nthread_; ++thread) {
            auto cont_a = R_iajb_buffer[thread][ij]->clone();
            cont_a->scale(2.0 / 3.0);
            auto cont_b = R_iajb_buffer[thread][ji]->transpose();
            cont_b->scale(2.0 / 3.0);
            auto cont_c = R_iajb_buffer[thread][ij]->transpose();
            cont_c->scale(1.0 / 3.0);
            auto cont_d = R_iajb_buffer[thread][ji]->clone();
            cont_d->scale(1.0 / 3.0);

            // Add all contributions
            R_iajb[ij]->add(cont_a);
            R_iajb[ij]->add(cont_b);
            R_iajb[ij]->add(cont_c);
            R_iajb[ij]->add(cont_d);
            // R_iajb[ij]->print_out();
            // exit(0);
        } // end thread
    } // end ij
}

void DLPNOCCSDT::compute_R_iajbkc(std::vector<SharedMatrix>& R_iajbkc) {

    size_t naocc = i_j_to_ij_.size();
    size_t n_lmo_pairs = ij_to_i_j_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    const double F_CUT = options_.get_double("F_CUT_T");

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();

        auto R_ijk = R_iajbkc[ijk];
        auto T_ijk = T_iajbkc_[ijk];

        R_ijk->zero();
        //R_ijk->copy(W_iajbkc_[ijk]);

        // Non-orthogonal Fock matrix terms, from DLPNO-(T)
        for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                        (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) += (*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) *
                            ((*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) + (*e_tno_[ijk])(c_ijk) 
                                - (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                }
            }
        }

        // Prepare extended domain for S integrals
        std::vector<int> triplet_ext_domain;
        for (int l = 0; l < naocc; ++l) {
            int ijl_dense = i * naocc * naocc + j * naocc + l;
            int ilk_dense = i * naocc * naocc + l * naocc + k;
            int ljk_dense = l * naocc * naocc + j * naocc + k;
                
            if (l != k && i_j_k_to_ijk_.count(ijl_dense) && std::fabs((*F_lmo_)(l, k)) >= F_CUT) {
                int ijl = i_j_k_to_ijk_[ijl_dense];
                triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ijl]);
            }
                
            if (l != j && i_j_k_to_ijk_.count(ilk_dense) && std::fabs((*F_lmo_)(l, j)) >= F_CUT) {
                int ilk = i_j_k_to_ijk_[ilk_dense];
                triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ilk]);
            }
            
            if (l != i && i_j_k_to_ijk_.count(ljk_dense) && std::fabs((*F_lmo_)(l, i)) >= F_CUT) {
                int ljk = i_j_k_to_ijk_[ljk_dense];
                triplet_ext_domain = merge_lists(triplet_ext_domain, lmotriplet_to_paos_[ljk]);
                
            }
        }
        auto S_ijk = submatrix_rows_and_cols(*S_pao_, triplet_ext_domain, lmotriplet_to_paos_[ijk]);
        S_ijk = linalg::doublet(S_ijk, X_tno_[ijk], false, false);

        // => Add non-orthogonal Fock matrix contributions
        for (int l = 0; l < naocc; l++) {
            int ijl_dense = i * naocc * naocc + j * naocc + l;
            if (l != k && i_j_k_to_ijk_.count(ijl_dense) && std::fabs((*F_lmo_)(l, k)) >= F_CUT) {
                int ijl = i_j_k_to_ijk_[ijl_dense];

                std::vector<int> ijl_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ijl]);
                auto S_ijk_ijl = linalg::doublet(submatrix_rows(*S_ijk, ijl_idx_list), X_tno_[ijl], true, false);

                SharedMatrix T_ijl;
                if (write_amplitudes_) {
                    std::stringstream t_name;
                    t_name << "T " << (ijl);
                    T_ijl = std::make_shared<Matrix>(t_name.str(), n_tno_[ijl], n_tno_[ijl] * n_tno_[ijl]);
#pragma omp critical
                    T_ijl->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                } else {
                    T_ijl = T_iajbkc_[ijl];
                }

                auto T_temp1 =
                    matmul_3d(triples_permuter(T_ijl, i, j, l), S_ijk_ijl, n_tno_[ijl], n_tno_[ijk]);
                C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, k), &(*T_temp1)(0, 0), 1,
                        &(*R_ijk)(0, 0), 1);
            }

            int ilk_dense = i * naocc * naocc + l * naocc + k;
            if (l != j && i_j_k_to_ijk_.count(ilk_dense) && std::fabs((*F_lmo_)(l, j)) >= F_CUT) {
                int ilk = i_j_k_to_ijk_[ilk_dense];

                std::vector<int> ilk_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ilk]);
                auto S_ijk_ilk = linalg::doublet(submatrix_rows(*S_ijk, ilk_idx_list), X_tno_[ilk], true, false);

                SharedMatrix T_ilk;
                if (write_amplitudes_) {
                    std::stringstream t_name;
                    t_name << "T " << (ilk);
                    T_ilk = std::make_shared<Matrix>(t_name.str(), n_tno_[ilk], n_tno_[ilk] * n_tno_[ilk]);
#pragma omp critical
                    T_ilk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                } else {
                    T_ilk = T_iajbkc_[ilk];
                }

                auto T_temp1 =
                    matmul_3d(triples_permuter(T_ilk, i, l, k), S_ijk_ilk, n_tno_[ilk], n_tno_[ijk]);
                C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, j), &(*T_temp1)(0, 0), 1,
                        &(*R_ijk)(0, 0), 1);
            }

            int ljk_dense = l * naocc * naocc + j * naocc + k;
            if (l != i && i_j_k_to_ijk_.count(ljk_dense) && std::fabs((*F_lmo_)(l, i)) >= F_CUT) {
                int ljk = i_j_k_to_ijk_[ljk_dense];

                std::vector<int> ljk_idx_list = index_list(triplet_ext_domain, lmotriplet_to_paos_[ljk]);
                auto S_ijk_ljk = linalg::doublet(submatrix_rows(*S_ijk, ljk_idx_list), X_tno_[ljk], true, false);

                SharedMatrix T_ljk;
                if (write_amplitudes_) {
                    std::stringstream t_name;
                    t_name << "T " << (ljk);
                    T_ljk = std::make_shared<Matrix>(t_name.str(), n_tno_[ljk], n_tno_[ljk] * n_tno_[ljk]);
#pragma omp critical
                    T_ljk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                } else {
                    T_ljk = T_iajbkc_[ljk];
                }

                auto T_temp1 =
                    matmul_3d(triples_permuter(T_ljk, l, j, k), S_ijk_ljk, n_tno_[ljk], n_tno_[ijk]);
                C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, i), &(*T_temp1)(0, 0), 1,
                        &(*R_ijk)(0, 0), 1);
            }
        }

        // T1-dress DF integrals (CC3 terms) 
        // (Yes, all three... this lowkey sucks)
        auto i_ijk = std::find(lmotriplet_to_lmos_[ijk].begin(), lmotriplet_to_lmos_[ijk].end(), i) - lmotriplet_to_lmos_[ijk].begin();
        auto j_ijk = std::find(lmotriplet_to_lmos_[ijk].begin(), lmotriplet_to_lmos_[ijk].end(), j) - lmotriplet_to_lmos_[ijk].begin();
        auto k_ijk = std::find(lmotriplet_to_lmos_[ijk].begin(), lmotriplet_to_lmos_[ijk].end(), k) - lmotriplet_to_lmos_[ijk].begin();

        Tensor<double, 1> T_i = (T_n_ijk_[ijk])(i_ijk, All);
        Tensor<double, 1> T_j = (T_n_ijk_[ijk])(j_ijk, All);
        Tensor<double, 1> T_k = (T_n_ijk_[ijk])(k_ijk, All);

        Tensor<double, 2> q_iv_t1 = q_iv_[ijk];
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1, -1.0, Indices{index::Q, index::l}, q_io_[ijk], Indices{index::l, index::a}, T_n_ijk_[ijk]);
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1, 1.0, Indices{index::Q, index::a, index::b}, q_vv_[ijk], Indices{index::b}, T_i);
        Tensor<double, 2> q_iv_t1_temp("q_iv_t1_temp", naux_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::Q, index::l}, &q_iv_t1_temp, 1.0, Indices{index::Q, index::l, index::b}, q_ov_[ijk], Indices{index::b}, T_i);
        einsum(1.0, Indices{index::Q, index::a}, &q_iv_t1, -1.0, Indices{index::Q, index::l}, q_iv_t1_temp, Indices{index::l, index::a}, T_n_ijk_[ijk]);

        Tensor<double, 2> q_jv_t1 = q_jv_[ijk];
        einsum(1.0, Indices{index::Q, index::a}, &q_jv_t1, -1.0, Indices{index::Q, index::l}, q_jo_[ijk], Indices{index::l, index::a}, T_n_ijk_[ijk]);
        einsum(1.0, Indices{index::Q, index::a}, &q_jv_t1, 1.0, Indices{index::Q, index::a, index::b}, q_vv_[ijk], Indices{index::b}, T_j);
        Tensor<double, 2> q_jv_t1_temp("q_jv_t1_temp", naux_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::Q, index::l}, &q_jv_t1_temp, 1.0, Indices{index::Q, index::l, index::b}, q_ov_[ijk], Indices{index::b}, T_j);
        einsum(1.0, Indices{index::Q, index::a}, &q_jv_t1, -1.0, Indices{index::Q, index::l}, q_jv_t1_temp, Indices{index::l, index::a}, T_n_ijk_[ijk]);

        Tensor<double, 2> q_kv_t1 = q_kv_[ijk];
        einsum(1.0, Indices{index::Q, index::a}, &q_kv_t1, -1.0, Indices{index::Q, index::l}, q_ko_[ijk], Indices{index::l, index::a}, T_n_ijk_[ijk]);
        einsum(1.0, Indices{index::Q, index::a}, &q_kv_t1, 1.0, Indices{index::Q, index::a, index::b}, q_vv_[ijk], Indices{index::b}, T_k);
        Tensor<double, 2> q_kv_t1_temp("q_kv_t1_temp", naux_ijk, nlmo_ijk);
        einsum(0.0, Indices{index::Q, index::l}, &q_kv_t1_temp, 1.0, Indices{index::Q, index::l, index::b}, q_ov_[ijk], Indices{index::b}, T_k);
        einsum(1.0, Indices{index::Q, index::a}, &q_kv_t1, -1.0, Indices{index::Q, index::l}, q_kv_t1_temp, Indices{index::l, index::a}, T_n_ijk_[ijk]);

        // This one is special... the second index is dressed instead of the first (per convention), to increase computational efficiency
        Tensor<double, 3> q_vv_t1 = q_vv_[ijk];
        Tensor<double, 3> q_vo("q_vo", naux_ijk, ntno_ijk, nlmo_ijk);
        sort(Indices{index::Q, index::a, index::l}, &q_vo, Indices{index::Q, index::l, index::a}, q_ov_[ijk]);
        einsum(1.0, Indices{index::Q, index::a, index::b}, &q_vv_t1, -1.0, Indices{index::Q, index::a, index::l}, q_vo, Indices{index::l, index::b}, T_n_ijk_[ijk]);

        Tensor<double, 2> q_io_t1 = q_io_[ijk];
        einsum(1.0, Indices{index::Q, index::l}, &q_io_t1, 1.0, Indices{index::Q, index::l, index::a}, q_ov_[ijk], Indices{index::a}, T_i);
        Tensor<double, 2> q_jo_t1 = q_jo_[ijk];
        einsum(1.0, Indices{index::Q, index::l}, &q_jo_t1, 1.0, Indices{index::Q, index::l, index::a}, q_ov_[ijk], Indices{index::a}, T_j);
        Tensor<double, 2> q_ko_t1 = q_ko_[ijk];
        einsum(1.0, Indices{index::Q, index::l}, &q_ko_t1, 1.0, Indices{index::Q, index::l, index::a}, q_ov_[ijk], Indices{index::a}, T_k);

        // Now contract them
        Tensor<double, 3> K_vvvi("K_vvvi", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);
        einsum(0.0, Indices{index::d, index::b, index::c}, &K_vvvi, 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::c}, q_iv_t1);
        Tensor<double, 3> K_vvvj("K_vvvj", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);
        einsum(0.0, Indices{index::d, index::b, index::c}, &K_vvvj, 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::c}, q_jv_t1);
        Tensor<double, 3> K_vvvk("K_vvvk", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);
        einsum(0.0, Indices{index::d, index::b, index::c}, &K_vvvk, 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::c}, q_kv_t1);

        Tensor<double, 2> K_oivj("K_oivj", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_oivj, 1.0, Indices{index::Q, index::l}, q_io_t1, Indices{index::Q, index::d}, q_jv_t1);
        Tensor<double, 2> K_ojvi("K_ojvi", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_ojvi, 1.0, Indices{index::Q, index::l}, q_jo_t1, Indices{index::Q, index::d}, q_iv_t1);
        Tensor<double, 2> K_ojvk("K_ojvk", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_ojvk, 1.0, Indices{index::Q, index::l}, q_jo_t1, Indices{index::Q, index::d}, q_kv_t1);
        Tensor<double, 2> K_okvj("K_okvj", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_okvj, 1.0, Indices{index::Q, index::l}, q_ko_t1, Indices{index::Q, index::d}, q_jv_t1);
        Tensor<double, 2> K_oivk("K_oivk", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_oivk, 1.0, Indices{index::Q, index::l}, q_io_t1, Indices{index::Q, index::d}, q_kv_t1);
        Tensor<double, 2> K_okvi("K_okvi", nlmo_ijk, n_tno_[ijk]);
        einsum(0.0, Indices{index::l, index::d}, &K_okvi, 1.0, Indices{index::Q, index::l}, q_ko_t1, Indices{index::Q, index::d}, q_iv_t1);

        std::vector<std::tuple<int, int, int>> perms = {std::make_tuple(i, j, k), std::make_tuple(i, k, j),
                                                        std::make_tuple(j, i, k), std::make_tuple(j, k, i),
                                                        std::make_tuple(k, i, j), std::make_tuple(k, j, i)};
        std::vector<Tensor<double, 3>> Wperms(perms.size());

        std::vector<Tensor<double, 3>> K_vvvo_list = {K_vvvk, K_vvvj, K_vvvk, K_vvvi, K_vvvj, K_vvvi};
        std::vector<Tensor<double, 2>> K_oovo_list = {K_ojvk, K_okvj, K_oivk, K_okvi, K_oivj, K_ojvi};

        for (int idx = 0; idx < perms.size(); ++idx) {
            auto &[i, j, k] = perms[idx];
            int ij = i_j_to_ij_[i][j];

            Wperms[idx] = Tensor("Wperm", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);
            
            // Compute overlap between TNOs of triplet ijk and PNOs of pair ij
            auto S_ij_ijk = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmotriplet_to_paos_[ijk]);
            S_ij_ijk = linalg::triplet(X_pno_[ij], S_ij_ijk, X_tno_[ijk], true, false, false);
            auto T_ij = linalg::triplet(S_ij_ijk, T_iajb_[ij], S_ij_ijk, true, false, false);
            Tensor<double, 2> T_ij_einsums("T_ij", n_tno_[ijk], n_tno_[ijk]);
            ::memcpy(T_ij_einsums.data(), T_ij->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

            einsum(0.0, Indices{index::a, index::b, index::c}, &Wperms[idx], 1.0, Indices{index::a, index::d}, T_ij_einsums, 
                    Indices{index::d, index::b, index::c}, K_vvvo_list[idx]);

            Tensor<double, 3> T_il_einsums("T_il", nlmo_ijk, n_tno_[ijk], n_tno_[ijk]);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];

                auto S_il_ijk = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[il], lmotriplet_to_paos_[ijk]);
                S_il_ijk = linalg::triplet(X_pno_[il], S_il_ijk, X_tno_[ijk], true, false, false);
                auto T_il = linalg::triplet(S_il_ijk, T_iajb_[il], S_il_ijk, true, false, false);
                ::memcpy(&T_il_einsums(l_ijk, 0, 0), T_il->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            } // end l_ijk
            
            einsum(1.0, Indices{index::a, index::b, index::c}, &Wperms[idx], -1.0, Indices{index::l, index::a, index::b}, T_il_einsums, 
                    Indices{index::l, index::c}, K_oovo_list[idx]);
        }

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) +=
                        (Wperms[0])(a_ijk, b_ijk, c_ijk) + (Wperms[1])(a_ijk, c_ijk, b_ijk) + (Wperms[2])(b_ijk, a_ijk, c_ijk) + 
                        (Wperms[3])(b_ijk, c_ijk, a_ijk) + (Wperms[4])(c_ijk, a_ijk, b_ijk) + (Wperms[5])(c_ijk, b_ijk, a_ijk);
                }
            }
        }

    } // end ijk
}

void DLPNOCCSDT::lccsdt_iterations() {

    int naocc = i_j_to_ij_.size();
    int n_lmo_pairs = ij_to_i_j_.size();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    // Thread and OMP Parallel Info
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    // => Initialize Residuals and Amplitude <= //

    std::vector<SharedMatrix> R_ia(naocc);
    std::vector<SharedMatrix> Rn_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> R_iajb(n_lmo_pairs);
    std::vector<SharedMatrix> R_iajbkc(n_lmo_triplets);

    for (int i = 0; i < naocc; ++i) {
        int ii = i_j_to_ij_[i][i];
        R_ia[i] = std::make_shared<Matrix>(n_pno_[ii], 1);
    }

    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        R_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        Rn_iajb[ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
    }

    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        R_iajbkc[ijk] = std::make_shared<Matrix>(n_tno_[ijk], n_tno_[ijk] * n_tno_[ijk]);
    }

    std::vector<std::vector<SharedMatrix>> R_ia_buffer(nthreads);
    std::vector<std::vector<SharedMatrix>> R_iajb_buffer(nthreads);

    for (int thread = 0; thread < nthreads; ++thread) {
        R_ia_buffer[thread].resize(naocc);
        R_iajb_buffer[thread].resize(n_lmo_pairs);
        
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            R_ia_buffer[thread][i] = std::make_shared<Matrix>(n_pno_[ii], 1);
        }

        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            R_iajb_buffer[thread][ij] = std::make_shared<Matrix>(n_pno_[ij], n_pno_[ij]);
        }
    }

    // Necessary intermediates
    T_n_ijk_.resize(n_lmo_triplets);
    T_iajbkc_tilde_.resize(n_lmo_triplets);

    // LCCSDT iterations

    outfile->Printf("\n  ==> Local CCSDT <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                       Corr. Energy    Delta E     Max R1     Max R2     Max R3     Time (s)\n");

    int iteration = 1, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r_curr1 = 0.0, r_curr2 = 0.0, r_curr3 = 0.0;
    bool e_converged = false, r_converged = false;

    while (!(e_converged && r_converged)) {
        // RMS of residual per LMO orbital, for assessing convergence
        std::vector<double> R_ia_rms(naocc, 0.0);
        // RMS of residual per LMO pair, for assessing convergence
        std::vector<double> R_iajb_rms(n_lmo_pairs, 0.0);
        // RMS of residual per LMO triplet, for assessing convergence
        std::vector<double> R_iajbkc_rms(n_lmo_triplets, 0.0);

        std::time_t time_start = std::time(nullptr);



        // Create T_n_ij and T_n_ijk intermediates
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];

            int nlmo_ij = lmopair_to_lmos_[ij].size();
            int npno_ij = n_pno_[ij];
            int ij_idx = (i <= j) ? ij : ij_to_ji_[ij];
            
            T_n_ij_[ij] = std::make_shared<Matrix>(nlmo_ij, npno_ij);

            for (int n_ij = 0; n_ij < nlmo_ij; ++n_ij) {
                int n = lmopair_to_lmos_[ij][n_ij];
                int nn = i_j_to_ij_[n][n];
                auto T_n_temp = linalg::doublet(S_pno_ij_nn_[ij_idx][n], T_ia_[n], false, false);
                
                for (int a_ij = 0; a_ij < npno_ij; ++a_ij) {
                    (*T_n_ij_[ij])(n_ij, a_ij) = (*T_n_temp)(a_ij, 0);
                } // end a_ij
            } // end n_ij
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            auto &[i, j, k] = ijk_to_i_j_k_[ijk];
            int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();

            T_n_ijk_[ijk] = Tensor<double, 2>("T_n_ijk", nlmo_ijk, n_tno_[ijk]);
            
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int ll = i_j_to_ij_[l][l];

                auto S_ijk_ll = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ll]);
                S_ijk_ll = linalg::triplet(X_tno_[ijk], S_ijk_ll, X_pno_[ll], true, false, false);
                auto T_l_temp = linalg::doublet(S_ijk_ll, T_ia_[l]);

                for (int a_ijk = 0; a_ijk < n_tno_[ijk]; ++a_ijk) {
                    (T_n_ijk_[ijk])(l_ijk, a_ijk) = (*T_l_temp)(a_ijk, 0);
                }
            } // end l_ijk
        } // end ijk

        // Create T_iajbkc_tilde intermediate
#pragma omp parallel for schedule(dynamic, 1)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            auto &[i, j, k] = ijk_to_i_j_k_[ijk];

            T_iajbkc_tilde_[ijk] = Tensor<double, 3>("T_iajbkc_tilde", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);
            
            for (int a_ijk = 0; a_ijk < n_tno_[ijk]; ++a_ijk) {
                for (int b_ijk = 0; b_ijk < n_tno_[ijk]; ++b_ijk) {
                    for (int c_ijk = 0; c_ijk < n_tno_[ijk]; ++c_ijk) {
                        T_iajbkc_tilde_[ijk](a_ijk, b_ijk, c_ijk) = 4 * (*T_iajbkc_[ijk])(a_ijk, b_ijk * n_tno_[ijk] + c_ijk)
                            - 2 * (*T_iajbkc_[ijk])(a_ijk, c_ijk * n_tno_[ijk] + b_ijk) - 2 * (*T_iajbkc_[ijk])(c_ijk, b_ijk * n_tno_[ijk] + a_ijk)
                            - 2 * (*T_iajbkc_[ijk])(b_ijk, a_ijk * n_tno_[ijk] + c_ijk) + (*T_iajbkc_[ijk])(b_ijk, c_ijk * n_tno_[ijk] + a_ijk)
                            + (*T_iajbkc_[ijk])(c_ijk, a_ijk * n_tno_[ijk] + b_ijk);
                    }
                }
            }
        }

        // T1-dress integrals and Fock matrices
        t1_ints();
        t1_fock();

        // compute singles amplitude
        compute_R_ia_triples(R_ia, R_ia_buffer);
        // compute doubles amplitude
        compute_R_iajb_triples(R_iajb, Rn_iajb, R_iajb_buffer);
        // compute triples amplitude
        compute_R_iajbkc(R_iajbkc);

        // Update singles amplitude
#pragma omp parallel for
        for (int i = 0; i < naocc; ++i) {
            int ii = i_j_to_ij_[i][i];
            for (int a_ii = 0; a_ii < n_pno_[ii]; ++a_ii) {
                (*T_ia_[i])(a_ii, 0) -= (*R_ia[i])(a_ii, 0) / (e_pno_[ii]->get(a_ii) - F_lmo_->get(i,i));
            }
            R_ia_rms[i] = R_ia[i]->rms();
        }

        // Update doubles amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];
            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    (*T_iajb_[ij])(a_ij, b_ij) -= (*R_iajb[ij])(a_ij, b_ij) / 
                                    (e_pno_[ij]->get(a_ij) + e_pno_[ij]->get(b_ij) - F_lmo_->get(i,i) - F_lmo_->get(j,j));
                }
            }
            Tt_iajb_[ij] = T_iajb_[ij]->clone();
            Tt_iajb_[ij]->scale(2.0);
            Tt_iajb_[ij]->subtract(T_iajb_[ij]->transpose());

            R_iajb_rms[ij] = R_iajb[ij]->rms();
        }

        // Update triples amplitude
#pragma omp parallel for schedule(dynamic, 1)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            auto &[i, j, k] = ijk_to_i_j_k_[ijk];
            for (int a_ijk = 0; a_ijk < n_tno_[ijk]; ++a_ijk) {
                for (int b_ijk = 0; b_ijk < n_tno_[ijk]; ++b_ijk) {
                    for (int c_ijk = 0; c_ijk < n_tno_[ijk]; ++c_ijk) {
                        (*T_iajbkc_[ijk])(a_ijk, b_ijk * n_tno_[ijk] + c_ijk) -= (*R_iajbkc[ijk])(a_ijk, b_ijk * n_tno_[ijk] + c_ijk) /
                                            (e_tno_[ijk]->get(a_ijk) + e_tno_[ijk]->get(b_ijk) + e_tno_[ijk]->get(c_ijk) - F_lmo_->get(i,i) - F_lmo_->get(j,j) - F_lmo_->get(k,k));
                    }
                }
            }
            R_iajbkc_rms[ijk] = R_iajbkc[ijk]->rms();
        }

        // evaluate energy and convergence
        e_prev = e_curr;
        e_curr = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];
            int ij_idx = (i < j) ? ij : ij_to_ji_[ij];

            auto tau = T_iajb_[ij]->clone();
            auto S_ij_ii = S_pno_ij_nn_[ij_idx][i];
            auto S_ij_jj = S_pno_ij_nn_[ij_idx][j];
            auto tia_temp = linalg::doublet(S_ij_ii, T_ia_[i]);
            auto tjb_temp = linalg::doublet(S_ij_jj, T_ia_[j]);

            for (int a_ij = 0; a_ij < n_pno_[ij]; ++a_ij) {
                for (int b_ij = 0; b_ij < n_pno_[ij]; ++b_ij) {
                    (*tau)(a_ij, b_ij) += (*tia_temp)(a_ij, 0) * (*tjb_temp)(b_ij, 0);
                } // end b_ij
            } // end a_ij

            e_curr += tau->vector_dot(L_iajb_[ij]);
        }

        double r_curr1 = *max_element(R_ia_rms.begin(), R_ia_rms.end());
        double r_curr2 = *max_element(R_iajb_rms.begin(), R_iajb_rms.end());
        double r_curr3 = *max_element(R_iajbkc_rms.begin(), R_iajbkc_rms.end());

        r_converged = fabs(r_curr1) < options_.get_double("R_CONVERGENCE");
        r_converged &= fabs(r_curr2) < options_.get_double("R_CONVERGENCE");
        r_converged &= fabs(r_curr3) < options_.get_double("R_CONVERGENCE");
        e_converged = fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE");

        e_lccsdt_ = e_curr;

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @LCCSDT iter %3d: %16.12f %10.3e %10.3e %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, r_curr1, r_curr2, r_curr3, (int)time_stop - (int)time_start);

        ++iteration;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }
}

double DLPNOCCSDT::compute_energy() {
    double E_DLPNO_CCSD_T = DLPNOCCSD_T::compute_energy();

    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    print_header();
    estimate_memory();
    compute_integrals();
    lccsdt_iterations();
    // print_results();

    double E_DLPNOCCSDT = e_lccsdt_;

    return E_DLPNOCCSDT;
}

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "DLPNO_CCSDT_Q"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- Debug level -*/
        options.add_int("DEBUG", 0);
        /*- Bench level -*/
        options.add_int("BENCH", 0);
        /*- Basis set -*/
        options.add_str("DF_BASIS_MP2", "");
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction dlpno_ccsdt_q(SharedWavefunction ref_wfn, Options& options) {

    einsums::initialize();

    auto dlpno_ccsdt_wfn = DLPNOCCSDT(ref_wfn, options);
    dlpno_ccsdt_wfn.compute_energy();

    einsums::finalize(true);

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}}} // End namespaces

