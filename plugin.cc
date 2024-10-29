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
#include "psi4/libdiis/diismanager.h"

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
    // TNO overlap integrals
    std::vector<SharedMatrix> S_ijk_ii_;
    std::vector<SharedMatrix> S_ijk_jj_;
    std::vector<SharedMatrix> S_ijk_kk_;

    std::vector<std::vector<SharedMatrix>> S_ijk_ll_;

    std::vector<SharedMatrix> S_ijk_ij_;
    std::vector<SharedMatrix> S_ijk_jk_;
    std::vector<SharedMatrix> S_ijk_ik_;

    std::vector<std::vector<SharedMatrix>> S_ijk_il_;
    std::vector<std::vector<SharedMatrix>> S_ijk_jl_;
    std::vector<std::vector<SharedMatrix>> S_ijk_kl_;

    std::vector<std::vector<SharedMatrix>> S_ijk_lm_;

    std::vector<std::vector<SharedMatrix>> S_ijk_ljk_;
    std::vector<std::vector<SharedMatrix>> S_ijk_ilk_;
    std::vector<std::vector<SharedMatrix>> S_ijk_ijl_;

    std::vector<std::vector<SharedMatrix>> S_ijk_mli_;
    std::vector<std::vector<SharedMatrix>> S_ijk_mlj_;
    std::vector<std::vector<SharedMatrix>> S_ijk_mlk_;

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

    // Run CC3 only?
    bool cc3_;
    // Number of threads
    int nthread_;
    // Energy expression
    double e_lccsdt_;
    
    /// computes singles residuals in LCCSDT equations
    void compute_R_ia_triples(std::vector<SharedMatrix>& R_ia, std::vector<std::vector<SharedMatrix>>& R_ia_buffer);
    /// compute doubles residuals in LCCSDT equations
    void compute_R_iajb_triples(std::vector<SharedMatrix>& R_iajb, std::vector<SharedMatrix>& Rn_iajb, std::vector<std::vector<SharedMatrix>>& R_iajb_buffer);
    /// computes triples residuals in LCC3 equations
    void compute_R_iajbkc_cc3(std::vector<SharedMatrix>& R_iajbkc);
    /// computes triples residuals in LCCSDT equations
    void compute_R_iajbkc(std::vector<SharedMatrix>& R_iajbkc);

    void print_header();
    void estimate_memory();
    void compute_integrals();
    void compute_tno_overlaps();
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

    size_t naocc = i_j_to_ij_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    size_t K_iojv_memory = 0;
    size_t K_ivov_memory = 0;
    size_t K_ivvv_memory = 0;
    size_t qij_memory = 0;
    size_t qia_memory = 0;
    size_t qab_memory = 0;
    size_t S_tno_osv_small = 0;
    size_t S_tno_osv_large = 0;
    size_t S_tno_pno_small = 0;
    size_t S_tno_pno_med = 0;
    size_t S_tno_pno_large = 0;
    size_t S_tno_tno_small = 0;
    size_t S_tno_tno_large = 0;

#pragma omp parallel for reduction(+ : K_iojv_memory, K_ivov_memory, K_ivvv_memory, qij_memory, qia_memory, qab_memory, S_tno_osv_small, S_tno_osv_large, S_tno_pno_small, S_tno_pno_med, S_tno_pno_large, S_tno_tno_small, S_tno_tno_large)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], kk = i_j_to_ij_[k][k];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        int npao_ijk = lmotriplet_to_paos_[ijk].size();
        int ntno_ijk = n_tno_[ijk];

        K_iojv_memory += 6 * nlmo_ijk * ntno_ijk;
        K_ivov_memory += 3 * nlmo_ijk * ntno_ijk * ntno_ijk;
        K_ivvv_memory += 3 * ntno_ijk * ntno_ijk * ntno_ijk;
        qij_memory += 3 * naux_ijk * nlmo_ijk;
        qia_memory += 3 * naux_ijk * nlmo_ijk * ntno_ijk;
        qab_memory += naux_ijk * ntno_ijk * ntno_ijk;

        S_tno_osv_small += ntno_ijk * (n_pno_[ii] + n_pno_[jj] + n_pno_[kk]);
        S_tno_pno_small += ntno_ijk * (n_pno_[ij] + n_pno_[jk] + n_pno_[ik]);

        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];
            int ll = i_j_to_ij_[l][l];
            int il = i_j_to_ij_[i][l], jl = i_j_to_ij_[j][l], kl = i_j_to_ij_[k][l];

            S_tno_osv_large += ntno_ijk * n_pno_[ll];
            S_tno_pno_med += ntno_ijk * (n_pno_[il] + n_pno_[jl] + n_pno_[kl]);

            int ljk_dense = l * naocc * naocc + j * naocc + k;
            if (i_j_k_to_ijk_.count(ljk_dense)) {
                int ljk = i_j_k_to_ijk_[ljk_dense];
                S_tno_tno_small += ntno_ijk * n_tno_[ljk];
            } // end if

            int ilk_dense = i * naocc * naocc + l * naocc + k;
            if (i_j_k_to_ijk_.count(ilk_dense)) {
                int ilk = i_j_k_to_ijk_[ilk_dense];
                S_tno_tno_small += ntno_ijk * n_tno_[ilk];
            } // end if

            int ijl_dense = i * naocc * naocc + j * naocc + l;
            if (i_j_k_to_ijk_.count(ijl_dense)) {
                int ijl = i_j_k_to_ijk_[ijl_dense];
                S_tno_tno_small += ntno_ijk * n_tno_[ijl];
            } // end if

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                if (l_ijk > m_ijk) continue;
                int lm_idx = l_ijk * nlmo_ijk + m_ijk;
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int lm = i_j_to_ij_[l][m];
                if (lm != -1) {
                    S_tno_pno_large += ntno_ijk * n_pno_[lm];
                }

                int mli_dense = m * naocc * naocc + l * naocc + i;
                if (i_j_k_to_ijk_.count(mli_dense)) {
                    int mli = i_j_k_to_ijk_[mli_dense];
                    S_tno_tno_large += ntno_ijk * n_tno_[mli];
                } // end if

                int mlj_dense = m * naocc * naocc + l * naocc + j;
                if (i_j_k_to_ijk_.count(mlj_dense)) {
                    int mlj = i_j_k_to_ijk_[mlj_dense];
                    S_tno_tno_large += ntno_ijk * n_tno_[mlj];
                } // end if

                int mlk_dense = m * naocc * naocc + l * naocc + k;
                if (i_j_k_to_ijk_.count(mlk_dense)) {
                    int mlk = i_j_k_to_ijk_[mlk_dense];
                    S_tno_tno_large += ntno_ijk * n_tno_[mlk];
                } // end if
            } // end m_ijk
        } // end l_ijk
    }

    size_t total_memory = K_iojv_memory + K_ivov_memory + K_ivvv_memory + qij_memory + qia_memory + qab_memory 
                            + S_tno_osv_small + S_tno_osv_large + S_tno_pno_small + S_tno_pno_med + S_tno_pno_large + S_tno_tno_small + S_tno_tno_large;

    outfile->Printf("    (i j | l_{ijk} a_{ijk})        : %.3f [GiB]\n", K_iojv_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (i a_{ijk} | l_{ijk} b_{ijk})  : %.3f [GiB]\n", K_ivov_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (i a_{ijk} | b_{ijk} c_{ijk})  : %.3f [GiB]\n", K_ivvv_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | l_{ijk} [i,j,k])    : %.3f [GiB]\n", qij_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | l_{ijk} a_{ijk})    : %.3f [GiB]\n", qia_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (Q_{ijk} | a_{ijk} b_{ijk})    : %.3f [GiB]\n", qab_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{ii})             : %.3f [GiB]\n", S_tno_osv_small * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{ll})             : %.3f [GiB]\n", S_tno_osv_large * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{ij})             : %.3f [GiB]\n", S_tno_pno_small * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{il})             : %.3f [GiB]\n", S_tno_pno_med * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{lm})             : %.3f [GiB]\n", S_tno_pno_large * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{ljk})            : %.3f [GiB]\n", S_tno_tno_small * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    S(a_{ijk}, a_{mli})            : %.3f [GiB]\n", S_tno_tno_large * pow(2.0, -30) * sizeof(double));
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

void DLPNOCCSDT::compute_tno_overlaps() {

    size_t naocc = i_j_to_ij_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

    S_ijk_ii_.resize(n_lmo_triplets);
    S_ijk_jj_.resize(n_lmo_triplets);
    S_ijk_kk_.resize(n_lmo_triplets);

    S_ijk_ll_.resize(n_lmo_triplets);

    S_ijk_ij_.resize(n_lmo_triplets);
    S_ijk_jk_.resize(n_lmo_triplets);
    S_ijk_ik_.resize(n_lmo_triplets);

    S_ijk_il_.resize(n_lmo_triplets);
    S_ijk_jl_.resize(n_lmo_triplets);
    S_ijk_kl_.resize(n_lmo_triplets);

    S_ijk_lm_.resize(n_lmo_triplets);

    S_ijk_ljk_.resize(n_lmo_triplets);
    S_ijk_ilk_.resize(n_lmo_triplets);
    S_ijk_ijl_.resize(n_lmo_triplets);

    S_ijk_mli_.resize(n_lmo_triplets);
    S_ijk_mlj_.resize(n_lmo_triplets);
    S_ijk_mlk_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], kk = i_j_to_ij_[k][k];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();

        // TNO/OSV overlaps
        S_ijk_ii_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ii]);
        S_ijk_ii_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ii_[ijk], X_pno_[ii], true, false, false);

        S_ijk_jj_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[jj]);
        S_ijk_jj_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_jj_[ijk], X_pno_[jj], true, false, false);

        S_ijk_kk_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[kk]);
        S_ijk_kk_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_kk_[ijk], X_pno_[kk], true, false, false);

        S_ijk_ll_[ijk].resize(nlmo_ijk);

        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];
            int ll = i_j_to_ij_[l][l];

            S_ijk_ll_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ll]);
            S_ijk_ll_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ll_[ijk][l_ijk], X_pno_[ll], true, false, false);
        }

        // TNO/PNO overlaps
        S_ijk_ij_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ij]);
        S_ijk_ij_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ij_[ijk], X_pno_[ij], true, false, false);

        S_ijk_jk_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[jk]);
        S_ijk_jk_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_jk_[ijk], X_pno_[jk], true, false, false);

        S_ijk_ik_[ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ik]);
        S_ijk_ik_[ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ik_[ijk], X_pno_[ik], true, false, false);

        S_ijk_il_[ijk].resize(nlmo_ijk);
        S_ijk_jl_[ijk].resize(nlmo_ijk);
        S_ijk_kl_[ijk].resize(nlmo_ijk);

        S_ijk_lm_[ijk].resize(nlmo_ijk * nlmo_ijk);

        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];
            int il = i_j_to_ij_[i][l], jl = i_j_to_ij_[j][l], kl = i_j_to_ij_[k][l];

            S_ijk_il_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[il]);
            S_ijk_il_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_il_[ijk][l_ijk], X_pno_[il], true, false, false);

            S_ijk_jl_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[jl]);
            S_ijk_jl_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_jl_[ijk][l_ijk], X_pno_[jl], true, false, false);

            S_ijk_kl_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[kl]);
            S_ijk_kl_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_kl_[ijk][l_ijk], X_pno_[kl], true, false, false);

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                if (l_ijk > m_ijk) continue;
                int lm_idx = l_ijk * nlmo_ijk + m_ijk;
                int lm = i_j_to_ij_[l][m];
                if (lm == -1) continue;

                S_ijk_lm_[ijk][lm_idx] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[lm]);
                S_ijk_lm_[ijk][lm_idx] = linalg::triplet(X_tno_[ijk], S_ijk_lm_[ijk][lm_idx], X_pno_[lm], true, false, false);
            }
        }

        // TNO/TNO overlaps
        S_ijk_ljk_[ijk].resize(nlmo_ijk);
        S_ijk_ilk_[ijk].resize(nlmo_ijk);
        S_ijk_ijl_[ijk].resize(nlmo_ijk);

        S_ijk_mli_[ijk].resize(nlmo_ijk * nlmo_ijk);
        S_ijk_mlj_[ijk].resize(nlmo_ijk * nlmo_ijk);
        S_ijk_mlk_[ijk].resize(nlmo_ijk * nlmo_ijk);

        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];
            int ljk_dense = l * naocc * naocc + j * naocc + k;
            if (i_j_k_to_ijk_.count(ljk_dense)) {
                int ljk = i_j_k_to_ijk_[ljk_dense];
                S_ijk_ljk_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ljk]);
                S_ijk_ljk_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ljk_[ijk][l_ijk], X_tno_[ljk], true, false, false);
            } // end if

            int ilk_dense = i * naocc * naocc + l * naocc + k;
            if (i_j_k_to_ijk_.count(ilk_dense)) {
                int ilk = i_j_k_to_ijk_[ilk_dense];
                S_ijk_ilk_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ilk]);
                S_ijk_ilk_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ilk_[ijk][l_ijk], X_tno_[ilk], true, false, false);
            } // end if

            int ijl_dense = i * naocc * naocc + j * naocc + l;
            if (i_j_k_to_ijk_.count(ijl_dense)) {
                int ijl = i_j_k_to_ijk_[ijl_dense];
                S_ijk_ijl_[ijk][l_ijk] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijl]);
                S_ijk_ijl_[ijk][l_ijk] = linalg::triplet(X_tno_[ijk], S_ijk_ijl_[ijk][l_ijk], X_tno_[ijl], true, false, false);
            } // end if

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                if (l_ijk > m_ijk) continue;
                int lm_idx = l_ijk * nlmo_ijk + m_ijk;
                int m = lmotriplet_to_lmos_[ijk][m_ijk];

                int mli_dense = m * naocc * naocc + l * naocc + i;
                if (i_j_k_to_ijk_.count(mli_dense)) {
                    int mli = i_j_k_to_ijk_[mli_dense];
                    S_ijk_mli_[ijk][lm_idx] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[mli]);
                    S_ijk_mli_[ijk][lm_idx] = linalg::triplet(X_tno_[ijk], S_ijk_mli_[ijk][lm_idx], X_tno_[mli], true, false, false);
                } // end if

                int mlj_dense = m * naocc * naocc + l * naocc + j;
                if (i_j_k_to_ijk_.count(mlj_dense)) {
                    int mlj = i_j_k_to_ijk_[mlj_dense];
                    S_ijk_mlj_[ijk][lm_idx] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[mlj]);
                    S_ijk_mlj_[ijk][lm_idx] = linalg::triplet(X_tno_[ijk], S_ijk_mlj_[ijk][lm_idx], X_tno_[mlj], true, false, false);
                } // end if

                int mlk_dense = m * naocc * naocc + l * naocc + k;
                if (i_j_k_to_ijk_.count(mlk_dense)) {
                    int mlk = i_j_k_to_ijk_[mlk_dense];
                    S_ijk_mlk_[ijk][lm_idx] = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[mlk]);
                    S_ijk_mlk_[ijk][lm_idx] = linalg::triplet(X_tno_[ijk], S_ijk_mlk_[ijk][lm_idx], X_tno_[mlk], true, false, false);
                } // end if
            } // end m_ijk

        } // end l_ijk
    }
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
        std::vector<SharedMatrix> S_ijk_ii_list = {S_ijk_ii_[ijk], S_ijk_jj_[ijk], S_ijk_kk_[ijk]};

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

            R_ia_buffer[thread][i]->add(linalg::doublet(S_ijk_ii_list[perm_idx], R_ia_psi, true, false));
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
    
    if (weak_pair_algorithm_ != WeakPairAlgorithm::MP2) {
        DLPNOCCSD::compute_R_iajb_weak(R_iajb);
    }

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
        std::vector<SharedMatrix> S_ijk_ij_list = {S_ijk_ij_[ijk], S_ijk_ik_[ijk], S_ijk_jk_[ijk]};

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

            R_iajb_buffer[thread][ij]->add(linalg::triplet(S_ijk_ij_list[idx], psi_buffer, S_ijk_ij_list[idx], true, false, false));
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

        std::vector<SharedMatrix> S_ijk_ij_list_long = {S_ijk_ij_[ijk], S_ijk_ik_[ijk], S_ijk_ij_[ijk],
                                                            S_ijk_jk_[ijk], S_ijk_ik_[ijk], S_ijk_jk_[ijk]};
        std::vector<std::vector<SharedMatrix>> S_ijk_il_list = {S_ijk_il_[ijk], S_ijk_il_[ijk], S_ijk_jl_[ijk],
                                                                    S_ijk_jl_[ijk], S_ijk_kl_[ijk], S_ijk_kl_[ijk]};

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

            R_iajb_buffer[thread][ij]->add(linalg::triplet(S_ijk_ij_list_long[idx], psi_buffer, S_ijk_ij_list_long[idx], true, false, false));
            
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];
                // Flush il buffer
                ::memcpy(psi_buffer->get_pointer(), &(R_iajb_cont_b)(l_ijk, 0, 0), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
                
                R_iajb_buffer[thread][il]->subtract(linalg::triplet(S_ijk_il_list[idx][l_ijk], psi_buffer, S_ijk_il_list[idx][l_ijk], true, false, false));
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

void DLPNOCCSDT::compute_R_iajbkc_cc3(std::vector<SharedMatrix>& R_iajbkc) {

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

void DLPNOCCSDT::compute_R_iajbkc(std::vector<SharedMatrix>& R_iajbkc) {

    size_t naocc = i_j_to_ij_.size();
    size_t n_lmo_pairs = ij_to_i_j_.size();
    size_t n_lmo_triplets = ijk_to_i_j_k_.size();

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        auto &[i, j, k] = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();

        auto R_ijk = R_iajbkc[ijk];
        auto T_ijk = T_iajbkc_[ijk];

        R_ijk->zero();

        // => STEP 0: T1-dress DF integrals <= //
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

        std::vector<int> ijk_idx = {i, j, k};
        std::vector<Tensor<double, 1>> T_i_list = {T_i, T_j, T_k};
        std::vector<Tensor<double, 2>> q_io_t1_list = {q_io_t1, q_jo_t1, q_ko_t1};
        std::vector<Tensor<double, 2>> q_iv_t1_list = {q_iv_t1, q_jv_t1, q_kv_t1};

        // outfile->Printf("With a thousand hallelujahs!\n");

        // => Form Fock Matrix Intermediates <= //

        // Gamma_Q is used universally for J-like contractions
        Tensor<double, 1> gamma_Q("gamma_Q", naux_ijk);
        einsum(0.0, Indices{index::Q}, &gamma_Q, 1.0, Indices{index::Q, index::m, index::e}, q_ov_[ijk], Indices{index::m, index::e}, T_n_ijk_[ijk]);

        // => F_ld (this is scoped to ensure that the intermediate tensors are not persistent in memory <= //
        Tensor<double, 2> F_ld("F_ld", nlmo_ijk, ntno_ijk); {
            // J contractions
            einsum(0.0, Indices{index::l, index::d}, &F_ld, 2.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q}, gamma_Q);
            
            // K contractions
            Tensor<double, 3> F_ld_K_temp("F_ld_K_temp", naux_ijk, nlmo_ijk, nlmo_ijk);
            einsum(0.0, Indices{index::Q, index::l, index::m}, &F_ld_K_temp, 1.0, Indices{index::Q, index::l, index::e}, q_ov_[ijk], Indices{index::m, index::e}, T_n_ijk_[ijk]);
            Tensor<double, 3> F_ld_K_temp2("F_ld_K_temp2", naux_ijk, nlmo_ijk, nlmo_ijk);
            sort(Indices{index::Q, index::m, index::l}, &F_ld_K_temp2, Indices{index::Q, index::l, index::m}, F_ld_K_temp);
            einsum(1.0, Indices{index::l, index::d}, &F_ld, -1.0, Indices{index::Q, index::m, index::l}, F_ld_K_temp2, Indices{index::Q, index::m, index::d}, q_ov_[ijk]);
        }

        // => F_li <= //
        std::vector<Tensor<double, 1>> F_li_list(ijk_idx.size()); // F_li, F_lj, F_lk

        for (int idx = 0; idx < ijk_idx.size(); ++idx) {
            int i = ijk_idx[idx];

            // F_lmo (non-T1 contribution)
            F_li_list[idx] = Tensor<double, 1>("F_li", nlmo_ijk);
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                F_li_list[idx](l_ijk) = (*F_lmo_)(l, i);
            }

            // J Contractions
            Tensor<double, 2> q_io_slice = q_ov_[ijk](All, All, i_ijk);
            einsum(1.0, Indices{index::l}, &F_li_list[idx], 2.0, Indices{index::Q, index::l}, q_io_slice, Indices{index::Q}, gamma_Q);

            // K contractions
            Tensor<double, 2> F_li_K_temp("F_li_K_temp", naux_ijk, ntno_ijk);
            einsum(0.0, Indices{index::Q, index::a}, &F_li_K_temp, 1.0, Indices{index::Q, index::m}, q_io_slice, Indices{index::m, index::a}, T_n_ijk_[ijk]);
            einsum(1.0, Indices{index::l}, &F_li_list[idx], -1.0, Indices{index::Q, index::a, index::l}, q_vo, Indices{index::Q, index::a}, F_li_K_temp);

            // Add F_ld contribution
            einsum(1.0, Indices{index::l}, &F_li_list[idx], 1.0, Indices{index::l, index::d}, F_ld, Indices{index::d}, T_i);
        }

        // => F_ad <= //
        Tensor<double, 2> F_ad("F_ad", ntno_ijk, ntno_ijk); {
            F_ad.zero();
            // e_tno (non-t1 contribution)
            for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                F_ad(a_ijk, a_ijk) = (*e_tno_[ijk])(a_ijk);
            }

            // J contribution
            einsum(1.0, Indices{index::a, index::d}, &F_ad, 2.0, Indices{index::Q, index::a, index::d}, q_vv_[ijk], Indices{index::Q}, gamma_Q);
            // K contribution
            Tensor<double, 3> F_ad_K_temp("F_ad_K_temp", naux_ijk, ntno_ijk, nlmo_ijk);
            einsum(0.0, Indices{index::Q, index::a, index::m}, &F_ad_K_temp, 1.0, Indices{index::Q, index::a, index::e}, q_vv_[ijk], Indices{index::m, index::e}, T_n_ijk_[ijk]);
            Tensor<double, 3> F_ad_K_temp2("F_ad_K_temp", naux_ijk, nlmo_ijk, ntno_ijk);
            sort(Indices{index::Q, index::m, index::a}, &F_ad_K_temp2, Indices{index::Q, index::a, index::m}, F_ad_K_temp);
            einsum(1.0, Indices{index::a, index::d}, &F_ad, -1.0, Indices{index::Q, index::m, index::a}, F_ad_K_temp2, Indices{index::Q, index::m, index::d}, q_ov_[ijk]);

            // Add the F_ld contribution to F_ad
            einsum(1.0, Indices{index::a, index::d}, &F_ad, -1.0, Indices{index::m, index::a}, T_n_ijk_[ijk], Indices{index::m, index::d}, F_ld);
        }

        // outfile->Printf("We magnify Your name!\n");

        // ==> Stopping point 10/22 10:02 AM <== //

        // rho_dbci, dbcj, dbck (Lesiuk Eq. 16)
        std::vector<Tensor<double, 3>> rho_dbck_list(ijk_idx.size());

        // T_lm amplitudes
        Tensor<double, 4> T_lm("T_lm", nlmo_ijk, nlmo_ijk, ntno_ijk, ntno_ijk);
        T_lm.zero();
        for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
            int l = lmotriplet_to_lmos_[ijk][l_ijk];

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int lm = i_j_to_ij_[l][m];
                if (lm == -1) continue;
                int lm_idx = (l_ijk > m_ijk) ? m_ijk * nlmo_ijk + l_ijk : l_ijk * nlmo_ijk + m_ijk;

                auto T_lm_ijk = linalg::triplet(S_ijk_lm_[ijk][lm_idx], T_iajb_[lm], S_ijk_lm_[ijk][lm_idx], false, false, true);
                ::memcpy(&T_lm(l_ijk, m_ijk, 0, 0), T_lm_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
            }
        }

        // K_dble integrals
        Tensor<double, 4> K_dble("K_dble", ntno_ijk, ntno_ijk, nlmo_ijk, ntno_ijk);
        einsum(0.0, Indices{index::d, index::b, index::l, index::e}, &K_dble, 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::l, index::e}, q_ov_[ijk]);
        Tensor<double, 4> K_dble_T("K_dble_T", ntno_ijk, ntno_ijk, nlmo_ijk, ntno_ijk);
        sort(Indices{index::e, index::b, index::l, index::d}, &K_dble_T, Indices{index::d, index::b, index::l, index::e}, K_dble);
        Tensor<double, 4> K_dble_T2("K_dble_T2", nlmo_ijk, ntno_ijk, ntno_ijk, ntno_ijk);
        sort(Indices{index::l, index::c, index::e, index::d}, &K_dble_T2, Indices{index::d, index::c, index::l, index::e}, K_dble);

        // K_ldme integrals
        Tensor<double, 4> K_ldme("K_ldme", nlmo_ijk, ntno_ijk, nlmo_ijk, ntno_ijk);
        einsum(0.0, Indices{index::l, index::d, index::m, index::e}, &K_ldme, 1.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q, index::m, index::e}, q_ov_[ijk]);
        Tensor<double, 4> K_ldme_T("K_ldme_T", nlmo_ijk, nlmo_ijk, ntno_ijk, ntno_ijk);
        sort(Indices{index::l, index::m, index::d, index::e}, &K_ldme_T, Indices{index::l, index::d, index::m, index::e}, K_ldme);
        Tensor<double, 4> K_ldme_T2("K_ldme_T2", nlmo_ijk, ntno_ijk, nlmo_ijk, ntno_ijk);
        sort(Indices{index::l, index::e, index::m, index::d}, &K_ldme_T2, Indices{index::l, index::d, index::m, index::e}, K_ldme);

        std::vector<std::vector<SharedMatrix>> S_ijk_li_list = {S_ijk_il_[ijk], S_ijk_jl_[ijk], S_ijk_kl_[ijk]};
        std::vector<std::vector<SharedMatrix>> S_ijk_mli_list = {S_ijk_mli_[ijk], S_ijk_mlj_[ijk], S_ijk_mlk_[ijk]};

        for (int idx = 0; idx < ijk_idx.size(); ++idx) {
            int i = ijk_idx[idx];

            rho_dbck_list[idx] = Tensor<double, 3>("rho_dbck_list", ntno_ijk, ntno_ijk, ntno_ijk);
            einsum(0.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::c}, q_iv_t1_list[idx]);

            Tensor<double, 3> T_li("T_li", nlmo_ijk, ntno_ijk, ntno_ijk);
            Tensor<double, 3> U_li("U_li", nlmo_ijk, ntno_ijk, ntno_ijk);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int li = i_j_to_ij_[l][i];

                auto T_li_ijk = linalg::triplet(S_ijk_li_list[idx][l_ijk], T_iajb_[li], S_ijk_li_list[idx][l_ijk], false, false, true);
                auto U_li_ijk = linalg::triplet(S_ijk_li_list[idx][l_ijk], Tt_iajb_[li], S_ijk_li_list[idx][l_ijk], false, false, true);

                ::memcpy(&T_li(l_ijk, 0, 0), T_li_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
                ::memcpy(&U_li(l_ijk, 0, 0), U_li_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            } // end l_ijk

            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], -1.0, Indices{index::l, index::d}, F_ld, Indices{index::l, index::b, index::c}, T_li);

            Tensor<double, 3> K_limd("K_limd", nlmo_ijk, nlmo_ijk, ntno_ijk);
            einsum(0.0, Indices{index::l, index::m, index::d}, &K_limd, 1.0, Indices{index::Q, index::l}, q_io_t1_list[idx], Indices{index::Q, index::m, index::d}, q_ov_[ijk]);
            Tensor<double, 3> K_limd_T("K_limd_T", nlmo_ijk, nlmo_ijk, ntno_ijk);
            sort(Indices{index::m, index::l, index::d}, &K_limd_T, Indices{index::l, index::m, index::d}, K_limd);
            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], 1.0, Indices{index::m, index::l, index::d}, K_limd_T, Indices{index::m, index::l, index::b, index::c}, T_lm);

            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], 1.0, Indices{index::d, index::b, index::l, index::e}, K_dble, Indices{index::l, index::e, index::c}, U_li);

            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], -1.0, Indices{index::d, index::b, index::l, index::e}, K_dble_T, Indices{index::l, index::e, index::c}, T_li);

            Tensor<double, 3> T_li_T("T_li_T", nlmo_ijk, ntno_ijk, ntno_ijk);
            sort(Indices{index::l, index::e, index::b}, &T_li_T, Indices{index::l, index::b, index::e}, T_li);
            Tensor<double, 3> rho_dbck_temp("rho_dbck_temp", ntno_ijk, ntno_ijk, ntno_ijk);
            einsum(0.0, Indices{index::d, index::c, index::b}, &rho_dbck_temp, 1.0, Indices{index::d, index::c, index::l, index::e}, K_dble_T, Indices{index::l, index::e, index::b}, T_li_T);
            Tensor<double, 3> rho_dbck_temp2("rho_dbck_temp2", ntno_ijk, ntno_ijk, ntno_ijk);
            sort(Indices{index::d, index::b, index::c}, &rho_dbck_temp2, Indices{index::d, index::c, index::b}, rho_dbck_temp);
            rho_dbck_list[idx] -= rho_dbck_temp2;

            // Triples terms (ca-nasty)
            Tensor<double, 5> T_mli_large("T_mli", nlmo_ijk, nlmo_ijk, ntno_ijk, ntno_ijk, ntno_ijk);
            T_mli_large.zero();

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];

                for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                    int l = lmotriplet_to_lmos_[ijk][l_ijk];
                    int mli_dense = m * naocc * naocc + l * naocc + i;
                    if (!i_j_k_to_ijk_.count(mli_dense)) continue;
                    int mli = i_j_k_to_ijk_[mli_dense];

                    int ml_idx = (m_ijk > l_ijk) ? (l_ijk * nlmo_ijk + m_ijk) : (m_ijk * nlmo_ijk + l_ijk);
                    auto S_ijk_mli = S_ijk_mli_list[idx][ml_idx];

                    // (2 t_{mli}^{ebc} - t_{mli}^{cbe} - t_{mli}^{bec})
                    auto T_mli_psi = matmul_3d(triples_permuter(T_iajbkc_[mli], m, l, i), S_ijk_mli, n_tno_[mli], n_tno_[ijk]);

                    Tensor<double, 3> T_mli("T_mli", ntno_ijk, ntno_ijk, ntno_ijk);
                    ::memcpy(T_mli.data(), T_mli_psi->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

                    Tensor<double, 3> T_lmi("T_lmi", ntno_ijk, ntno_ijk, ntno_ijk);
                    sort(Indices{index::b, index::a, index::c}, &T_lmi, Indices{index::a, index::b, index::c}, T_mli);

                    Tensor<double, 3> T_ilm("T_ilm", ntno_ijk, ntno_ijk, ntno_ijk);
                    sort(Indices{index::c, index::b, index::a}, &T_ilm, Indices{index::a, index::b, index::c}, T_mli);

                    T_mli *= 2.0;
                    T_mli -= T_lmi;
                    T_mli -= T_ilm;
                    
                    ::memcpy(&T_mli_large(m_ijk, l_ijk, 0, 0, 0), T_mli.data(), ntno_ijk * ntno_ijk * ntno_ijk * sizeof(double));
                } // end m_ijk
            } // end l_ijk
            einsum(1.0, Indices{index::d, index::b, index::c}, &rho_dbck_list[idx], -1.0, Indices{index::m, index::l, index::e, index::d}, K_ldme_T, Indices{index::m, index::l, index::e, index::b, index::c}, T_mli_large);
        } // end idx

        // outfile->Printf("You alone deserve the glory\n");

        // => Stopping point 10/22 3:45 PM <= //

        // Form rho_ljck intermediate (Lesiuk Eq. 15)
        std::vector<Tensor<double, 2>> rho_ljck_list;

        std::vector<std::tuple<int, int, int>> long_perms = {std::make_tuple(i, j, k), std::make_tuple(i, k, j),
                                                                std::make_tuple(j, i, k), std::make_tuple(j, k, i),
                                                                std::make_tuple(k, i, j), std::make_tuple(k, j, i)};

        std::vector<std::tuple<int, int, int>> long_perms_idx = {std::make_tuple(0, 1, 2), std::make_tuple(0, 2, 1),
                                                                    std::make_tuple(1, 0, 2), std::make_tuple(1, 2, 0),
                                                                    std::make_tuple(2, 0, 1), std::make_tuple(2, 1, 0)};
        
        std::vector<SharedMatrix> S_ijk_jk_list = {S_ijk_jk_[ijk], S_ijk_jk_[ijk], S_ijk_ik_[ijk], 
                                                    S_ijk_ik_[ijk], S_ijk_ij_[ijk], S_ijk_ij_[ijk]};
        std::vector<std::vector<SharedMatrix>> S_ijk_jmk_list = {S_ijk_ljk_[ijk], S_ijk_ljk_[ijk], S_ijk_ilk_[ijk], 
                                                                    S_ijk_ilk_[ijk], S_ijk_ijl_[ijk], S_ijk_ijl_[ijk]};

        for (int idx = 0; idx < long_perms.size(); ++idx) {
            auto &[i, j, k] = long_perms[idx];
            auto &[i_idx, j_idx, k_idx] = long_perms_idx[idx];
            int jk = i_j_to_ij_[j][k];
            
            Tensor<double, 2> rho_ljck("rho_ljck", nlmo_ijk, ntno_ijk);
            einsum(0.0, Indices{index::l, index::c}, &rho_ljck, 1.0, Indices{index::Q, index::l}, q_io_t1_list[j_idx], Indices{index::Q, index::c}, q_iv_t1_list[k_idx]);

            // => CHECKPOINT 10/25 3:40 PM <= //

            Tensor<double, 3> T_jm("T_jm", nlmo_ijk, ntno_ijk, ntno_ijk);
            Tensor<double, 3> T_mk("T_mk", nlmo_ijk, ntno_ijk, ntno_ijk);
            Tensor<double, 3> U_mk("U_mk", nlmo_ijk, ntno_ijk, ntno_ijk);

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int mk = i_j_to_ij_[m][k], jm = i_j_to_ij_[j][m];

                auto S_ijk_jm = S_ijk_li_list[j_idx][m_ijk];
                auto T_jm_ijk = linalg::triplet(S_ijk_jm, T_iajb_[jm], S_ijk_jm, false, false, true);
                ::memcpy(&T_jm(m_ijk, 0, 0), T_jm_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

                auto S_ijk_mk = S_ijk_li_list[k_idx][m_ijk];
                auto T_mk_ijk = linalg::triplet(S_ijk_mk, T_iajb_[mk], S_ijk_mk, false, false, true);
                auto U_mk_ijk = linalg::triplet(S_ijk_mk, Tt_iajb_[mk], S_ijk_mk, false, false, true);

                ::memcpy(&T_mk(m_ijk, 0, 0), T_mk_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
                ::memcpy(&U_mk(m_ijk, 0, 0), U_mk_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            }

            Tensor<double, 3> K_mdlj("K_mdlj", nlmo_ijk, ntno_ijk, nlmo_ijk);
            einsum(0.0, Indices{index::m, index::d, index::l}, &K_mdlj, 1.0, Indices{index::Q, index::m, index::d}, q_ov_[ijk], Indices{index::Q, index::l}, q_io_t1_list[j_idx]);

            einsum(1.0, Indices{index::l, index::c}, &rho_ljck, 1.0, Indices{index::m, index::d, index::l}, K_mdlj, Indices{index::m, index::d, index::c}, U_mk);

            Tensor<double, 3> K_mdlj_T("K_mdlj_T", nlmo_ijk, nlmo_ijk, ntno_ijk);
            sort(Indices{index::l, index::m, index::d}, &K_mdlj_T, Indices{index::l, index::d, index::m}, K_mdlj);
            einsum(1.0, Indices{index::l, index::c}, &rho_ljck, -1.0, Indices{index::l, index::m, index::d}, K_mdlj_T, Indices{index::m, index::d, index::c}, T_mk);

            Tensor<double, 3> K_ldmk("K_ldmk", nlmo_ijk, ntno_ijk, nlmo_ijk);
            einsum(0.0, Indices{index::l, index::d, index::m}, &K_ldmk, 1.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q, index::m}, q_io_t1_list[k_idx]);

            Tensor<double, 3> K_ldmk_T("K_ldmk_T", nlmo_ijk, nlmo_ijk, ntno_ijk);
            sort(Indices{index::l, index::m, index::d}, &K_ldmk_T, Indices{index::l, index::d, index::m}, K_ldmk);

            einsum(1.0, Indices{index::l, index::c}, &rho_ljck, -1.0, Indices{index::l, index::m, index::d}, K_ldmk_T, Indices{index::m, index::d, index::c}, T_jm);

            Tensor<double, 2> T_jk("T_jk", ntno_ijk, ntno_ijk);

            auto S_ijk_jk = S_ijk_jk_list[idx];
            auto T_jk_ijk = linalg::triplet(S_ijk_jk, T_iajb_[jk], S_ijk_jk, false, false, true);
            ::memcpy(T_jk.data(), T_jk_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            einsum(1.0, Indices{index::l, index::c}, &rho_ljck, 1.0, Indices{index::l, index::c, index::e, index::d}, K_dble_T2, Indices{index::e, index::d}, T_jk);

            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int jmk_dense = j * naocc * naocc + m * naocc + k;
                if (!i_j_k_to_ijk_.count(jmk_dense)) continue;

                Tensor<double, 2> q_mv_slice = q_ov_[ijk](All, m_ijk, All);
                Tensor<double, 3> K_lde("K_lde", nlmo_ijk, ntno_ijk, ntno_ijk);
                einsum(0.0, Indices{index::l, index::d, index::e}, &K_lde, 1.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q, index::e}, q_mv_slice);

                int jmk = i_j_k_to_ijk_[jmk_dense];
                auto S_ijk_jmk = S_ijk_jmk_list[idx][m_ijk];

                // (2 t_{jmk}^{dec} - t_{jmk}^{dce} - t_{jmk}^{edc})
                auto T_jmk_psi = matmul_3d(triples_permuter(T_iajbkc_[jmk], j, m, k), S_ijk_jmk, n_tno_[jmk], n_tno_[ijk]);

                Tensor<double, 3> T_jmk("T_jmk", ntno_ijk, ntno_ijk, ntno_ijk);
                ::memcpy(T_jmk.data(), T_jmk_psi->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

                Tensor<double, 3> T_jkm("T_jkm", ntno_ijk, ntno_ijk, ntno_ijk);
                sort(Indices{index::a, index::c, index::b}, &T_jkm, Indices{index::a, index::b, index::c}, T_jmk);

                Tensor<double, 3> T_mjk("T_mjk", ntno_ijk, ntno_ijk, ntno_ijk);
                sort(Indices{index::b, index::a, index::c}, &T_mjk, Indices{index::a, index::b, index::c}, T_jmk);

                T_jmk *= 2.0;
                T_jmk -= T_mjk;
                T_jmk -= T_jkm;
                
                einsum(1.0, Indices{index::l, index::c}, &rho_ljck, 1.0, Indices{index::l, index::d, index::e}, K_lde, Indices{index::d, index::e, index::c}, T_jmk);
            }

            rho_ljck_list.push_back(rho_ljck);
        }

        // outfile->Printf("The honor and the praise\n");

        // Lesiuk Eq. 11a (flip sign)
        std::vector<Tensor<double, 3>> Wperms(long_perms.size());

        std::vector<SharedMatrix> S_ijk_ij_list = {S_ijk_ij_[ijk], S_ijk_ik_[ijk], S_ijk_ij_[ijk],
                                                    S_ijk_jk_[ijk], S_ijk_ik_[ijk], S_ijk_jk_[ijk]};
        std::vector<std::vector<SharedMatrix>> S_ijk_il_list = {S_ijk_il_[ijk], S_ijk_il_[ijk], S_ijk_jl_[ijk],
                                                                    S_ijk_jl_[ijk], S_ijk_kl_[ijk], S_ijk_kl_[ijk]};

        for (int idx = 0; idx < long_perms.size(); ++idx) {
            auto &[i, j, k] = long_perms[idx];
            auto &[i_idx, j_idx, k_idx] = long_perms_idx[idx];
            int ij = i_j_to_ij_[i][j];

            Wperms[idx] = Tensor("Wperm", n_tno_[ijk], n_tno_[ijk], n_tno_[ijk]);

            /// => rho_dbck contribution <= ///

            // Compute overlap between TNOs of triplet ijk and PNOs of pair ij
            auto S_ijk_ij = S_ijk_ij_list[idx];
            auto T_ij = linalg::triplet(S_ijk_ij, T_iajb_[ij], S_ijk_ij, false, false, true);
            Tensor<double, 2> T_ij_einsums("T_ij", n_tno_[ijk], n_tno_[ijk]);
            ::memcpy(T_ij_einsums.data(), T_ij->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));

            einsum(0.0, Indices{index::a, index::b, index::c}, &Wperms[idx], 1.0, Indices{index::a, index::d}, T_ij_einsums, 
                    Indices{index::d, index::b, index::c}, rho_dbck_list[k_idx]);

            /// => rho_ljck contribution <= //
            Tensor<double, 3> T_il_einsums("T_il", nlmo_ijk, n_tno_[ijk], n_tno_[ijk]);

            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];

                auto S_ijk_il = S_ijk_il_list[idx][l_ijk];
                auto T_il = linalg::triplet(S_ijk_il, T_iajb_[il], S_ijk_il, false, false, true);
                ::memcpy(&T_il_einsums(l_ijk, 0, 0), T_il->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            } // end l_ijk
            
            einsum(1.0, Indices{index::a, index::b, index::c}, &Wperms[idx], -1.0, Indices{index::l, index::a, index::b}, T_il_einsums, 
                    Indices{index::l, index::c}, rho_ljck_list[idx]);
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

        // outfile->Printf("Lord Jesus, this song is forever Yours\n");

        // => STEP 2: SHORT PERMUTATION TERMS <= //
        std::vector<std::tuple<int, int, int>> short_perms = {std::make_tuple(i, j, k), std::make_tuple(j, i, k), std::make_tuple(k, j, i)};
        std::vector<std::tuple<int, int, int>> short_perms_idx = {std::make_tuple(0, 1, 2), std::make_tuple(1, 0, 2), std::make_tuple(2, 1, 0)};

        // chi_li, chi_lj, chi_lk (Lesiuk Eq. 12a)
        std::vector<Tensor<double, 1>> chi_li_list(ijk_idx.size());

        for (int idx = 0; idx < ijk_idx.size(); ++idx) {
            int i = ijk_idx[idx];

            chi_li_list[idx] = F_li_list[idx];

            Tensor<double, 3> U_mi("U_mi", nlmo_ijk, ntno_ijk, ntno_ijk);
            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int mi = i_j_to_ij_[m][i];

                auto S_ijk_mi = S_ijk_li_list[idx][m_ijk];
                auto U_mi_ijk = linalg::triplet(S_ijk_mi, Tt_iajb_[mi], S_ijk_mi, false, false, true);
                ::memcpy(&U_mi(m_ijk, 0, 0), U_mi_ijk->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));
            }

            Tensor<double, 2> chi_li_temp("chi_li_temp", naux_ijk, ntno_ijk);
            einsum(0.0, Indices{index::Q, index::d}, &chi_li_temp, 1.0, Indices{index::Q, index::m, index::e}, q_ov_[ijk], Indices{index::m, index::e, index::d}, U_mi);
            einsum(1.0, Indices{index::l}, &chi_li_list[idx], 1.0, Indices{index::Q, index::d, index::l}, q_vo, Indices{index::Q, index::d}, chi_li_temp);
        }

        // outfile->Printf("A thousand hallelujahs and a thousand more\n");

        // Lesiuk Eq. 12b
        Tensor<double, 2> chi_ad("chi_ad", ntno_ijk, ntno_ijk);
        chi_ad = F_ad;

        Tensor<double, 4> U_lm = T_lm;
        Tensor<double, 4> T_lm_T("T_lm_T", nlmo_ijk, nlmo_ijk, ntno_ijk, ntno_ijk);
        sort(Indices{index::l, index::m, index::e, index::d}, &T_lm_T, Indices{index::l, index::m, index::d, index::e}, T_lm);
        U_lm *= 2.0;
        U_lm -= T_lm_T;
        einsum(1.0, Indices{index::a, index::d}, &chi_ad, -1.0, Indices{index::m, index::l, index::e, index::a}, U_lm, Indices{index::m, index::l, index::e, index::d}, K_ldme_T);

        // outfile->Printf("Who else would die for our redemption\n");

        // chi_jklm, chi_iklm, chi_jilm (Lesiuk Eq. 13a)
        std::vector<Tensor<double, 2>> chi_jk_list(short_perms.size());

        std::vector<SharedMatrix> S_ijk_jk_list_short = {S_ijk_jk_[ijk], S_ijk_ik_[ijk], S_ijk_ij_[ijk]};

        for (int idx = 0; idx < short_perms.size(); ++idx) {
            auto &[i, j, k] = short_perms[idx];
            auto &[i_idx, j_idx, k_idx] = short_perms_idx[idx];
            int jk = i_j_to_ij_[j][k];

            chi_jk_list[idx] = Tensor<double, 2>("chi_jk", nlmo_ijk, nlmo_ijk);
            einsum(0.0, Indices{index::l, index::m}, &chi_jk_list[idx], 1.0, Indices{index::Q, index::l}, q_io_t1_list[j_idx], Indices{index::Q, index::m}, q_io_t1_list[k_idx]);

            auto S_ijk_jk = S_ijk_jk_list_short[idx];
            auto T_jk_psi = linalg::triplet(S_ijk_jk, T_iajb_[jk], S_ijk_jk, false, false, true);

            Tensor<double, 2> T_jk("T_jk", ntno_ijk, ntno_ijk);
            ::memcpy(T_jk.data(), T_jk_psi->get_pointer(), ntno_ijk * ntno_ijk * sizeof(double));

            Tensor<double, 3> chi_jk_temp("chi_jk_temp", naux_ijk, nlmo_ijk, ntno_ijk);
            einsum(0.0, Indices{index::Q, index::l, index::e}, &chi_jk_temp, 1.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::d, index::e}, T_jk);
            
            Tensor<double, 3> chi_jk_temp_T("chi_jk_temp_T", naux_ijk, ntno_ijk, nlmo_ijk);
            sort(Indices{index::Q, index::e, index::l}, &chi_jk_temp_T, Indices{index::Q, index::l, index::e}, chi_jk_temp);

            einsum(1.0, Indices{index::l, index::m}, &chi_jk_list[idx], 1.0, Indices{index::Q, index::e, index::l}, chi_jk_temp_T, Indices{index::Q, index::e, index::m}, q_vo);
        }

        // outfile->Printf("Whose resurrection means I'll rise\n");

        // chi_bdce (Lesiuk Eq. 13b, different order b/c different T1 convention)
        // Praise... to the Lord... only 20 lines
        Tensor<double, 4> chi_dbec("chi_dbec", ntno_ijk, ntno_ijk, ntno_ijk, ntno_ijk);
        einsum(0.0, Indices{index::d, index::b, index::e, index::c}, &chi_dbec, 1.0, Indices{index::Q, index::d, index::b}, q_vv_t1, Indices{index::Q, index::e, index::c}, q_vv_t1);

        Tensor<double, 4> chi_dbec_temp("chi_dbec_temp", ntno_ijk, ntno_ijk, ntno_ijk, ntno_ijk);
        einsum(0.0, Indices{index::d, index::e, index::b, index::c}, &chi_dbec_temp, 1.0, Indices{index::l, index::m, index::d, index::e}, K_ldme_T, Indices{index::l, index::m, index::b, index::c}, T_lm);
        
        Tensor<double, 4> chi_dbec_temp_T("chi_dbec_temp_T", ntno_ijk, ntno_ijk, ntno_ijk, ntno_ijk);
        sort(Indices{index::d, index::b, index::e, index::c}, &chi_dbec_temp_T, Indices{index::d, index::e, index::b, index::c}, chi_dbec_temp);
        chi_dbec += chi_dbec_temp_T;

        // Rearrange chi_dbec
        sort(Indices{index::d, index::e, index::b, index::c}, &chi_dbec_temp, Indices{index::d, index::b, index::e, index::c}, chi_dbec);
        chi_dbec = chi_dbec_temp;

        // outfile->Printf("There isn't time enough to sing of what You've done\n");

        // Lesiuk Eq. 14a (chi_lida, chi_ljda, chi_lkda)
        std::vector<Tensor<double, 3>> chi_lida_list(short_perms.size());

        for (int idx = 0; idx < ijk_idx.size(); ++idx) {
            int i = ijk_idx[idx];

            chi_lida_list[idx] = Tensor<double, 3>("chi_lida", nlmo_ijk, ntno_ijk, ntno_ijk);
            einsum(0.0, Indices{index::l, index::d, index::a}, &chi_lida_list[idx], 1.0, Indices{index::Q, index::l}, q_io_t1_list[idx], Indices{index::Q, index::d, index::a}, q_vv_t1);

            Tensor<double, 3> T_im("T_im", nlmo_ijk, ntno_ijk, ntno_ijk);
            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int im = i_j_to_ij_[i][m];

                auto S_ijk_im = S_ijk_li_list[idx][m_ijk];
                auto T_im_ijk = linalg::triplet(S_ijk_im, T_iajb_[im], S_ijk_im, false, false, true);
                ::memcpy(&T_im(m_ijk, 0, 0), T_im_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            }

            einsum(1.0, Indices{index::l, index::d, index::a}, &chi_lida_list[idx], -1.0, Indices{index::m, index::e, index::l, index::d}, K_ldme_T2, Indices{index::m, index::e, index::a}, T_im);
        }

        // outfile->Printf("But I have eternity to try\n");

        // Lesiuk Eq. 14b (chi_ldai, chi_ldaj, chi_ldak)
        std::vector<Tensor<double, 3>> chi_ldai_list(short_perms.size());
        for (int idx = 0; idx < ijk_idx.size(); ++idx) {
            int i = ijk_idx[idx];

            chi_ldai_list[idx] = Tensor<double, 3>("chi_ldai", nlmo_ijk, ntno_ijk, ntno_ijk);
            einsum(0.0, Indices{index::l, index::d, index::a}, &chi_ldai_list[idx], 1.0, Indices{index::Q, index::l, index::d}, q_ov_[ijk], Indices{index::Q, index::a}, q_iv_t1_list[idx]);

            Tensor<double, 3> T_mi("T_mi", nlmo_ijk, ntno_ijk, ntno_ijk);
            Tensor<double, 3> U_mi("U_mi", nlmo_ijk, ntno_ijk, ntno_ijk);
            for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int mi = i_j_to_ij_[m][i];

                auto S_ijk_mi = S_ijk_li_list[idx][m_ijk];

                auto T_mi_ijk = linalg::triplet(S_ijk_mi, T_iajb_[mi], S_ijk_mi, false, false, true);
                auto U_mi_ijk = linalg::triplet(S_ijk_mi, Tt_iajb_[mi], S_ijk_mi, false, false, true);
                ::memcpy(&T_mi(m_ijk, 0, 0), T_mi_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
                ::memcpy(&U_mi(m_ijk, 0, 0), U_mi_ijk->get_pointer(), n_tno_[ijk] * n_tno_[ijk] * sizeof(double));
            }

            // (l, e, m, d) (m, a, e) => (a, d, l)
            einsum(1.0, Indices{index::l, index::d, index::a}, &chi_ldai_list[idx], -1.0, Indices{index::l, index::d, index::m, index::e}, K_ldme_T2, Indices{index::m, index::e, index::a}, T_mi);
            // (l, d, m, e) (m, a, e) => (a, d, l)
            einsum(1.0, Indices{index::l, index::d, index::a}, &chi_ldai_list[idx], 1.0, Indices{index::l, index::d, index::m, index::e}, K_ldme, Indices{index::m, index::e, index::a}, U_mi);
        }

        // outfile->Printf("Praise to the Lord\n");

        std::vector<Tensor<double, 3>> Vperms(short_perms.size());

        std::vector<std::vector<SharedMatrix>> S_ijk_ljk_list = {S_ijk_ljk_[ijk], S_ijk_ilk_[ijk], S_ijk_ijl_[ijk]};
        std::vector<std::vector<SharedMatrix>> S_ijk_ilm_list = {S_ijk_mli_[ijk], S_ijk_mlj_[ijk], S_ijk_mlk_[ijk]};

        for (int idx = 0; idx < short_perms.size(); ++idx) {
            auto &[i, j, k] = short_perms[idx];
            auto &[i_idx, j_idx, k_idx] = short_perms_idx[idx];

            Vperms[idx] = Tensor<double, 3>("V_iajbkc", ntno_ijk, ntno_ijk, ntno_ijk);

            // T_ijk contributions
            Tensor<double, 3> T_ijk("T_ijk", ntno_ijk, ntno_ijk, ntno_ijk);
            auto T_ijk_psi = triples_permuter(T_iajbkc_[ijk], i, j, k);
            ::memcpy(T_ijk.data(), T_ijk_psi->get_pointer(), ntno_ijk * ntno_ijk * ntno_ijk * sizeof(double));
            einsum(0.0, Indices{index::a, index::b, index::c}, &Vperms[idx], 1.0, Indices{index::a, index::d}, chi_ad, Indices{index::d, index::b, index::c}, T_ijk);

            einsum(1.0, Indices{index::a, index::b, index::c}, &Vperms[idx], 1.0, Indices{index::a, index::d, index::e}, T_ijk, Indices{index::d, index::e, index::b, index::c}, chi_dbec);

            // T_ljk contributions
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int ljk_dense = l * naocc * naocc + j * naocc + k;
                if (!i_j_k_to_ijk_.count(ljk_dense)) continue;
                int ljk = i_j_k_to_ijk_[ljk_dense];
                
                auto S_ijk_ljk = S_ijk_ljk_list[idx][l_ijk];

                Tensor<double, 3> T_ljk("T_ljk", ntno_ijk, ntno_ijk, ntno_ijk);
                auto T_ljk_psi = matmul_3d(triples_permuter(T_iajbkc_[ljk], l, j, k), S_ijk_ljk, n_tno_[ljk], n_tno_[ijk]);
                ::memcpy(T_ljk.data(), T_ljk_psi->get_pointer(), ntno_ijk * ntno_ijk * ntno_ijk * sizeof(double));

                Tensor<double, 3> T_ljk_clone = T_ljk;
                T_ljk_clone *= -(chi_li_list[i_idx])(l_ijk);
                Vperms[idx] += T_ljk_clone;

                // chi_lida contributions
                Tensor<double, 2> chi_lida_temp = chi_lida_list[i_idx](l_ijk, All, All);
                einsum(1.0, Indices{index::a, index::b, index::c}, &Vperms[idx], -1.0, Indices{index::d, index::a}, chi_lida_temp, Indices{index::d, index::b, index::c}, T_ljk);
                Tensor<double, 3> Vperm_temp("Vperm_temp", ntno_ijk, ntno_ijk, ntno_ijk);
                sort(Indices{index::b, index::a, index::c}, &Vperm_temp, Indices{index::a, index::b, index::c}, T_ljk);
                Tensor<double, 3> Vperm_temp2("Vperm_temp2", ntno_ijk, ntno_ijk, ntno_ijk);
                einsum(0.0, Indices{index::b, index::a, index::c}, &Vperm_temp2, 1.0, Indices{index::d, index::b}, chi_lida_temp, Indices{index::d, index::a, index::c}, Vperm_temp);
                sort(Indices{index::a, index::b, index::c}, &Vperm_temp, Indices{index::b, index::a, index::c}, Vperm_temp2);
                Vperms[idx] -= Vperm_temp;
                einsum(1.0, Indices{index::a, index::b, index::c}, &Vperms[idx], -1.0, Indices{index::d, index::c}, chi_lida_temp, Indices{index::a, index::b, index::d}, T_ljk);

                // chi_ldai contributions
                Tensor<double, 3> T_jlk("T_jlk", ntno_ijk, ntno_ijk, ntno_ijk);
                sort(Indices{index::b, index::a, index::c}, &T_jlk, Indices{index::a, index::b, index::c}, T_ljk);

                Tensor<double, 3> T_kjl("T_kjl", ntno_ijk, ntno_ijk, ntno_ijk);
                sort(Indices{index::c, index::b, index::a}, &T_kjl, Indices{index::a, index::b, index::c}, T_ljk);

                T_ljk *= 2.0;
                T_ljk -= T_jlk;
                T_ljk -= T_kjl;

                Tensor<double, 2> chi_ldai_temp = chi_ldai_list[i_idx](l_ijk, All, All);
                einsum(1.0, Indices{index::a, index::b, index::c}, &Vperms[idx], 1.0, Indices{index::d, index::a}, chi_ldai_temp, Indices{index::d, index::b, index::c}, T_ljk);

                /*
                for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                            for (int d_ijk = 0; d_ijk < ntno_ijk; ++d_ijk) {
                                // chi_lida contributions
                                Vperms[idx](a_ijk, b_ijk, c_ijk) -= chi_lida_list[i_idx](l_ijk, d_ijk, a_ijk) * (T_ljk)(d_ijk, b_ijk, c_ijk)
                                    + chi_lida_list[i_idx](l_ijk, d_ijk, b_ijk) * (T_ljk)(a_ijk, d_ijk, c_ijk) + chi_lida_list[i_idx](l_ijk, d_ijk, c_ijk) * (T_ljk)(a_ijk, b_ijk, d_ijk);

                                // chi_ldai contributions
                                Vperms[idx](a_ijk, b_ijk, c_ijk) += chi_ldai_list[i_idx](l_ijk, d_ijk, a_ijk) * (2 * T_ljk(d_ijk, b_ijk, c_ijk) 
                                                                    - T_ljk(c_ijk, b_ijk, d_ijk) - T_ljk(b_ijk, d_ijk, c_ijk));
                            } // end d_ijk
                        } // end c_ijk
                    } // end b_ijk
                } // end a_ijk
                */
            }

            // T_ilm contributions
            for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];

                for (int m_ijk = 0; m_ijk < nlmo_ijk; ++m_ijk) {
                    int m = lmotriplet_to_lmos_[ijk][m_ijk];
                    int ilm_dense = i * naocc * naocc + l * naocc + m;
                    if (!i_j_k_to_ijk_.count(ilm_dense)) continue;
                    int ilm = i_j_k_to_ijk_[ilm_dense];
                    int lm_idx = (l_ijk > m_ijk) ? m_ijk * nlmo_ijk + l_ijk : l_ijk * nlmo_ijk + m_ijk;

                    auto S_ijk_ilm = S_ijk_ilm_list[idx][lm_idx];

                    Tensor<double, 3> T_ilm("T_ilm", ntno_ijk, ntno_ijk, ntno_ijk);
                    auto T_ilm_psi = matmul_3d(triples_permuter(T_iajbkc_[ilm], i, l, m), S_ijk_ilm, n_tno_[ilm], n_tno_[ijk]);
                    ::memcpy(T_ilm.data(), T_ilm_psi->get_pointer(), ntno_ijk * ntno_ijk * ntno_ijk * sizeof(double));

                    Tensor<double, 3> T_ilm_clone = T_ilm;
                    T_ilm_clone *= (chi_jk_list[idx])(l_ijk, m_ijk);
                    Vperms[idx] += T_ilm_clone;
                } // end m_ijk
            } // end l_ijk
        } // end idx

        // outfile->Printf("To the lamb, to the KING of heaven\n");

        // Flush Vperms
        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) +=
                        (Vperms[0])(a_ijk, b_ijk, c_ijk) + (Vperms[1])(b_ijk, a_ijk, c_ijk) + (Vperms[2])(c_ijk, b_ijk, a_ijk);
                } // end c_ijk
            } // end b_ijk
        } // end a_ijk

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
    double e_curr = 0.0, e_prev = 0.0, e_weak = 0.0, r_curr1 = 0.0, r_curr2 = 0.0, r_curr3 = 0.0;
    bool e_converged = false, r_converged = false;

    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LCCSDT DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

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

                auto T_l_temp = linalg::doublet(S_ijk_ll_[ijk][l_ijk], T_ia_[l]);

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
        if (options_.get_bool("DLPNO_CC3")) {
            compute_R_iajbkc_cc3(R_iajbkc);
        } else {
            compute_R_iajbkc(R_iajbkc);
        }

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

        // DIIS Extrapolation
        std::vector<SharedMatrix> T_vecs;
        T_vecs.reserve(T_ia_.size() + T_iajb_.size() + T_iajbkc_.size());
        T_vecs.insert(T_vecs.end(), T_ia_.begin(), T_ia_.end());
        T_vecs.insert(T_vecs.end(), T_iajb_.begin(), T_iajb_.end());
        T_vecs.insert(T_vecs.end(), T_iajbkc_.begin(), T_iajbkc_.end());

        std::vector<SharedMatrix> R_vecs;
        R_vecs.reserve(R_ia.size() + R_iajb.size() + R_iajbkc.size());
        R_vecs.insert(R_vecs.end(), R_ia.begin(), R_ia.end());
        R_vecs.insert(R_vecs.end(), R_iajb.begin(), R_iajb.end());
        R_vecs.insert(R_vecs.end(), R_iajbkc.begin(), R_iajbkc.end());

        auto T_vecs_flat = flatten_mats(T_vecs);
        auto R_vecs_flat = flatten_mats(R_vecs);

        if (iteration == 1) {
            diis.set_error_vector_size(R_vecs_flat);
            diis.set_vector_size(T_vecs_flat);
        }

        diis.add_entry(R_vecs_flat.get(), T_vecs_flat.get());
        diis.extrapolate(T_vecs_flat.get());

        copy_flat_mats(T_vecs_flat, T_vecs);

        // evaluate energy and convergence
        e_prev = e_curr;
        e_curr = 0.0;
        e_weak = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : e_curr, e_weak)
        for (int ij = 0; ij < n_lmo_pairs; ++ij) {
            auto &[i, j] = ij_to_i_j_[ij];
            int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j];
            int ij_idx = (i < j) ? ij : ij_to_ji_[ij];

            // Update anti-symmetrized amplitudes
            Tt_iajb_[ij] = T_iajb_[ij]->clone();
            Tt_iajb_[ij]->scale(2.0);
            Tt_iajb_[ij]->subtract(T_iajb_[ij]->transpose());

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

            double e_ij = tau->vector_dot(L_iajb_[ij]);
            
            e_curr += e_ij;
            if (i_j_to_ij_strong_[i][j] == -1) e_weak += e_ij;
        }

        double r_curr1 = *max_element(R_ia_rms.begin(), R_ia_rms.end());
        double r_curr2 = *max_element(R_iajb_rms.begin(), R_iajb_rms.end());
        double r_curr3 = *max_element(R_iajbkc_rms.begin(), R_iajbkc_rms.end());

        r_converged = fabs(r_curr1) < options_.get_double("R_CONVERGENCE");
        r_converged &= fabs(r_curr2) < options_.get_double("R_CONVERGENCE");
        r_converged &= fabs(r_curr3) < options_.get_double("R_CONVERGENCE");
        e_converged = fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE");

        e_lccsdt_ = e_curr - e_weak;
        de_weak_ = e_weak;

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
    compute_tno_overlaps();
    lccsdt_iterations();

    double e_scf = variables_["SCF TOTAL ENERGY"];
    double e_ccsdt_corr = e_lccsdt_ + (e_lccsd_t_ - e_lccsd_ - E_T_) + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    double e_ccsdt_total = e_scf + e_ccsdt_corr;

    set_scalar_variable("CCSDT CORRELATION ENERGY", e_ccsdt_corr);
    set_scalar_variable("CURRENT CORRELATION ENERGY", e_ccsdt_corr);
    set_scalar_variable("CCSDT TOTAL ENERGY", e_ccsdt_total);
    set_scalar_variable("CURRENT ENERGY", e_ccsdt_total);

    print_results();

    return e_ccsdt_total;
}

void DLPNOCCSDT::print_results() {

    int naocc = i_j_to_ij_.size();
    double t1diag = 0.0;
#pragma omp parallel for reduction(+ : t1diag)
    for (int i = 0; i < naocc; ++i) {
        t1diag += T_ia_[i]->vector_dot(T_ia_[i]);
    }
    t1diag = std::sqrt(t1diag / (2.0 * naocc));
    outfile->Printf("\n  T1 Diagnostic: %8.8f \n", t1diag);
    if (t1diag > 0.02) {
        outfile->Printf("    WARNING: T1 Diagnostic is greater than 0.02, CCSD results may be unreliable!\n");
    }
    set_scalar_variable("CC T1 DIAGNOSTIC", t1diag);

    double e_total = e_lccsdt_ + (e_lccsd_t_ - e_lccsd_ - E_T_) + de_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;

    outfile->Printf("  \n");
    outfile->Printf("  Total DLPNO-CCSDT Correlation Energy: %16.12f \n", e_total);
    outfile->Printf("    LCCSDT Correlation Energy:          %16.12f \n", e_lccsdt_);
    outfile->Printf("    Weak Pair Contribution:             %16.12f \n", de_weak_);
    outfile->Printf("    Eliminated Pair MP2 Correction:     %16.12f \n", de_lmp2_eliminated_);
    outfile->Printf("    Dipole Pair Correction:             %16.12f \n", de_dipole_);
    outfile->Printf("    PNO Truncation Correction:          %16.12f \n", de_pno_total_);
    outfile->Printf("    Triples Rank Correction (T0):       %16.12f \n", e_lccsd_t_ - e_lccsd_ - E_T_);
    outfile->Printf("    Houston is (still) a delinquent\n");
    outfile->Printf("\n\n  @Total DLPNO-CCSDT Energy: %16.12f \n", variables_["SCF TOTAL ENERGY"] + e_total);
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
        /*- Perform CC3 instead of full triples? -*/
        options.add_bool("DLPNO_CC3", false);
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

