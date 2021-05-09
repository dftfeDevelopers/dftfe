// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------


#ifndef linearAlgebraOperations_h
#define linearAlgebraOperations_h

#include <elpaScalaManager.h>
#include <headers.h>
#include <operator.h>
#include "process_grid.h"
#include "scalapackWrapper.h"

namespace dftfe
{
  //
  // extern declarations for blas-lapack routines
  //
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  extern "C"
  {
    void
    dgemv_(const char *        TRANS,
           const unsigned int *M,
           const unsigned int *N,
           const double *      alpha,
           const double *      A,
           const unsigned int *LDA,
           const double *      X,
           const unsigned int *INCX,
           const double *      beta,
           double *            C,
           const unsigned int *INCY);
    void
    dgesv_(int *   n,
           int *   nrhs,
           double *a,
           int *   lda,
           int *   ipiv,
           double *b,
           int *   ldb,
           int *   info);
    void
    dscal_(const unsigned int *n,
           const double *      alpha,
           double *            x,
           const unsigned int *inc);
    void
    sscal_(const unsigned int *n,
           const float *       alpha,
           float *             x,
           const unsigned int *inc);
    void
    zscal_(const unsigned int *        n,
           const std::complex<double> *alpha,
           std::complex<double> *      x,
           const unsigned int *        inc);
    void
    zdscal_(const unsigned int *  n,
            const double *        alpha,
            std::complex<double> *x,
            const unsigned int *  inc);
    void
    daxpy_(const unsigned int *n,
           const double *      alpha,
           double *            x,
           const unsigned int *incx,
           double *            y,
           const unsigned int *incy);
    void
    dgemm_(const char *        transA,
           const char *        transB,
           const unsigned int *m,
           const unsigned int *n,
           const unsigned int *k,
           const double *      alpha,
           const double *      A,
           const unsigned int *lda,
           const double *      B,
           const unsigned int *ldb,
           const double *      beta,
           double *            C,
           const unsigned int *ldc);
    void
    sgemm_(const char *        transA,
           const char *        transB,
           const unsigned int *m,
           const unsigned int *n,
           const unsigned int *k,
           const float *       alpha,
           const float *       A,
           const unsigned int *lda,
           const float *       B,
           const unsigned int *ldb,
           const float *       beta,
           float *             C,
           const unsigned int *ldc);
    void
    dsyevd_(const char *        jobz,
            const char *        uplo,
            const unsigned int *n,
            double *            A,
            const unsigned int *lda,
            double *            w,
            double *            work,
            const unsigned int *lwork,
            int *               iwork,
            const unsigned int *liwork,
            int *               info);
    void
    dsyevr_(const char *        jobz,
            const char *        range,
            const char *        uplo,
            const unsigned int *n,
            double *            A,
            const unsigned int *lda,
            const double *      vl,
            const double *      vu,
            const unsigned int *il,
            const unsigned int *iu,
            const double *      abstol,
            const unsigned int *m,
            double *            w,
            double *            Z,
            const unsigned int *ldz,
            unsigned int *      isuppz,
            double *            work,
            const int *         lwork,
            int *               iwork,
            const int *         liwork,
            int *               info);
    void
    dsyrk_(const char *        uplo,
           const char *        trans,
           const unsigned int *n,
           const unsigned int *k,
           const double *      alpha,
           const double *      A,
           const unsigned int *lda,
           const double *      beta,
           double *            C,
           const unsigned int *ldc);
    void
    dcopy_(const unsigned int *n,
           const double *      x,
           const unsigned int *incx,
           double *            y,
           const unsigned int *incy);
    void
    scopy_(const unsigned int *n,
           const float *       x,
           const unsigned int *incx,
           float *             y,
           const unsigned int *incy);
    void
    zgemm_(const char *                transA,
           const char *                transB,
           const unsigned int *        m,
           const unsigned int *        n,
           const unsigned int *        k,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           const unsigned int *        lda,
           const std::complex<double> *B,
           const unsigned int *        ldb,
           const std::complex<double> *beta,
           std::complex<double> *      C,
           const unsigned int *        ldc);
    void
    cgemm_(const char *               transA,
           const char *               transB,
           const unsigned int *       m,
           const unsigned int *       n,
           const unsigned int *       k,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           const unsigned int *       lda,
           const std::complex<float> *B,
           const unsigned int *       ldb,
           const std::complex<float> *beta,
           std::complex<float> *      C,
           const unsigned int *       ldc);
    void
    zheevd_(const char *          jobz,
            const char *          uplo,
            const unsigned int *  n,
            std::complex<double> *A,
            const unsigned int *  lda,
            double *              w,
            std::complex<double> *work,
            const unsigned int *  lwork,
            double *              rwork,
            const unsigned int *  lrwork,
            int *                 iwork,
            const unsigned int *  liwork,
            int *                 info);
    void
    zheevr_(const char *          jobz,
            const char *          range,
            const char *          uplo,
            const unsigned int *  n,
            std::complex<double> *A,
            const unsigned int *  lda,
            const double *        vl,
            const double *        vu,
            const unsigned int *  il,
            const unsigned int *  iu,
            const double *        abstol,
            const unsigned int *  m,
            double *              w,
            std::complex<double> *Z,
            const unsigned int *  ldz,
            unsigned int *        isuppz,
            std::complex<double> *work,
            const int *           lwork,
            double *              rwork,
            const int *           lrwork,
            int *                 iwork,
            const int *           liwork,
            int *                 info);
    void
    zherk_(const char *                uplo,
           const char *                trans,
           const unsigned int *        n,
           const unsigned int *        k,
           const double *              alpha,
           const std::complex<double> *A,
           const unsigned int *        lda,
           const double *              beta,
           std::complex<double> *      C,
           const unsigned int *        ldc);
    void
    zcopy_(const unsigned int *        n,
           const std::complex<double> *x,
           const unsigned int *        incx,
           std::complex<double> *      y,
           const unsigned int *        incy);
    void
    zdotc_(std::complex<double> *      C,
           const int *                 N,
           const std::complex<double> *X,
           const int *                 INCX,
           const std::complex<double> *Y,
           const int *                 INCY);
    void
    zaxpy_(const unsigned int *        n,
           const std::complex<double> *alpha,
           std::complex<double> *      x,
           const unsigned int *        incx,
           std::complex<double> *      y,
           const unsigned int *        incy);
    void
    dpotrf_(const char *        uplo,
            const unsigned int *n,
            double *            a,
            const unsigned int *lda,
            int *               info);
    void
    zpotrf_(const char *          uplo,
            const unsigned int *  n,
            std::complex<double> *a,
            const unsigned int *  lda,
            int *                 info);
    void
    dtrtri_(const char *        uplo,
            const char *        diag,
            const unsigned int *n,
            double *            a,
            const unsigned int *lda,
            int *               info);
    void
    ztrtri_(const char *          uplo,
            const char *          diag,
            const unsigned int *  n,
            std::complex<double> *a,
            const unsigned int *  lda,
            int *                 info);
  }
#endif

  inline void
  xgemm(const char *        transA,
        const char *        transB,
        const unsigned int *m,
        const unsigned int *n,
        const unsigned int *k,
        const double *      alpha,
        const double *      A,
        const unsigned int *lda,
        const double *      B,
        const unsigned int *ldb,
        const double *      beta,
        double *            C,
        const unsigned int *ldc)
  {
    dgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void
  xgemm(const char *        transA,
        const char *        transB,
        const unsigned int *m,
        const unsigned int *n,
        const unsigned int *k,
        const float *       alpha,
        const float *       A,
        const unsigned int *lda,
        const float *       B,
        const unsigned int *ldb,
        const float *       beta,
        float *             C,
        const unsigned int *ldc)
  {
    sgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void
  xgemm(const char *                transA,
        const char *                transB,
        const unsigned int *        m,
        const unsigned int *        n,
        const unsigned int *        k,
        const std::complex<double> *alpha,
        const std::complex<double> *A,
        const unsigned int *        lda,
        const std::complex<double> *B,
        const unsigned int *        ldb,
        const std::complex<double> *beta,
        std::complex<double> *      C,
        const unsigned int *        ldc)
  {
    zgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void
  xgemm(const char *               transA,
        const char *               transB,
        const unsigned int *       m,
        const unsigned int *       n,
        const unsigned int *       k,
        const std::complex<float> *alpha,
        const std::complex<float> *A,
        const unsigned int *       lda,
        const std::complex<float> *B,
        const unsigned int *       ldb,
        const std::complex<float> *beta,
        std::complex<float> *      C,
        const unsigned int *       ldc)
  {
    cgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }


  inline void
  xscal(const unsigned int *n,
        const double *      alpha,
        double *            x,
        const unsigned int *inc)
  {
    dscal_(n, alpha, x, inc);
  }

  inline void
  xscal(const unsigned int *        n,
        const std::complex<double> *alpha,
        std::complex<double> *      x,
        const unsigned int *        inc)
  {
    zscal_(n, alpha, x, inc);
  }

  inline void
  xcopy(const unsigned int *n,
        const double *      x,
        const unsigned int *incx,
        double *            y,
        const unsigned int *incy)
  {
    dcopy_(n, x, incx, y, incy);
  }

  inline void
  xcopy(const unsigned int *        n,
        const std::complex<double> *x,
        const unsigned int *        incx,
        std::complex<double> *      y,
        const unsigned int *        incy)
  {
    zcopy_(n, x, incx, y, incy);
  }

  /**
   *  @brief Contains linear algebra functions used in the implementation of an eigen solver
   *
   *  @author Phani Motamarri, Sambit Das
   */
  namespace linearAlgebraOperations
  {
    /** @brief Calculates an estimate of lower and upper bounds of a matrix using
     *  k-step Lanczos method.
     *
     *  @param  operatorMatrix An object which has access to the given matrix
     *  @param  vect A dummy vector
     *  @return std::pair<double,double> An estimate of the lower and upper bound of the given matrix
     */
    template <typename T>
    std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(operatorDFTClass &operatorMatrix,
                                        const distributedCPUVec<T> &vect);


    /** @brief Apply Chebyshev filter to a given subspace
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as a dealii array representing multiple
     * fields as a flattened array. In-place update of the given subspace.
     *  @param[in]  numberComponents Number of multiple-fields
     *  @param[in]  m Chebyshev polynomial degree
     *  @param[in]  a lower bound of unwanted spectrum
     *  @param[in]  b upper bound of unwanted spectrum
     *  @param[in]  a0 lower bound of wanted spectrum
     */
    template <typename T>
    void
    chebyshevFilter(operatorDFTClass &    operatorMatrix,
                    distributedCPUVec<T> &X,
                    const unsigned int    numberComponents,
                    const unsigned int    m,
                    const double          a,
                    const double          b,
                    const double          a0);


    template <typename T>
    void
    chebyshevFilterOpt(operatorDFTClass &              operatorMatrix,
                       distributedCPUVec<T> &          X,
                       std::vector<dataTypes::number> &cellWaveFunctionMatrix,
                       const unsigned int              numberComponents,
                       const unsigned int              m,
                       const double                    a,
                       const double                    b,
                       const double                    a0);



    /** @brief Orthogonalize given subspace using GramSchmidt orthogonalization
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] mpiComm global communicator
     */
    template <typename T>
    void
    gramSchmidtOrthogonalization(std::vector<T> &   X,
                                 const unsigned int numberComponents,
                                 const MPI_Comm &   mpiComm);


    /** @brief Orthogonalize given subspace using Lowden orthogonalization for double data-type
     *  (serial version using LAPACK)
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] mpiComm global communicator
     *  @return flag indicating success/failure. 1 for failure, 0 for success
     */
    unsigned int
    lowdenOrthogonalization(std::vector<dataTypes::number> &X,
                            const unsigned int              numberComponents,
                            const MPI_Comm &                mpiComm);


    /** @brief Orthogonalize given subspace using Pseudo-Gram-Schmidt orthogonalization
     * (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm global communicator
     *
     *  @return flag indicating success/failure. 1 for failure, 0 for success
     */
    template <typename T>
    unsigned int
    pseudoGramSchmidtOrthogonalization(elpaScalaManager & elpaScala,
                                       std::vector<T> &   X,
                                       const unsigned int numberComponents,
                                       const MPI_Comm &   interBandGroupComm,
                                       const MPI_Comm &   mpiComm,
                                       const bool         useMixedPrec);


    /** @brief Compute Rayleigh-Ritz projection
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place rotated subspace
     *  @param[in] numberComponents Number of vectors
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzGEP(operatorDFTClass &   operatorMatrix,
                    elpaScalaManager &   elpaScala,
                    std::vector<T> &     X,
                    const unsigned int   numberComponents,
                    const MPI_Comm &     interBandGroupComm,
                    const MPI_Comm &     mpiComm,
                    std::vector<double> &eigenValues,
                    const bool           useMixedPrec);


    /** @brief Compute Rayleigh-Ritz projection
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place rotated subspace
     *  @param[in] numberComponents Number of vectors
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitz(operatorDFTClass &   operatorMatrix,
                 elpaScalaManager &   elpaScala,
                 std::vector<T> &     X,
                 const unsigned int   numberComponents,
                 const MPI_Comm &     interBandGroupComm,
                 const MPI_Comm &     mpiComm,
                 std::vector<double> &eigenValues,
                 const bool           doCommAfterBandParal = true);

    /** @brief Compute Rayleigh-Ritz projection in case of spectrum split using direct diagonalization
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as flattened array of multi-vectors.
     *  @param[out] Y rotated subspace of top states
     *  @param[in] numberComponents Number of vectors
     *  @param[in] numberCoreStates Number of core states to be used for
     * spectrum splitting
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzGEPSpectrumSplitDirect(operatorDFTClass &   operatorMatrix,
                                       elpaScalaManager &   elpaScala,
                                       std::vector<T> &     X,
                                       std::vector<T> &     Y,
                                       const unsigned int   numberComponents,
                                       const unsigned int   numberCoreStates,
                                       const MPI_Comm &     interBandGroupComm,
                                       const MPI_Comm &     mpiComm,
                                       const bool           useMixedPrec,
                                       std::vector<double> &eigenValues);


    /** @brief Compute Rayleigh-Ritz projection in case of spectrum split using direct diagonalization
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as flattened array of multi-vectors.
     *  @param[out] Y rotated subspace of top states
     *  @param[in] numberComponents Number of vectors
     *  @param[in] numberCoreStates Number of core states to be used for
     * spectrum splitting
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzSpectrumSplitDirect(operatorDFTClass &    operatorMatrix,
                                    elpaScalaManager &    elpaScala,
                                    const std::vector<T> &X,
                                    std::vector<T> &      Y,
                                    const unsigned int    numberComponents,
                                    const unsigned int    numberCoreStates,
                                    const MPI_Comm &      interBandGroupComm,
                                    const MPI_Comm &      mpiComm,
                                    const bool            useMixedPrec,
                                    std::vector<double> & eigenValues);


#ifdef DFTFE_WITH_ELPA
    void
    elpaDiagonalization(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               numberWavefunctions,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<double> &                 projHamPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid);


    void
    elpaDiagonalizationGEP(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               numberWavefunctions,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<double> &                 projHamPar,
      dftfe::ScaLAPACKMatrix<double> &                 overlapMatPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid);


    void
    elpaPartialDiagonalization(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               N,
      const unsigned int                               Noc,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<double> &                 projHamPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid);

    void
    elpaPartialDiagonalizationGEP(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               N,
      const unsigned int                               Noc,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<double> &                 projHamPar,
      dftfe::ScaLAPACKMatrix<double> &                 overlapMatPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid);
#endif

    /** @brief Compute Compute residual norm associated with eigenValue problem of the given operator
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as STL vector of dealii vectors
     *  @param[in]  eigenValues eigenValues of the operator
     *  @param[in]  mpiComm domain decomposition communicator
     *  @param[out] residualNorms of the eigen Value problem
     */
    template <typename T>
    void
    computeEigenResidualNorm(operatorDFTClass &         operatorMatrix,
                             std::vector<T> &           X,
                             const std::vector<double> &eigenValues,
                             const MPI_Comm &           mpiComm,
                             const MPI_Comm &           interBandGroupComm,
                             std::vector<double> &      residualNorm);

  } // namespace linearAlgebraOperations

} // namespace dftfe
#endif
