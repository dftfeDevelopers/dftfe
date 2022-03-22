// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
//
// @author Phani Motamarri, Sambit Das
//


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <dftParameters.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include "elpaScalaManager.h"
#include "pseudoGS.cc"

namespace dftfe
{
  namespace linearAlgebraOperations
  {
    void
    callevd(const unsigned int dimensionMatrix,
            double *           matrix,
            double *           eigenValues)
    {
      int                info;
      const unsigned int lwork = 1 + 6 * dimensionMatrix +
                                 2 * dimensionMatrix * dimensionMatrix,
                         liwork = 3 + 5 * dimensionMatrix;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'V', uplo = 'U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
              &uplo,
              &dimensionMatrix,
              matrix,
              &dimensionMatrix,
              eigenValues,
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);
    }


    void
    callevd(const unsigned int    dimensionMatrix,
            std::complex<double> *matrix,
            double *              eigenValues)
    {
      int                info;
      const unsigned int lwork = 1 + 6 * dimensionMatrix +
                                 2 * dimensionMatrix * dimensionMatrix,
                         liwork = 3 + 5 * dimensionMatrix;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V', uplo = 'U';
      const unsigned int lrwork =
        1 + 5 * dimensionMatrix + 2 * dimensionMatrix * dimensionMatrix;
      std::vector<double>               rwork(lrwork);
      std::vector<std::complex<double>> work(lwork);


      zheevd_(&jobz,
              &uplo,
              &dimensionMatrix,
              matrix,
              &dimensionMatrix,
              eigenValues,
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<std::complex<double>>().swap(work);
      std::vector<int>().swap(iwork);
    }


    void
    callevr(const unsigned int    dimensionMatrix,
            std::complex<double> *matrixInput,
            std::complex<double> *eigenVectorMatrixOutput,
            double *              eigenValues)
    {
      char                              jobz = 'V', uplo = 'U', range = 'A';
      const double                      vl = 0.0, vu = 0.0;
      const unsigned int                il = 0, iu = 0;
      const double                      abstol = 1e-08;
      std::vector<unsigned int>         isuppz(2 * dimensionMatrix);
      const int                         lwork = 2 * dimensionMatrix;
      std::vector<std::complex<double>> work(lwork);
      const int                         liwork = 10 * dimensionMatrix;
      std::vector<int>                  iwork(liwork);
      const int                         lrwork = 24 * dimensionMatrix;
      std::vector<double>               rwork(lrwork);
      int                               info;

      zheevr_(&jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              matrixInput,
              &dimensionMatrix,
              &vl,
              &vu,
              &il,
              &iu,
              &abstol,
              &dimensionMatrix,
              eigenValues,
              eigenVectorMatrixOutput,
              &dimensionMatrix,
              &isuppz[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      AssertThrow(info == 0, dealii::ExcMessage("Error in zheevr"));
    }



    void
    callevr(const unsigned int dimensionMatrix,
            double *           matrixInput,
            double *           eigenVectorMatrixOutput,
            double *           eigenValues)
    {
      char                      jobz = 'V', uplo = 'U', range = 'A';
      const double              vl = 0.0, vu = 0.0;
      const unsigned int        il = 0, iu = 0;
      const double              abstol = 0.0;
      std::vector<unsigned int> isuppz(2 * dimensionMatrix);
      const int                 lwork = 26 * dimensionMatrix;
      std::vector<double>       work(lwork);
      const int                 liwork = 10 * dimensionMatrix;
      std::vector<int>          iwork(liwork);
      int                       info;

      dsyevr_(&jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              matrixInput,
              &dimensionMatrix,
              &vl,
              &vu,
              &il,
              &iu,
              &abstol,
              &dimensionMatrix,
              eigenValues,
              eigenVectorMatrixOutput,
              &dimensionMatrix,
              &isuppz[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      AssertThrow(info == 0, dealii::ExcMessage("Error in dsyevr"));
    }



    void
    callgemm(const unsigned int         numberEigenValues,
             const unsigned int         localVectorSize,
             const std::vector<double> &eigenVectorSubspaceMatrix,
             const std::vector<double> &X,
             std::vector<double> &      Y)

    {
      const char   transA = 'T', transB = 'N';
      const double alpha = 1.0, beta = 0.0;
      dgemm_(&transA,
             &transB,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &eigenVectorSubspaceMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &Y[0],
             &numberEigenValues);
    }


    void
    callgemm(const unsigned int                       numberEigenValues,
             const unsigned int                       localVectorSize,
             const std::vector<std::complex<double>> &eigenVectorSubspaceMatrix,
             const std::vector<std::complex<double>> &X,
             std::vector<std::complex<double>> &      Y)

    {
      const char                 transA = 'T', transB = 'N';
      const std::complex<double> alpha = 1.0, beta = 0.0;
      zgemm_(&transA,
             &transB,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &eigenVectorSubspaceMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &Y[0],
             &numberEigenValues);
    }


    //
    // chebyshev filtering of given subspace XArray
    //
    template <typename T>
    void
    chebyshevFilter(operatorDFTClass &    operatorMatrix,
                    distributedCPUVec<T> &XArray,
                    const unsigned int    numberWaveFunctions,
                    const unsigned int    m,
                    const double          a,
                    const double          b,
                    const double          a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;

      distributedCPUVec<T> YArray; //,YNewArray;

      //
      // create YArray
      //
      YArray.reinit(XArray);


      //
      // initialize to zeros.
      // x
      const T zeroValue = 0.0;
      YArray            = zeroValue;


      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;

      operatorMatrix.HX(XArray, numberWaveFunctions, scaleFlag, scalar, YArray);


      double alpha1 = sigma1 / e, alpha2 = -c;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      YArray.add(alpha2, XArray);
      YArray *= alpha1;

      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          //
          // multiply XArray with alpha2
          //
          XArray *= alpha2;
          XArray.add(-c * alpha1, YArray);


          //
          // call HX
          //
          bool scaleFlag = true;

          operatorMatrix.HX(
            YArray, numberWaveFunctions, scaleFlag, alpha1, XArray);

          //
          // XArray = YArray
          //
          XArray.swap(YArray);

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }

      // copy back YArray to XArray
      XArray = YArray;
    }


    //
    // chebyshev filtering of given subspace XArray
    //
    template <typename T>
    void
    chebyshevFilterOpt(operatorDFTClass &              operatorMatrix,
                       distributedCPUVec<T> &          XArray,
                       std::vector<dataTypes::number> &cellXWaveFunctionMatrix,
                       const unsigned int              numberWaveFunctions,
                       const unsigned int              m,
                       const double                    a,
                       const double                    b,
                       const double                    a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;

      distributedCPUVec<T> YArray;
      std::vector<T>       cellYWaveFunctionMatrix;

      // init cellYWaveFunctionMatrix to a given scalar
      double scalarValue = 0.0;
      operatorMatrix.initWithScalar(numberWaveFunctions,
                                    scalarValue,
                                    cellYWaveFunctionMatrix);


      std::vector<unsigned int> globalArrayClassificationMap;
      operatorMatrix.getInteriorSurfaceNodesMapFromGlobalArray(
        globalArrayClassificationMap);



      //
      // create YArray
      //
      YArray.reinit(XArray);


      //
      // initialize to zeros.
      // x
      const T zeroValue = 0.0;
      YArray            = zeroValue;


      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;


      operatorMatrix.HX(XArray,
                        cellXWaveFunctionMatrix,
                        numberWaveFunctions,
                        scaleFlag,
                        scalar,
                        scalar,
                        scalar,
                        YArray,
                        cellYWaveFunctionMatrix);


      double alpha1 = sigma1 / e, alpha2 = -c;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //

      //
      // Do surface nodes recursive iteration for dealii vectors
      //
      const unsigned int numberDofs = YArray.local_size() / numberWaveFunctions;
      for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
        {
          if (globalArrayClassificationMap[iDof] == 1)
            {
              for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
                {
                  YArray.local_element(iDof * numberWaveFunctions + iWave) =
                    alpha1 *
                      YArray.local_element(iDof * numberWaveFunctions + iWave) +
                    alpha1 * alpha2 *
                      XArray.local_element(iDof * numberWaveFunctions + iWave);
                }
            }
        }


      //
      // Do recursive iteration only for interior cell nodes using cell-level
      // loop
      // Y = a*X + Y
      operatorMatrix.axpby(alpha1 * alpha2,
                           alpha1,
                           numberWaveFunctions,
                           cellXWaveFunctionMatrix,
                           cellYWaveFunctionMatrix);


      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          //
          // multiply XArray with alpha2
          // and XArray = XArray - c*alpha1*YArray


          //
          // Do surface nodes recursive iteration for dealii vectors
          //
          for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
            {
              if (globalArrayClassificationMap[iDof] == 1)
                {
                  for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                       ++iWave)
                    {
                      // XArray.local_element(iDof*numberWaveFunctions+iWave) *=
                      // alpha2;
                      XArray.local_element(iDof * numberWaveFunctions + iWave) =
                        alpha2 * XArray.local_element(
                                   iDof * numberWaveFunctions + iWave) -
                        c * alpha1 *
                          YArray.local_element(iDof * numberWaveFunctions +
                                               iWave);
                    }
                }
            }

          //
          // call HX
          //
          bool   scaleFlag = true;
          double scalarA   = -c * alpha1;
          double scalarB   = alpha2;
          operatorMatrix.HX(YArray,
                            cellYWaveFunctionMatrix,
                            numberWaveFunctions,
                            scaleFlag,
                            alpha1,
                            scalarA,
                            scalarB,
                            XArray,
                            cellXWaveFunctionMatrix);

          //
          // XArray = YArray (may have to optimize this, so that swap happens
          // only for surface nodes for deallii vectors and interior nodes for
          // cellwavefunction matrices
          //
          XArray.swap(YArray);
          cellXWaveFunctionMatrix.swap(cellYWaveFunctionMatrix);

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }

      // copy back YArray to XArray
      XArray                  = YArray;
      cellXWaveFunctionMatrix = cellYWaveFunctionMatrix;
    }


    template <typename T>
    void
    gramSchmidtOrthogonalization(std::vector<T> &   X,
                                 const unsigned int numberVectors,
                                 const MPI_Comm &   mpiComm)
    {
#ifdef USE_PETSC
      const unsigned int localVectorSize = X.size() / numberVectors;

      //
      // Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(mpiComm, localVectorSize, PETSC_DETERMINE, &templateVec);
      VecSetFromOptions(templateVec);


      //
      // Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(mpiComm, &columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors, templateVec, numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      // create list of indices
      //
      std::vector<PetscInt>    indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize, 0.0);

      PetscInt low, high;

      VecGetOwnershipRange(templateVec, &low, &high);


      for (PetscInt index = 0; index < localVectorSize; ++index)
        indices[index] = low + index;

      VecDestroy(&templateVec);

      //
      // Fill in data into BV object
      //
      Vec v;
      for (unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
        {
          BVGetColumn(columnSpaceOfVectors, iColumn, &v);
          VecSet(v, 0.0);
          for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
            data[iNode] = X[numberVectors * iNode + iColumn];

          VecSetValues(
            v, localVectorSize, &indices[0], &data[0], INSERT_VALUES);

          VecAssemblyBegin(v);
          VecAssemblyEnd(v);

          BVRestoreColumn(columnSpaceOfVectors, iColumn, &v);
        }

      //
      // orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors, NULL);

      //
      // Copy data back into X
      //
      Vec          v1;
      PetscScalar *pointerv1;
      for (unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
        {
          BVGetColumn(columnSpaceOfVectors, iColumn, &v1);

          VecGetArray(v1, &pointerv1);

          for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
            X[numberVectors * iNode + iColumn] = pointerv1[iNode];

          VecRestoreArray(v1, &pointerv1);

          BVRestoreColumn(columnSpaceOfVectors, iColumn, &v1);
        }

      BVDestroy(&columnSpaceOfVectors);
#else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: Please link to dealii installed with petsc and slepc to Gram-Schidt orthogonalization."));
#endif
    }


    template <typename T>
    void
    rayleighRitzGEP(operatorDFTClass &   operatorMatrix,
                    elpaScalaManager &   elpaScala,
                    std::vector<T> &     X,
                    const unsigned int   numberWaveFunctions,
                    const MPI_Comm &     interBandGroupComm,
                    const MPI_Comm &     mpi_communicator,
                    std::vector<double> &eigenValues,
                    const bool           useMixedPrec)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(mpi_communicator,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      if (dftParameters::useMixedPrecCGS_O && useMixedPrec)
        computing_timer.enter_subsection(
          "SConj=X^{T}XConj Mixed Prec, RR GEP step");
      else
        computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<T> overlapMatPar(numberWaveFunctions,
                                              processGrid,
                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  T(0.0));

      // SConj=X^{T}*XConj.
      if (!(dftParameters::useMixedPrecCGS_O && useMixedPrec))
        {
          internal::fillParallelOverlapMatrix(&X[0],
                                              X.size(),
                                              numberWaveFunctions,
                                              processGrid,
                                              interBandGroupComm,
                                              mpi_communicator,
                                              overlapMatPar);
        }
      else
        {
          if (std::is_same<T, std::complex<double>>::value)
            internal::fillParallelOverlapMatrixMixedPrec<T,
                                                         std::complex<float>>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpi_communicator,
              overlapMatPar);
          else
            internal::fillParallelOverlapMatrixMixedPrec<T, float>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpi_communicator,
              overlapMatPar);
        }

      if (dftParameters::useMixedPrecCGS_O && useMixedPrec)
        computing_timer.leave_subsection(
          "SConj=X^{T}XConj Mixed Prec, RR GEP step");
      else
        computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");

      // SConj=LConj*L^{T}
      computing_timer.enter_subsection("Cholesky and triangular matrix invert");


      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParameters::useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<T> overlapMatParConjTrans(numberWaveFunctions,
                                                           processGrid,
                                                           rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      T(0.0));

          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;
              elpaCholesky(elpaScala.getElpaHandle(),
                           &overlapMatParConjTrans.local_el(0, 0),
                           &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));
            }
          overlapMatPar.copy_conjugate_transposed(overlapMatParConjTrans);
          overlapMatPropertyPostCholesky =
            dftfe::LAPACKSupport::Property::lower_triangular;
        }
      else
        {
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();
        }

      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<T> LMatPar(
        numberWaveFunctions,
        processGrid,
        rowsBlockSize,
        dftfe::LAPACKSupport::Property::lower_triangular);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_j < glob_i)
                  LMatPar.local_el(j, i) = T(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // compute LConj^{-1}
      LMatPar.invert();

      computing_timer.leave_subsection("Cholesky and triangular matrix invert");


      computing_timer.enter_subsection("Compute ProjHam, RR step");
      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));


      operatorMatrix.XtHX(X, numberWaveFunctions, processGrid, projHamPar);
      computing_timer.leave_subsection("Compute ProjHam, RR step");

      computing_timer.enter_subsection(
        "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                    processGrid,
                                                    rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  T(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      projHamPar.add(projHamParConjTrans, T(1.0), T(1.0));

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= T(0.5);
              }
          }

      dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                               processGrid,
                                               rowsBlockSize);

      // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // conjugate transpose LAPACK notation)
      LMatPar.mmult(projHamParCopy, projHamPar);
      projHamParCopy.zmCmult(projHamPar, LMatPar);

      computing_timer.leave_subsection(
        "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");
      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);
      if (dftParameters::useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandle(),
                               &projHamPar.local_el(0, 0),
                               &eigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(&eigenValues[0],
                    eigenValues.size(),
                    MPI_DOUBLE,
                    0,
                    mpi_communicator);


          eigenVectors.copy_to(projHamPar);

          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, numberWaveFunctions - 1), true);
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      //
      // rotate the basis in the subspace
      // X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, stored in the column major
      // format In the above we use Q^{T}={QConjPrime}^{C}*LConj^{-1}
      if (!(dftParameters::useMixedPrecSubspaceRotRR && useMixedPrec))
        computing_timer.enter_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
      else
        computing_timer.enter_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR step");

      projHamParCopy.copy_conjugate_transposed(projHamPar);
      projHamParCopy.mmult(projHamPar, LMatPar);

      if (!(dftParameters::useMixedPrecSubspaceRotRR && useMixedPrec))
        internal::subspaceRotation(&X[0],
                                   X.size(),
                                   numberWaveFunctions,
                                   processGrid,
                                   interBandGroupComm,
                                   mpi_communicator,
                                   projHamPar,
                                   false,
                                   false,
                                   false);
      else
        {
          if (std::is_same<T, std::complex<double>>::value)
            internal::subspaceRotationMixedPrec<T, std::complex<float>>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpi_communicator,
              projHamPar,
              false,
              false);
          else
            internal::subspaceRotationMixedPrec<T, float>(&X[0],
                                                          X.size(),
                                                          numberWaveFunctions,
                                                          processGrid,
                                                          interBandGroupComm,
                                                          mpi_communicator,
                                                          projHamPar,
                                                          false,
                                                          false);
        }

      if (!(dftParameters::useMixedPrecSubspaceRotRR && useMixedPrec))
        computing_timer.leave_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
      else
        computing_timer.leave_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR step");
    }

    template <typename T>
    void
    rayleighRitz(operatorDFTClass &   operatorMatrix,
                 elpaScalaManager &   elpaScala,
                 std::vector<T> &     X,
                 const unsigned int   numberWaveFunctions,
                 const MPI_Comm &     interBandGroupComm,
                 const MPI_Comm &     mpi_communicator,
                 std::vector<double> &eigenValues,
                 const bool           doCommAfterBandParal)

    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(mpi_communicator,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);
      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));

      computing_timer.enter_subsection("Blocked XtHX, RR step");
      operatorMatrix.XtHX(X, numberWaveFunctions, processGrid, projHamPar);
      computing_timer.leave_subsection("Blocked XtHX, RR step");

      //
      // compute eigendecomposition of ProjHam HConjProj= QConj*D*QConj^{C} (C
      // denotes conjugate transpose LAPACK notation)
      //
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);
      if (dftParameters::useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          // For ELPA eigendecomposition the full matrix is required unlike
          // ScaLAPACK which can work with only the lower triangular part
          dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      T(0.0));


          projHamParConjTrans.copy_conjugate_transposed(projHamPar);
          projHamPar.add(projHamParConjTrans, T(1.0), T(1.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
              {
                const unsigned int glob_i = projHamPar.global_column(i);
                for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
                  {
                    const unsigned int glob_j = projHamPar.global_row(j);
                    if (glob_i == glob_j)
                      projHamPar.local_el(j, i) *= T(0.5);
                  }
              }

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandle(),
                               &projHamPar.local_el(0, 0),
                               &eigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(&eigenValues[0],
                    eigenValues.size(),
                    MPI_DOUBLE,
                    0,
                    mpi_communicator);


          eigenVectors.copy_to(projHamPar);

          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, numberWaveFunctions - 1), true);
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }


      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      //
      // rotate the basis in the subspace X = X*Q, implemented as
      // X^{T}=Qc^{C}*X^{T} with X^{T} stored in the column major format
      //
      computing_timer.enter_subsection("Blocked subspace rotation, RR step");
      dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                               processGrid,
                                               rowsBlockSize);
      projHamParCopy.copy_conjugate_transposed(projHamPar);
      internal::subspaceRotation(&X[0],
                                 X.size(),
                                 numberWaveFunctions,
                                 processGrid,
                                 interBandGroupComm,
                                 mpi_communicator,
                                 projHamParCopy,
                                 false,
                                 false,
                                 doCommAfterBandParal);

      computing_timer.leave_subsection("Blocked subspace rotation, RR step");
    }

    template <typename T>
    void
    rayleighRitzGEPSpectrumSplitDirect(operatorDFTClass &   operatorMatrix,
                                       elpaScalaManager &   elpaScala,
                                       std::vector<T> &     X,
                                       std::vector<T> &     Y,
                                       const unsigned int   numberWaveFunctions,
                                       const unsigned int   numberCoreStates,
                                       const MPI_Comm &     interBandGroupComm,
                                       const MPI_Comm &     mpiComm,
                                       const bool           useMixedPrec,
                                       std::vector<double> &eigenValues)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(mpiComm,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      if (dftParameters::useMixedPrecCGS_O && useMixedPrec)
        computing_timer.enter_subsection(
          "SConj=X^{T}XConj Mixed Prec, RR GEP step");
      else
        computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<T> overlapMatPar(numberWaveFunctions,
                                              processGrid,
                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  T(0.0));

      // SConj=X^{T}*XConj
      if (!(dftParameters::useMixedPrecCGS_O && useMixedPrec))
        {
          internal::fillParallelOverlapMatrix(&X[0],
                                              X.size(),
                                              numberWaveFunctions,
                                              processGrid,
                                              interBandGroupComm,
                                              mpiComm,
                                              overlapMatPar);
        }
      else
        {
          if (std::is_same<T, std::complex<double>>::value)
            internal::fillParallelOverlapMatrixMixedPrec<T,
                                                         std::complex<float>>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpiComm,
              overlapMatPar);
          else
            internal::fillParallelOverlapMatrixMixedPrec<T, float>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpiComm,
              overlapMatPar);
        }


      if (dftParameters::useMixedPrecCGS_O && useMixedPrec)
        computing_timer.leave_subsection(
          "SConj=X^{T}XConj Mixed Prec, RR GEP step");
      else
        computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");
      // Sc=Lc*L^{T}
      computing_timer.enter_subsection("Cholesky and triangular matrix invert");

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParameters::useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<T> overlapMatParConjTrans(numberWaveFunctions,
                                                           processGrid,
                                                           rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      T(0.0));

          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;
              elpaCholesky(elpaScala.getElpaHandle(),
                           &overlapMatParConjTrans.local_el(0, 0),
                           &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));
            }
          overlapMatPar.copy_conjugate_transposed(overlapMatParConjTrans);
          overlapMatPropertyPostCholesky =
            dftfe::LAPACKSupport::Property::lower_triangular;
        }
      else
        {
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();
        }

      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<T> LMatPar(
        numberWaveFunctions,
        processGrid,
        rowsBlockSize,
        dftfe::LAPACKSupport::Property::lower_triangular);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_j < glob_i)
                  LMatPar.local_el(j, i) = T(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // compute LConj^{-1}
      LMatPar.invert();
      computing_timer.leave_subsection("Cholesky and triangular matrix invert");



      if (dftParameters::useMixedPrecXTHXSpectrumSplit && useMixedPrec)
        computing_timer.enter_subsection(
          "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
      else
        computing_timer.enter_subsection(
          "HConjProj=X^{T}*HConj*XConj, RR GEP step");
      //
      // compute projected Hamiltonian HConjProj=X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));

      if (useMixedPrec && dftParameters::useMixedPrecXTHXSpectrumSplit)
        {
          operatorMatrix.XtHXMixedPrec(
            X, numberWaveFunctions, numberCoreStates, processGrid, projHamPar);
        }
      else
        {
          operatorMatrix.XtHX(X, numberWaveFunctions, processGrid, projHamPar);
        }


      if (dftParameters::useMixedPrecXTHXSpectrumSplit && useMixedPrec)
        computing_timer.leave_subsection(
          "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
      else
        computing_timer.leave_subsection(
          "HConjProj=X^{T}*HConj*XConj, RR GEP step");

      computing_timer.enter_subsection(
        "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                    processGrid,
                                                    rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  T(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      if (dftParameters::useELPA)
        projHamPar.add(projHamParConjTrans, T(-1.0), T(-1.0));
      else
        projHamPar.add(projHamParConjTrans, T(1.0), T(1.0));


      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= T(0.5);
              }
          }

      dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                               processGrid,
                                               rowsBlockSize);

      // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // conjugate transpose LAPACK notation)
      LMatPar.mmult(projHamParCopy, projHamPar);
      projHamParCopy.zmCmult(projHamPar, LMatPar);

      computing_timer.leave_subsection(
        "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");
      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      //
      const unsigned int numValenceStates =
        numberWaveFunctions - numberCoreStates;
      eigenValues.resize(numValenceStates);
      if (dftParameters::useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          std::vector<double>       allEigenValues(numberWaveFunctions, 0.0);
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandlePartialEigenVec(),
                               &projHamPar.local_el(0, 0),
                               &allEigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(
                error == ELPA_OK,
                dealii::ExcMessage(
                  "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
            }

          for (unsigned int i = 0; i < numValenceStates; ++i)
            eigenValues[numValenceStates - i - 1] = -allEigenValues[i];

          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiComm);


          dftfe::ScaLAPACKMatrix<T> permutedIdentityMat(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);
          if (processGrid->is_process_active())
            std::fill(&permutedIdentityMat.local_el(0, 0),
                      &permutedIdentityMat.local_el(0, 0) +
                        permutedIdentityMat.local_m() *
                          permutedIdentityMat.local_n(),
                      T(0.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
              {
                const unsigned int glob_i = permutedIdentityMat.global_row(i);
                if (glob_i < numValenceStates)
                  {
                    for (unsigned int j = 0; j < permutedIdentityMat.local_n();
                         ++j)
                      {
                        const unsigned int glob_j =
                          permutedIdentityMat.global_column(j);
                        if (glob_j < numValenceStates)
                          {
                            const unsigned int rowIndexToSetOne =
                              (numValenceStates - 1) - glob_j;
                            if (glob_i == rowIndexToSetOne)
                              permutedIdentityMat.local_el(i, j) = T(1.0);
                          }
                      }
                  }
              }

          eigenVectors.mmult(projHamPar, permutedIdentityMat);



          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(numberCoreStates, numberWaveFunctions - 1), true);
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }


      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");

      //
      // rotate the basis in the subspace
      // Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}
      //
      projHamParCopy.copy_conjugate_transposed(projHamPar);
      projHamParCopy.mmult(projHamPar, LMatPar);

      computing_timer.enter_subsection(
        "Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");

      internal::subspaceRotationSpectrumSplit(&X[0],
                                              &Y[0],
                                              X.size(),
                                              numberWaveFunctions,
                                              processGrid,
                                              numberWaveFunctions -
                                                numberCoreStates,
                                              interBandGroupComm,
                                              mpiComm,
                                              projHamPar,
                                              false);

      computing_timer.leave_subsection(
        "Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");

      // X^{T}=LConj^{-1}*X^{T}
      if (!(dftParameters::useMixedPrecCGS_SR && useMixedPrec))
        {
          computing_timer.enter_subsection("X^{T}=Lconj^{-1}*X^{T}, RR step");
          internal::subspaceRotation(&X[0],
                                     X.size(),
                                     numberWaveFunctions,
                                     processGrid,
                                     interBandGroupComm,
                                     mpiComm,
                                     LMatPar,
                                     false,
                                     true,
                                     false);
          computing_timer.leave_subsection("X^{T}=Lconj^{-1}*X^{T}, RR step");
        }
      else
        {
          computing_timer.enter_subsection(
            "X^{T}=Lconj^{-1}*X^{T} mixed prec, RR step");
          if (std::is_same<T, std::complex<double>>::value)
            internal::subspaceRotationCGSMixedPrec<T, std::complex<float>>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpiComm,
              LMatPar,
              false,
              false);
          else
            internal::subspaceRotationCGSMixedPrec<T, float>(
              &X[0],
              X.size(),
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              mpiComm,
              LMatPar,
              false,
              false);
          computing_timer.leave_subsection(
            "X^{T}=Lconj^{-1}*X^{T} mixed prec, RR step");
        }
    }


    template <typename T>
    void
    rayleighRitzSpectrumSplitDirect(operatorDFTClass &    operatorMatrix,
                                    elpaScalaManager &    elpaScala,
                                    const std::vector<T> &X,
                                    std::vector<T> &      Y,
                                    const unsigned int    numberWaveFunctions,
                                    const unsigned int    numberCoreStates,
                                    const MPI_Comm &      interBandGroupComm,
                                    const MPI_Comm &      mpi_communicator,
                                    const bool            useMixedPrec,
                                    std::vector<double> & eigenValues)

    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(mpi_communicator,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);
      //
      // compute projected Hamiltonian HConjProj= X^{T}*HConj*XConj
      //
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();


      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));

      if (useMixedPrec && dftParameters::useMixedPrecXTHXSpectrumSplit)
        {
          computing_timer.enter_subsection("Blocked XtHX Mixed Prec, RR step");
          operatorMatrix.XtHXMixedPrec(
            X, numberWaveFunctions, numberCoreStates, processGrid, projHamPar);

          computing_timer.leave_subsection("Blocked XtHX Mixed Prec, RR step");
        }
      else
        {
          computing_timer.enter_subsection("Blocked XtHX, RR step");
          operatorMatrix.XtHX(X, numberWaveFunctions, processGrid, projHamPar);
          computing_timer.leave_subsection("Blocked XtHX, RR step");
        }

      const unsigned int numValenceStates =
        numberWaveFunctions - numberCoreStates;
      eigenValues.resize(numValenceStates);
      // compute eigendecomposition of ProjHam HConjProj= Qc*D*Qc^{C}
      if (dftParameters::useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          std::vector<double>       allEigenValues(numberWaveFunctions, 0.0);
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          // For ELPA eigendecomposition the full HConjProj matrix is required
          // unlike ScaLAPACK which can work with only the lower triangular part
          dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);
          if (processGrid->is_process_active())
            std::fill(&projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      T(0.0));

          projHamParConjTrans.copy_conjugate_transposed(projHamPar);
          projHamPar.add(projHamParConjTrans, T(-1.0), T(-1.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
              {
                const unsigned int glob_i = projHamPar.global_column(i);
                for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
                  {
                    const unsigned int glob_j = projHamPar.global_row(j);
                    if (glob_i == glob_j)
                      projHamPar.local_el(j, i) *= T(0.5);
                  }
              }

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandlePartialEigenVec(),
                               &projHamPar.local_el(0, 0),
                               &allEigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(
                error == ELPA_OK,
                dealii::ExcMessage(
                  "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
            }

          for (unsigned int i = 0; i < numValenceStates; ++i)
            eigenValues[numValenceStates - i - 1] = -allEigenValues[i];

          MPI_Bcast(&eigenValues[0],
                    eigenValues.size(),
                    MPI_DOUBLE,
                    0,
                    mpi_communicator);


          dftfe::ScaLAPACKMatrix<T> permutedIdentityMat(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);
          if (processGrid->is_process_active())
            std::fill(&permutedIdentityMat.local_el(0, 0),
                      &permutedIdentityMat.local_el(0, 0) +
                        permutedIdentityMat.local_m() *
                          permutedIdentityMat.local_n(),
                      T(0.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
              {
                const unsigned int glob_i = permutedIdentityMat.global_row(i);
                if (glob_i < numValenceStates)
                  {
                    for (unsigned int j = 0; j < permutedIdentityMat.local_n();
                         ++j)
                      {
                        const unsigned int glob_j =
                          permutedIdentityMat.global_column(j);
                        if (glob_j < numValenceStates)
                          {
                            const unsigned int rowIndexToSetOne =
                              (numValenceStates - 1) - glob_j;
                            if (glob_i == rowIndexToSetOne)
                              permutedIdentityMat.local_el(i, j) = T(1.0);
                          }
                      }
                  }
              }

          eigenVectors.mmult(projHamPar, permutedIdentityMat);



          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(numberCoreStates, numberWaveFunctions - 1), true);
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }


      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");

      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);
      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      //
      // rotate the basis in the subspace Xfr = X*Qfr, implemented as
      // Xfr^{T}=QfrConj^{C}*X^{T} with X^{T} stored in the column major format
      //
      dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                               processGrid,
                                               rowsBlockSize);
      projHamParCopy.copy_conjugate_transposed(projHamPar);

      computing_timer.enter_subsection("Blocked subspace rotation, RR step");

      internal::subspaceRotationSpectrumSplit(&X[0],
                                              &Y[0],
                                              X.size(),
                                              numberWaveFunctions,
                                              processGrid,
                                              numberWaveFunctions -
                                                numberCoreStates,
                                              interBandGroupComm,
                                              mpi_communicator,
                                              projHamParCopy,
                                              false);

      computing_timer.leave_subsection("Blocked subspace rotation, RR step");
    }

    template <typename NumberType>
    void
    elpaDiagonalization(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               numberWaveFunctions,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType> &             projHamPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();

      dftfe::ScaLAPACKMatrix<NumberType> eigenVectors(numberWaveFunctions,
                                                      processGrid,
                                                      rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  NumberType(0.0));

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<NumberType> projHamParTrans(numberWaveFunctions,
                                                         processGrid,
                                                         rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);


      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, 1.0, 1.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors_d(elpaScala.getElpaHandle(),
                              &projHamPar.local_el(0, 0),
                              &eigenValues[0],
                              &eigenVectors.local_el(0, 0),
                              &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_eigenvectors error."));
        }


      MPI_Bcast(&eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiComm);


      eigenVectors.copy_to(projHamPar);
    }

    template <typename NumberType>
    void
    elpaDiagonalizationGEP(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               numberWaveFunctions,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType> &             projHamPar,
      dftfe::ScaLAPACKMatrix<NumberType> &             overlapMatPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;

      // For ELPA cholesky only the upper triangular part is enough
      dftfe::ScaLAPACKMatrix<double> overlapMatParTrans(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatParTrans.local_el(0, 0),
                  &overlapMatParTrans.local_el(0, 0) +
                    overlapMatParTrans.local_m() * overlapMatParTrans.local_n(),
                  0.0);

      overlapMatParTrans.copy_transposed(overlapMatPar);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_cholesky_d(elpaScala.getElpaHandle(),
                          &overlapMatParTrans.local_el(0, 0),
                          &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_cholesky_d error."));
        }
      overlapMatParTrans.copy_to(overlapMatPar);
      overlapMatPropertyPostCholesky =
        dftfe::LAPACKSupport::Property::upper_triangular;

      AssertThrow(
        overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::lower_triangular ||
          overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::upper_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

      dftfe::ScaLAPACKMatrix<double> LMatPar(numberWaveFunctions,
                                             processGrid,
                                             rowsBlockSize,
                                             overlapMatPropertyPostCholesky);

      // copy triangular part of overlapMatPar into LMatPar
      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = overlapMatPar.global_column(i);
            for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = overlapMatPar.global_row(j);
                if (overlapMatPropertyPostCholesky ==
                    dftfe::LAPACKSupport::Property::lower_triangular)
                  {
                    if (glob_i <= glob_j)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
                else
                  {
                    if (glob_j <= glob_i)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
              }
          }


      // invert triangular matrix
      if (processGrid->is_process_active())
        {
          int error;
          elpa_invert_trm_d(elpaScala.getElpaHandle(),
                            &LMatPar.local_el(0, 0),
                            &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_invert_trm_d error."));
        }

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<double> projHamParTrans(numberWaveFunctions,
                                                     processGrid,
                                                     rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);


      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, 1.0, 1.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      dftfe::ScaLAPACKMatrix<double> projHamParCopy(numberWaveFunctions,
                                                    processGrid,
                                                    rowsBlockSize);

      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        {
          LMatPar.mmult(projHamParCopy, projHamPar);
          projHamParCopy.mTmult(projHamPar, LMatPar);
        }
      else
        {
          LMatPar.Tmmult(projHamParCopy, projHamPar);
          projHamParCopy.mmult(projHamPar, LMatPar);
        }

      //
      // compute eigendecomposition of ProjHam
      //
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);

      dftfe::ScaLAPACKMatrix<double> eigenVectors(numberWaveFunctions,
                                                  processGrid,
                                                  rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  0.0);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors_d(elpaScala.getElpaHandle(),
                              &projHamPar.local_el(0, 0),
                              &eigenValues[0],
                              &eigenVectors.local_el(0, 0),
                              &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_eigenvectors error."));
        }


      MPI_Bcast(&eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiComm);


      eigenVectors.copy_to(projHamPar);

      projHamPar.copy_to(projHamParCopy);
      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        LMatPar.Tmmult(projHamPar, projHamParCopy);
      else
        LMatPar.mmult(projHamPar, projHamParCopy);
    }


    template <typename NumberType>
    void
    elpaPartialDiagonalization(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               N,
      const unsigned int                               Noc,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType> &             projHamPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      //
      // compute projected Hamiltonian
      //
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();

      const unsigned int numValenceStates = N - Noc;
      eigenValues.resize(numValenceStates);
      std::vector<double>            allEigenValues(N, 0.0);
      dftfe::ScaLAPACKMatrix<double> eigenVectors(N,
                                                  processGrid,
                                                  rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  0.0);

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<double> projHamParTrans(N,
                                                     processGrid,
                                                     rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);

      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, -1.0, -1.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors_d(elpaScala.getElpaHandlePartialEigenVec(),
                              &projHamPar.local_el(0, 0),
                              &allEigenValues[0],
                              &eigenVectors.local_el(0, 0),
                              &error);
          AssertThrow(
            error == ELPA_OK,
            dealii::ExcMessage(
              "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
        }

      for (unsigned int i = 0; i < numValenceStates; ++i)
        {
          eigenValues[numValenceStates - i - 1] = -allEigenValues[i];
        }

      MPI_Bcast(&eigenValues[0],
                eigenValues.size(),
                MPI_DOUBLE,
                0,
                elpaScala.getMPICommunicator());


      dftfe::ScaLAPACKMatrix<double> permutedIdentityMat(N,
                                                         processGrid,
                                                         rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&permutedIdentityMat.local_el(0, 0),
                  &permutedIdentityMat.local_el(0, 0) +
                    permutedIdentityMat.local_m() *
                      permutedIdentityMat.local_n(),
                  0.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
          {
            const unsigned int glob_i = permutedIdentityMat.global_row(i);
            if (glob_i < numValenceStates)
              {
                for (unsigned int j = 0; j < permutedIdentityMat.local_n(); ++j)
                  {
                    const unsigned int glob_j =
                      permutedIdentityMat.global_column(j);
                    if (glob_j < numValenceStates)
                      {
                        const unsigned int rowIndexToSetOne =
                          (numValenceStates - 1) - glob_j;
                        if (glob_i == rowIndexToSetOne)
                          permutedIdentityMat.local_el(i, j) = 1.0;
                      }
                  }
              }
          }

      eigenVectors.mmult(projHamPar, permutedIdentityMat);
    }


    template <typename NumberType>
    void
    elpaPartialDiagonalizationGEP(
      elpaScalaManager &                               elpaScala,
      const unsigned int                               N,
      const unsigned int                               Noc,
      const MPI_Comm &                                 mpiComm,
      std::vector<double> &                            eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType> &             projHamPar,
      dftfe::ScaLAPACKMatrix<NumberType> &             overlapMatPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;

      // For ELPA cholesky only the upper triangular part is enough
      dftfe::ScaLAPACKMatrix<double> overlapMatParTrans(N,
                                                        processGrid,
                                                        rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatParTrans.local_el(0, 0),
                  &overlapMatParTrans.local_el(0, 0) +
                    overlapMatParTrans.local_m() * overlapMatParTrans.local_n(),
                  0.0);

      overlapMatParTrans.copy_transposed(overlapMatPar);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_cholesky_d(elpaScala.getElpaHandle(),
                          &overlapMatParTrans.local_el(0, 0),
                          &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_cholesky_d error."));
        }
      overlapMatParTrans.copy_to(overlapMatPar);
      overlapMatPropertyPostCholesky =
        dftfe::LAPACKSupport::Property::upper_triangular;

      AssertThrow(
        overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::lower_triangular ||
          overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::upper_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

      dftfe::ScaLAPACKMatrix<double> LMatPar(N,
                                             processGrid,
                                             rowsBlockSize,
                                             overlapMatPropertyPostCholesky);


      // copy triangular part of overlapMatPar into LMatPar
      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = overlapMatPar.global_column(i);
            for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = overlapMatPar.global_row(j);
                if (overlapMatPropertyPostCholesky ==
                    dftfe::LAPACKSupport::Property::lower_triangular)
                  {
                    if (glob_i <= glob_j)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
                else
                  {
                    if (glob_j <= glob_i)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
              }
          }


      if (processGrid->is_process_active())
        {
          int error;
          elpa_invert_trm_d(elpaScala.getElpaHandle(),
                            &LMatPar.local_el(0, 0),
                            &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_invert_trm_d error."));
        }

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<double> projHamParTrans(N,
                                                     processGrid,
                                                     rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);

      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, -1.0, -1.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      dftfe::ScaLAPACKMatrix<double> projHamParCopy(N,
                                                    processGrid,
                                                    rowsBlockSize);

      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        {
          LMatPar.mmult(projHamParCopy, projHamPar);
          projHamParCopy.mTmult(projHamPar, LMatPar);
        }
      else
        {
          LMatPar.Tmmult(projHamParCopy, projHamPar);
          projHamParCopy.mmult(projHamPar, LMatPar);
        }

      const unsigned int Nfr = N - Noc;
      eigenValues.resize(Nfr);
      std::vector<double>            allEigenValues(N, 0.0);
      dftfe::ScaLAPACKMatrix<double> eigenVectors(N,
                                                  processGrid,
                                                  rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  0.0);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors_d(elpaScala.getElpaHandlePartialEigenVec(),
                              &projHamPar.local_el(0, 0),
                              &allEigenValues[0],
                              &eigenVectors.local_el(0, 0),
                              &error);
          AssertThrow(
            error == ELPA_OK,
            dealii::ExcMessage(
              "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
        }

      for (unsigned int i = 0; i < Nfr; ++i)
        {
          eigenValues[Nfr - i - 1] = -allEigenValues[i];
        }

      MPI_Bcast(&eigenValues[0],
                eigenValues.size(),
                MPI_DOUBLE,
                0,
                elpaScala.getMPICommunicator());


      dftfe::ScaLAPACKMatrix<double> permutedIdentityMat(N,
                                                         processGrid,
                                                         rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&permutedIdentityMat.local_el(0, 0),
                  &permutedIdentityMat.local_el(0, 0) +
                    permutedIdentityMat.local_m() *
                      permutedIdentityMat.local_n(),
                  0.0);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
          {
            const unsigned int glob_i = permutedIdentityMat.global_row(i);
            if (glob_i < Nfr)
              {
                for (unsigned int j = 0; j < permutedIdentityMat.local_n(); ++j)
                  {
                    const unsigned int glob_j =
                      permutedIdentityMat.global_column(j);
                    if (glob_j < Nfr)
                      {
                        const unsigned int rowIndexToSetOne =
                          (Nfr - 1) - glob_j;
                        if (glob_i == rowIndexToSetOne)
                          permutedIdentityMat.local_el(i, j) = 1.0;
                      }
                  }
              }
          }

      eigenVectors.mmult(projHamPar, permutedIdentityMat);

      projHamPar.copy_to(projHamParCopy);
      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        LMatPar.Tmmult(projHamPar, projHamParCopy);
      else
        LMatPar.mmult(projHamPar, projHamParCopy);

      overlapMatPar.copy_transposed(LMatPar);
    }


    template <typename T>
    void
    computeEigenResidualNorm(operatorDFTClass &         operatorMatrix,
                             std::vector<T> &           X,
                             const std::vector<double> &eigenValues,
                             const MPI_Comm &           mpiComm,
                             const MPI_Comm &           interBandGroupComm,
                             std::vector<double> &      residualNorm)

    {
      //
      // get the number of eigenVectors
      //
      const unsigned int  totalNumberVectors = eigenValues.size();
      const unsigned int  localVectorSize    = X.size() / totalNumberVectors;
      std::vector<double> residualNormSquare(totalNumberVectors, 0.0);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, totalNumberVectors, bandGroupLowHighPlusOneIndices);

      // create temporary arrays XBlock,HXBlock
      distributedCPUVec<T> XBlock, HXBlock;

      // Do H*X using a blocked approach and compute
      // the residual norms: H*XBlock-XBlock*D, where
      // D is the eigenvalues matrix.
      // The blocked approach avoids additional full
      // wavefunction matrix memory
      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize,
                 bandGroupLowHighPlusOneIndices[1]);

      for (unsigned int jvec = 0; jvec < totalNumberVectors;
           jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge"
          const unsigned int B =
            std::min(vectorsBlockSize, totalNumberVectors - jvec);

          if (jvec == 0 || B != vectorsBlockSize)
            {
              operatorMatrix.reinit(B, XBlock, true);
              HXBlock.reinit(XBlock);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              XBlock = T(0.);
              // fill XBlock from X:
              for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
                for (unsigned int iWave = 0; iWave < B; ++iWave)
                  XBlock.local_element(iNode * B + iWave) =
                    X[iNode * totalNumberVectors + jvec + iWave];

              MPI_Barrier(mpiComm);
              // evaluate H times XBlock and store in HXBlock
              HXBlock                = T(0.);
              const bool   scaleFlag = false;
              const double scalar    = 1.0;
              operatorMatrix.HX(XBlock, B, scaleFlag, scalar, HXBlock);

              // compute residual norms:
              for (unsigned int iDof = 0; iDof < localVectorSize; ++iDof)
                for (unsigned int iWave = 0; iWave < B; iWave++)
                  {
                    const double temp =
                      std::abs(HXBlock.local_element(B * iDof + iWave) -
                               eigenValues[jvec + iWave] *
                                 XBlock.local_element(B * iDof + iWave));
                    residualNormSquare[jvec + iWave] += temp * temp;
                  }
            }
        }


      dealii::Utilities::MPI::sum(residualNormSquare,
                                  mpiComm,
                                  residualNormSquare);

      dealii::Utilities::MPI::sum(residualNormSquare,
                                  interBandGroupComm,
                                  residualNormSquare);

      if (dftParameters::verbosity >= 4)
        {
          if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "L-2 Norm of residue   :" << std::endl;
        }
      for (unsigned int iWave = 0; iWave < totalNumberVectors; ++iWave)
        residualNorm[iWave] = sqrt(residualNormSquare[iWave]);

      if (dftParameters::verbosity >= 4 &&
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        for (unsigned int iWave = 0; iWave < totalNumberVectors; ++iWave)
          std::cout << "eigen vector " << iWave << ": " << residualNorm[iWave]
                    << std::endl;

      if (dftParameters::verbosity >= 4)
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << std::endl;
    }

#ifdef USE_COMPLEX
    unsigned int
    lowdenOrthogonalization(std::vector<std::complex<double>> &X,
                            const unsigned int                 numberVectors,
                            const MPI_Comm &                   mpiComm)
    {
      const unsigned int localVectorSize = X.size() / numberVectors;
      std::vector<std::complex<double>> overlapMatrix(numberVectors *
                                                        numberVectors,
                                                      0.0);

      //
      // blas level 3 dgemm flags
      //
      const double       alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;

      //
      // compute overlap matrix S = {(Zc)^T}*Z on local proc
      // where Z is a matrix with size number of degrees of freedom times number
      // of column vectors and (Zc)^T is conjugate transpose of Z Since input
      // "X" is stored as number of column vectors times number of degrees of
      // freedom matrix corresponding to column-major format required for blas,
      // we compute the transpose of overlap matrix i.e S^{T} = X*{(Xc)^T} here
      //
      const char uplo  = 'U';
      const char trans = 'N';

      zherk_(&uplo,
             &trans,
             &numberVectors,
             &localVectorSize,
             &alpha,
             &X[0],
             &numberVectors,
             &beta,
             &overlapMatrix[0],
             &numberVectors);


      dealii::Utilities::MPI::sum(overlapMatrix, mpiComm, overlapMatrix);

      //
      // evaluate the conjugate of {S^T} to get actual overlap matrix
      //
      for (unsigned int i = 0; i < overlapMatrix.size(); ++i)
        overlapMatrix[i] = std::conj(overlapMatrix[i]);


      //
      // set lapack eigen decomposition flags and compute eigendecomposition of
      // S = Q*D*Q^{H}
      //
      int                info;
      const unsigned int lwork = 1 + 6 * numberVectors +
                                 2 * numberVectors * numberVectors,
                         liwork = 3 + 5 * numberVectors;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V';
      const unsigned int lrwork =
        1 + 5 * numberVectors + 2 * numberVectors * numberVectors;
      std::vector<double>               rwork(lrwork, 0.0);
      std::vector<std::complex<double>> work(lwork);
      std::vector<double>               eigenValuesOverlap(numberVectors, 0.0);

      zheevd_(&jobz,
              &uplo,
              &numberVectors,
              &overlapMatrix[0],
              &numberVectors,
              &eigenValuesOverlap[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      rwork.clear();
      std::vector<std::complex<double>>().swap(work);
      std::vector<double>().swap(rwork);
      std::vector<int>().swap(iwork);

      //
      // compute D^{-1/4} where S = Q*D*Q^{H}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,
                                                         0.0);

      unsigned int nanFlag = 0;
      for (unsigned i = 0; i < numberEigenValues; ++i)
        {
          invFourthRootEigenValuesMatrix[i] =
            1.0 / pow(eigenValuesOverlap[i], 1.0 / 4);
          if (std::isnan(invFourthRootEigenValuesMatrix[i]) ||
              eigenValuesOverlap[i] < 1e-13)
            {
              nanFlag = 1;
              break;
            }
        }
      nanFlag = dealii::Utilities::MPI::max(nanFlag, mpiComm);
      if (nanFlag == 1)
        return nanFlag;

      //
      // Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling
      // "zheevd"
      //
      const unsigned int inc = 1;
      for (unsigned int i = 0; i < numberEigenValues; ++i)
        {
          const double scalingCoeff = invFourthRootEigenValuesMatrix[i];
          zdscal_(&numberEigenValues,
                  &scalingCoeff,
                  &overlapMatrix[0] + i * numberEigenValues,
                  &inc);
        }

      //
      // Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{H} = (Q*D^{-1/4})*(Q*D^{-1/4))^{H}
      //
      std::vector<std::complex<double>> invSqrtOverlapMatrix(
        numberEigenValues * numberEigenValues, 0.0);
      const char                 transA1 = 'N';
      const char                 transB1 = 'C';
      const std::complex<double> alpha1 = 1.0, beta1 = 0.0;


      zgemm_(&transA1,
             &transB1,
             &numberEigenValues,
             &numberEigenValues,
             &numberEigenValues,
             &alpha1,
             &overlapMatrix[0],
             &numberEigenValues,
             &overlapMatrix[0],
             &numberEigenValues,
             &beta1,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues);

      //
      // free up memory associated with overlapMatrix
      //
      overlapMatrix.clear();
      std::vector<std::complex<double>>().swap(overlapMatrix);

      //
      // Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but
      // implemented as Y^T = {S^{-1/2}}^T*{X^T} using the column major format
      // of blas
      //
      const char transA2 = 'T', transB2 = 'N';
      // dealii::parallel::distributed::Vector<std::complex<double> >
      // orthoNormalizedBasis;
      std::vector<std::complex<double>> orthoNormalizedBasis(X.size(), 0.0);

      zgemm_(&transA2,
             &transB2,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha1,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta1,
             &orthoNormalizedBasis[0],
             &numberEigenValues);


      X = orthoNormalizedBasis;

      return 0;
    }
#else
    unsigned int
    lowdenOrthogonalization(std::vector<double> &X,
                            const unsigned int numberVectors,
                            const MPI_Comm &mpiComm)
    {
      const unsigned int localVectorSize = X.size() / numberVectors;

      std::vector<double> overlapMatrix(numberVectors * numberVectors, 0.0);


      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(mpiComm,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);



      //
      // blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char uplo = 'U';
      const char trans = 'N';

      //
      // compute overlap matrix S = {(Z)^T}*Z on local proc
      // where Z is a matrix with size number of degrees of freedom times number
      // of column vectors and (Z)^T is transpose of Z Since input "X" is stored
      // as number of column vectors times number of degrees of freedom matrix
      // corresponding to column-major format required for blas, we compute
      // the overlap matrix as S = S^{T} = X*{X^T} here
      //

      computing_timer.enter_subsection("local overlap matrix for lowden");
      dsyrk_(&uplo,
             &trans,
             &numberVectors,
             &localVectorSize,
             &alpha,
             &X[0],
             &numberVectors,
             &beta,
             &overlapMatrix[0],
             &numberVectors);
      computing_timer.leave_subsection("local overlap matrix for lowden");

      dealii::Utilities::MPI::sum(overlapMatrix, mpiComm, overlapMatrix);

      std::vector<double> eigenValuesOverlap(numberVectors);
      computing_timer.enter_subsection("eigen decomp. of overlap matrix");
      callevd(numberVectors, &overlapMatrix[0], &eigenValuesOverlap[0]);
      computing_timer.leave_subsection("eigen decomp. of overlap matrix");

      //
      // compute D^{-1/4} where S = Q*D*Q^{T}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues);
      unsigned int nanFlag = 0;
      for (unsigned i = 0; i < numberEigenValues; ++i)
        {
          invFourthRootEigenValuesMatrix[i] =
            1.0 / pow(eigenValuesOverlap[i], 1.0 / 4);
          if (std::isnan(invFourthRootEigenValuesMatrix[i]) ||
              eigenValuesOverlap[i] < 1e-10)
            {
              nanFlag = 1;
              break;
            }
        }

      nanFlag = dealii::Utilities::MPI::max(nanFlag, mpiComm);
      if (nanFlag == 1)
        return nanFlag;

      if (nanFlag == 1)
        {
          std::cout
            << "Nan obtained: switching to more robust dsyevr for eigen decomposition "
            << std::endl;
          std::vector<double> overlapMatrixEigenVectors(numberVectors *
                                                          numberVectors,
                                                        0.0);
          eigenValuesOverlap.clear();
          eigenValuesOverlap.resize(numberVectors);
          invFourthRootEigenValuesMatrix.clear();
          invFourthRootEigenValuesMatrix.resize(numberVectors);
          computing_timer.enter_subsection("eigen decomp. of overlap matrix");
          callevr(numberVectors,
                  &overlapMatrix[0],
                  &overlapMatrixEigenVectors[0],
                  &eigenValuesOverlap[0]);
          computing_timer.leave_subsection("eigen decomp. of overlap matrix");

          overlapMatrix = overlapMatrixEigenVectors;
          overlapMatrixEigenVectors.clear();
          std::vector<double>().swap(overlapMatrixEigenVectors);

          //
          // compute D^{-1/4} where S = Q*D*Q^{T}
          //
          for (unsigned i = 0; i < numberEigenValues; ++i)
            {
              invFourthRootEigenValuesMatrix[i] =
                1.0 / pow(eigenValuesOverlap[i], (1.0 / 4.0));
              AssertThrow(
                !std::isnan(invFourthRootEigenValuesMatrix[i]),
                dealii::ExcMessage(
                  "Eigen values of overlap matrix during Lowden Orthonormalization are close to zero."));
            }
        }

      //
      // Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling
      // "dsyevd"
      //
      computing_timer.enter_subsection("scaling in Lowden");
      const unsigned int inc = 1;
      for (unsigned int i = 0; i < numberEigenValues; ++i)
        {
          double scalingCoeff = invFourthRootEigenValuesMatrix[i];
          dscal_(&numberEigenValues,
                 &scalingCoeff,
                 &overlapMatrix[0] + i * numberEigenValues,
                 &inc);
        }
      computing_timer.leave_subsection("scaling in Lowden");

      //
      // Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{T} = (Q*D^{-1/4})*(Q*D^{-1/4}))^{T}
      //
      std::vector<double> invSqrtOverlapMatrix(numberEigenValues *
                                                 numberEigenValues,
                                               0.0);
      const char transA1 = 'N';
      const char transB1 = 'T';
      computing_timer.enter_subsection("inverse sqrt overlap");
      dgemm_(&transA1,
             &transB1,
             &numberEigenValues,
             &numberEigenValues,
             &numberEigenValues,
             &alpha,
             &overlapMatrix[0],
             &numberEigenValues,
             &overlapMatrix[0],
             &numberEigenValues,
             &beta,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues);
      computing_timer.leave_subsection("inverse sqrt overlap");

      //
      // free up memory associated with overlapMatrix
      //
      overlapMatrix.clear();
      std::vector<double>().swap(overlapMatrix);

      //
      // Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but
      // implemented as Yt = S^{-1/2}*Xt using the column major format of blas
      //
      const char transA2 = 'N', transB2 = 'N';
      // dealii::parallel::distributed::Vector<double> orthoNormalizedBasis;
      // orthoNormalizedBasis.reinit(X);
      std::vector<double> orthoNormalizedBasis(X.size(), 0.0);

      computing_timer.enter_subsection("subspace rotation in lowden");
      dgemm_(&transA2,
             &transB2,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &orthoNormalizedBasis[0],
             &numberEigenValues);
      computing_timer.leave_subsection("subspace rotation in lowden");


      X = orthoNormalizedBasis;

      return 0;
    }
#endif

    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    template <typename T>
    std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(operatorDFTClass &operatorMatrix,
                                        const distributedCPUVec<T> &vect)
    {
      const unsigned int this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(
          operatorMatrix.getMPICommunicator());

      const unsigned int lanczosIterations =
        dftParameters::reproducible_output ? 40 : 20;
      double beta;


      T alpha, alphaNeg;

      //
      // generate random vector v
      //
      distributedCPUVec<T> vVector, fVector, v0Vector;
      vVector.reinit(vect);
      fVector.reinit(vect);

      vVector = T(0.0), fVector = T(0.0);
      std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      for (unsigned int i = 0; i < local_size; i++)
        vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

      operatorMatrix.getOverloadedConstraintMatrix()->set_zero(vVector, 1);

      vVector.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVector /= vVector.l2_norm();
      vVector.update_ghost_values();

      //
      // call matrix times X
      //
      fVector                = T(0);
      const bool   scaleFlag = false;
      const double scalar    = 1.0;
      operatorMatrix.HX(vVector, 1, scaleFlag, scalar, fVector);

      // evaluate fVector^{H}*vVector
      alpha = fVector * vVector;
      fVector.add(-1.0 * alpha, vVector);
      std::vector<T> Tlanczos(lanczosIterations * lanczosIterations, 0.0);

      Tlanczos[0]    = alpha;
      unsigned index = 0;

      // filling only lower triangular part
      for (unsigned int j = 1; j < lanczosIterations; j++)
        {
          beta     = fVector.l2_norm();
          v0Vector = vVector;
          vVector.equ(1.0 / beta, fVector);

          fVector = T(0);
          operatorMatrix.HX(vVector, 1, scaleFlag, scalar, fVector);

          fVector.add(-1.0 * beta, v0Vector); // beta is real

          alpha = fVector * vVector;
          fVector.add(-1.0 * alpha, vVector);

          index += 1;
          Tlanczos[index] = beta;
          index += lanczosIterations;
          Tlanczos[index] = alpha;
        }

      // eigen decomposition to find max eigen value of T matrix
      std::vector<double> eigenValuesT(lanczosIterations);
      char                jobz = 'N', uplo = 'L';
      const unsigned int  n = lanczosIterations, lda = lanczosIterations;
      int                 info;
      const unsigned int  lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
      std::vector<int>    iwork(liwork, 0);

#ifdef USE_COMPLEX
      const unsigned int                lrwork = 1 + 5 * n + 2 * n * n;
      std::vector<double>               rwork(lrwork, 0.0);
      std::vector<std::complex<double>> work(lwork);
      zheevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);
#else
      std::vector<double> work(lwork, 0.0);
      dsyevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);
#endif


      for (unsigned int i = 0; i < eigenValuesT.size(); i++)
        {
          eigenValuesT[i] = eigenValuesT[i];
        }
      std::sort(eigenValuesT.begin(), eigenValuesT.end());
      //
      const double fvectorNorm = fVector.l2_norm();
      if (dftParameters::verbosity >= 5 && this_mpi_process == 0)
        {
          std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                    << ", fvector norm: " << fvectorNorm << std::endl;
          std::cout << "aLow: " << eigenValuesT[0] << std::endl;
        }

      double lowerBound = std::floor(eigenValuesT[0]);
      double upperBound =
        std::ceil(eigenValuesT[lanczosIterations - 1] +
                  (dftParameters::reproducible_output ? fvectorNorm :
                                                        fvectorNorm / 10.0));
      return (std::make_pair(lowerBound, upperBound));
    }


    template std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(
      operatorDFTClass &,
      const distributedCPUVec<dataTypes::number> &);


    template void
    chebyshevFilter(operatorDFTClass &operatorMatrix,
                    distributedCPUVec<dataTypes::number> &,
                    const unsigned int,
                    const unsigned int,
                    const double,
                    const double,
                    const double);

    template void
    chebyshevFilterOpt(operatorDFTClass &                    operatorMatrix,
                       distributedCPUVec<dataTypes::number> &X,
                       std::vector<dataTypes::number> &cellWaveFunctionMatrix,
                       const unsigned int              numberComponents,
                       const unsigned int              m,
                       const double                    a,
                       const double                    b,
                       const double                    a0);


    template void
    gramSchmidtOrthogonalization(std::vector<dataTypes::number> &,
                                 const unsigned int,
                                 const MPI_Comm &);

    template unsigned int
    pseudoGramSchmidtOrthogonalization(elpaScalaManager &elpaScala,
                                       std::vector<dataTypes::number> &,
                                       const unsigned int,
                                       const MPI_Comm &,
                                       const MPI_Comm &mpiComm,
                                       const bool      useMixedPrec);

    template void
    rayleighRitz(operatorDFTClass &operatorMatrix,
                 elpaScalaManager &elpaScala,
                 std::vector<dataTypes::number> &,
                 const unsigned int numberWaveFunctions,
                 const MPI_Comm &,
                 const MPI_Comm &,
                 std::vector<double> &eigenValues,
                 const bool           doCommAfterBandParal);

    template void
    rayleighRitzGEP(operatorDFTClass &operatorMatrix,
                    elpaScalaManager &elpaScala,
                    std::vector<dataTypes::number> &,
                    const unsigned int numberWaveFunctions,
                    const MPI_Comm &,
                    const MPI_Comm &,
                    std::vector<double> &eigenValues,
                    const bool           useMixedPrec);


    template void
    rayleighRitzSpectrumSplitDirect(operatorDFTClass &operatorMatrix,
                                    elpaScalaManager &elpaScala,
                                    const std::vector<dataTypes::number> &,
                                    std::vector<dataTypes::number> &,
                                    const unsigned int numberWaveFunctions,
                                    const unsigned int numberCoreStates,
                                    const MPI_Comm &,
                                    const MPI_Comm &,
                                    const bool           useMixedPrec,
                                    std::vector<double> &eigenValues);

    template void
    rayleighRitzGEPSpectrumSplitDirect(operatorDFTClass &operatorMatrix,
                                       elpaScalaManager &elpaScala,
                                       std::vector<dataTypes::number> &X,
                                       std::vector<dataTypes::number> &Y,
                                       const unsigned int   numberWaveFunctions,
                                       const unsigned int   numberCoreStates,
                                       const MPI_Comm &     interBandGroupComm,
                                       const MPI_Comm &     mpiComm,
                                       const bool           useMixedPrec,
                                       std::vector<double> &eigenValues);

    template void
    computeEigenResidualNorm(operatorDFTClass &              operatorMatrix,
                             std::vector<dataTypes::number> &X,
                             const std::vector<double> &     eigenValues,
                             const MPI_Comm &                mpiComm,
                             const MPI_Comm &                interBandGroupComm,
                             std::vector<double> &           residualNorm);

  } // namespace linearAlgebraOperations

} // namespace dftfe
