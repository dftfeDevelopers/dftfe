// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
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

#include "dftParameters.h"
#include "dftUtils.h"
#include "linearAlgebraOperations.h"
#include "linearAlgebraOperationsInternal.h"
#include "constants.h"
#include <DeviceAPICalls.h>

namespace dftfe
{
  namespace linearAlgebraOperations
  {
    //
    // chebyshev filtering of given subspace XArray
    //
    template <typename T, dftfe::utils::MemorySpace memorySpace>
    void
    chebyshevFilter(operatorDFTClass<memorySpace> &operatorMatrix,
                    dftfe::linearAlgebra::MultiVector<T, memorySpace> &X,
                    dftfe::linearAlgebra::MultiVector<T, memorySpace> &Y,
                    const dftfe::uInt                                  m,
                    const double                                       a,
                    const double                                       b,
                    const double                                       a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;


      //
      // create YArray
      // initialize to zeros.
      // x
      Y.setValue(T(0.0));


      //
      // call HX
      //
      double alpha1 = sigma1 / e, alpha2 = -c;
      operatorMatrix.overlapMatrixTimesX(X, 1.0, 0.0, 0.0, Y);

      operatorMatrix.HXCheby(Y, alpha1, 0.0, alpha1 * alpha2, X);
      X.swap(Y);
      //
      // polynomial loop
      //
      for (dftfe::uInt degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);



          //
          // call HX
          //
          operatorMatrix.HXCheby(Y, alpha1, alpha2, -c * alpha1, X);


          //
          // XArray = YArray
          //
          X.swap(Y);

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }
      operatorMatrix.overlapInverseMatrixTimesX(Y, 1.0, 0.0, 0.0, X);
      // copy back YArray to XArray
      // X = Y;
    }


    template <typename T1, typename T2, dftfe::utils::MemorySpace memorySpace>
    void
    reformulatedChebyshevFilter(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                         &BLASWrapperPtr,
      operatorDFTClass<memorySpace>                      &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<T1, memorySpace> &X,
      dftfe::linearAlgebra::MultiVector<T1, memorySpace> &Y,
      dftfe::linearAlgebra::MultiVector<T2, memorySpace> &Residual,
      dftfe::linearAlgebra::MultiVector<T2, memorySpace> &ResidualNew,
      std::vector<double>                                 eigenvalues,
      const dftfe::uInt                                   m,
      const double                                        a,
      const double                                        b,
      const double                                        a0,
      const bool                                          approxOverlapMatrix)
    {
      double e, c, sigma, sigma1, sigma2, gamma;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;



      dftfe::utils::MemoryStorage<double, memorySpace> eigenValuesFiltered,
        eigenValuesFiltered1, eigenValuesFiltered2;
      eigenValuesFiltered.resize(eigenvalues.size());
      eigenValuesFiltered.copyFrom(eigenvalues);
      eigenValuesFiltered1 = eigenValuesFiltered;
      eigenValuesFiltered2 = eigenValuesFiltered;
      eigenValuesFiltered1.setValue(1.0);


      double alpha1 = sigma1 / e, alpha2 = -c;
      // //compute initial Residual
      operatorMatrix.overlapMatrixTimesX(
        X, 1.0, 0.0, 0.0, Y, approxOverlapMatrix);
      BLASWrapperPtr->rightDiagonalScale(Y.numVectors(),
                                         Y.locallyOwnedSize(),
                                         Y.data(),
                                         eigenValuesFiltered.data());
      operatorMatrix.HX(X, 1.0, -1.0, 0.0, Y);

      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
        X.locallyOwnedSize() * X.numVectors(), Y.data(), ResidualNew.data());
      Residual.setValue(0.0);

      // //m=1 operations
      eigenValuesFiltered2.setValue(alpha1 * alpha2);
      BLASWrapperPtr->ApaBD(1,
                            eigenValuesFiltered2.size(),
                            alpha1,
                            eigenValuesFiltered2.data(),
                            eigenValuesFiltered1.data(),
                            eigenValuesFiltered.data(),
                            eigenValuesFiltered2.data());
      BLASWrapperPtr->xscal(ResidualNew.data(),
                            T2(alpha1),
                            X.locallyOwnedSize() * X.numVectors());


      // //
      // // polynomial loop
      // //
      for (dftfe::uInt degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          operatorMatrix.HXCheby(
            ResidualNew, alpha1, alpha2, -c * alpha1, Residual);

          BLASWrapperPtr->ApaBD(X.locallyOwnedSize(),
                                X.numVectors(),
                                alpha1,
                                Residual.data(),
                                Y.data(),
                                eigenValuesFiltered2.data(),
                                Residual.data());
          BLASWrapperPtr->axpby(eigenValuesFiltered2.size(),
                                -c * alpha1,
                                eigenValuesFiltered2.data(),
                                alpha2,
                                eigenValuesFiltered1.data());
          BLASWrapperPtr->ApaBD(1,
                                eigenValuesFiltered1.size(),
                                alpha1,
                                eigenValuesFiltered1.data(),
                                eigenValuesFiltered2.data(),
                                eigenValuesFiltered.data(),
                                eigenValuesFiltered1.data());

          //
          // XArray = YArray
          //
          ResidualNew.swap(Residual);
          eigenValuesFiltered1.swap(eigenValuesFiltered2);

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }
      operatorMatrix.overlapInverseMatrixTimesX(
        ResidualNew, 1.0, 0.0, 0.0, Residual);

      BLASWrapperPtr->ApaBD(X.locallyOwnedSize(),
                            X.numVectors(),
                            1.0,
                            Residual.data(),
                            X.data(),
                            eigenValuesFiltered2.data(),
                            X.data());
    }


    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    template <typename T, dftfe::utils::MemorySpace memorySpace>
    std::pair<double, double>
    generalisedLanczosLowerUpperBoundEigenSpectrum(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                        &BLASWrapperPtr,
      operatorDFTClass<memorySpace>                     &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<T, memorySpace> &X,
      dftfe::linearAlgebra::MultiVector<T, memorySpace> &Y,
      dftfe::linearAlgebra::MultiVector<T, memorySpace> &Z,
      dftfe::linearAlgebra::MultiVector<T, memorySpace> &tempVec,
      const dftParameters                               &dftParams)
    {
      const dftfe::uInt this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(
          operatorMatrix.getMPICommunicatorDomain());

      const dftfe::uInt lanczosIterations =
        dftParams.reproducible_output ? 60 : 20;
      double beta, betaNeg;
      T      betaTemp;

      T alpha, alphaNeg;

      //
      // generate random vector v
      //
      X.setValue(T(0.0));
      Y.setValue(T(0.0));
      Z.setValue(T(0.0));
      tempVec.setValue(T(0.0));
      const dftfe::uInt local_size = X.locallyOwnedSize();
#if defined(DFTFE_WITH_DEVICE)
      dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST> XHost(
        local_size, T(0.0));
      T *XHostDataPtr = XHost.data();
#else
      T                  *XHostDataPtr = X.data();
#endif


      std::srand(this_mpi_process);
      for (dftfe::uInt i = 0; i < local_size; i++)
        XHostDataPtr[i] = ((double)std::rand()) / ((double)RAND_MAX);

#if defined(DFTFE_WITH_DEVICE)
      XHost.template copyTo<memorySpace>(X.data());
#endif

      operatorMatrix.getOverloadedConstraintMatrix()->set_zero(X);
      X.zeroOutGhosts();
      //
      // evaluate l2 norm
      //

      T Onormsq;
      operatorMatrix.overlapMatrixTimesX(X, 1.0, 0.0, 0.0, tempVec, true);

      BLASWrapperPtr->xdot(local_size,
                           X.data(),
                           1,
                           tempVec.data(),
                           1,
                           operatorMatrix.getMPICommunicatorDomain(),
                           &Onormsq);
      BLASWrapperPtr->xscal(X.data(),
                            1.0 / sqrt(std::abs(Onormsq)),
                            local_size);


      operatorMatrix.HX(X, 1.0, 0.0, 0.0, tempVec);
      operatorMatrix.overlapInverseMatrixTimesX(tempVec, 1.0, 0.0, 0.0, Y);


      BLASWrapperPtr->xdot(local_size,
                           tempVec.data(),
                           1,
                           X.data(),
                           1,
                           operatorMatrix.getMPICommunicatorDomain(),
                           &alpha);

      alphaNeg = -alpha;
      BLASWrapperPtr->xaxpy(local_size, &alphaNeg, X.data(), 1, Y.data(), 1);

      std::vector<T> Tlanczos(lanczosIterations * lanczosIterations, 0.0);

      Tlanczos[0]       = alpha;
      dftfe::uInt index = 0;
      // filling only lower triangular part
      for (dftfe::uInt j = 1; j < lanczosIterations; j++)
        {
          operatorMatrix.overlapMatrixTimesX(Y, 1.0, 0.0, 0.0, tempVec, true);
          BLASWrapperPtr->xdot(local_size,
                               Y.data(),
                               1,
                               tempVec.data(),
                               1,
                               operatorMatrix.getMPICommunicatorDomain(),
                               &betaTemp);
          beta = sqrt(std::abs(betaTemp));
          Z    = X;
          BLASWrapperPtr->axpby(
            local_size, 1.0 / beta, Y.data(), 0.0, X.data());

          operatorMatrix.HX(X, 1.0, 0.0, 0.0, tempVec);
          operatorMatrix.overlapInverseMatrixTimesX(tempVec, 1.0, 0.0, 0.0, Y);


          alphaNeg = -beta;
          BLASWrapperPtr->xaxpy(
            local_size, &alphaNeg, Z.data(), 1, Y.data(), 1);

          BLASWrapperPtr->xdot(local_size,
                               tempVec.data(),
                               1,
                               X.data(),
                               1,
                               operatorMatrix.getMPICommunicatorDomain(),
                               &alpha);
          alphaNeg = -alpha;
          BLASWrapperPtr->xaxpy(
            local_size, &alphaNeg, X.data(), 1, Y.data(), 1);

          index += 1;
          Tlanczos[index] = beta;
          index += lanczosIterations;
          Tlanczos[index] = alpha;
        }
      operatorMatrix.overlapMatrixTimesX(Y, 1.0, 0.0, 0.0, tempVec, true);
      BLASWrapperPtr->xdot(local_size,
                           Y.data(),
                           1,
                           tempVec.data(),
                           1,
                           operatorMatrix.getMPICommunicatorDomain(),
                           &betaTemp);
      beta = sqrt(std::abs(betaTemp));
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


      std::sort(eigenValuesT.begin(), eigenValuesT.end());
      //
      double YNorm;
      BLASWrapperPtr->xnrm2(local_size,
                            Y.data(),
                            1,
                            operatorMatrix.getMPICommunicatorDomain(),
                            &YNorm);
      double lowerBound = std::floor(eigenValuesT[0]);
      double upperBound =
        std::ceil(eigenValuesT[lanczosIterations - 1] +
                  (dftParams.reproducible_output ? beta : beta / 10.0));

      if (dftParams.verbosity >= 5 && this_mpi_process == 0)
        {
          std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                    << ", fvector norm: " << beta << std::endl;
          std::cout << "aLow: " << eigenValuesT[0] << std::endl;
          std::cout << "boundL: " << lowerBound << std::endl;
          std::cout << "boundU: " << upperBound << std::endl;
        }

      return (std::make_pair(lowerBound, upperBound));
    }



    template std::pair<double, double>
    generalisedLanczosLowerUpperBoundEigenSpectrum(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      const dftParameters &dftParams);
#ifdef DFTFE_WITH_DEVICE
    template std::pair<double, double>
    generalisedLanczosLowerUpperBoundEigenSpectrum(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      const dftParameters &dftParams);
#endif



    template void
    chebyshevFilter(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &,
      const dftfe::uInt,
      const double,
      const double,
      const double);


    template void
    reformulatedChebyshevFilter(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &X,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &Y,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::HOST>
        &Residual,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::HOST>
                         &ResidualNew,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);
    template void
    reformulatedChebyshevFilter(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &X,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &Y,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST>
        &Residual,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST>
                         &ResidualNew,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);

#ifdef DFTFE_WITH_DEVICE
    template void
    chebyshevFilter(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &,
      const dftfe::uInt,
      const double,
      const double,
      const double);


    template void
    reformulatedChebyshevFilter(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                          &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
        &Residual,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
                         &ResidualNew,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);

    template void
    reformulatedChebyshevFilter(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                          &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &Residual,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
                         &ResidualNew,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);

#endif



  } // namespace linearAlgebraOperations

} // namespace dftfe
