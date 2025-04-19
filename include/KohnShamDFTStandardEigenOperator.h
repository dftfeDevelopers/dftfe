// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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


#ifndef KohnShamDFTStandardEigenOperatorClass_H_
#define KohnShamDFTStandardEigenOperatorClass_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <headers.h>
#include <KohnShamDFTBaseOperator.h>
#include <BLASWrapper.h>
#include <FEBasisOperations.h>
#include <oncvClass.h>
#include <AuxDensityMatrix.h>

#include "hubbardClass.h"

namespace dftfe
{
  // KohnShamDFTStandardEigenOperator is derived from KohnShamDFTBaseOperator
  // base class. This class is used where the underlying PDE is a
  // standard eigen value problem. Currently used for the all-electron
  // and norm conserving pseudopotential
  template <dftfe::utils::MemorySpace memorySpace>
  class KohnShamDFTStandardEigenOperator
    : public KohnShamDFTBaseOperator<memorySpace>
  {
  public:
    KohnShamDFTStandardEigenOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<
        dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                               pseudopotentialClassPtr,
      std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      dftParameters                           *dftParamsPtr,
      const dftfe::uInt                        densityQuadratureID,
      const dftfe::uInt                        lpspQuadratureID,
      const dftfe::uInt                        feOrderPlusOneQuadratureID,
      const MPI_Comm                          &mpi_comm_parent,
      const MPI_Comm                          &mpi_comm_domain);



    /**
     * @brief Computing Y = scalarOX*OX + scalarX*X + scalarY*Y for a given X and Y in full precision
     *
     * @param src X vector
     * @param scalarHX scalar for OX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param useApproximateMatrixEntries flag to use approximate overlap matrix
     */
    void
    overlapMatrixTimesX(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      const double scalarOX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const bool useApproximateMatrixEntries = true);



    /**
     * @brief Computing Y = scalarOinvX*O^{-1}X + scalarX*X + scalarY*Y for a given X and Y in full precision
     *
     * @param src X vector
     * @param scalarOinvX scalar for O^{-1}X
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     */
    void
    overlapInverseMatrixTimesX(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      const double scalarOinvX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst);

    /**
     * @brief Computing Y = scalarOinvX*O^{-1}X + scalarX*X + scalarY*Y for a given X and Y in Reduced precision
     *
     * @param src X vector
     * @param scalarOinvX scalar for O^{-1}X
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     */
    void
    overlapInverseMatrixTimesX(
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                  &src,
      const double scalarOinvX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
        &dst);

    /**
     * @brief Computing Y = scalarHX*HM^{-1}X + scalarX*X + scalarY*Y for a given X and Y in reduced precision
     *
     * @param src X vector
     * @param scalarHX scalar for HX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
     * @param skip1 flag to skip extraction
     * @param skip2 flag to skip nonLoal All Reduce
     * @param skip3 flag to skip local HX and Assembly
     */

    void
    HXCheby(dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                              memorySpace> &src,
            const double                                    scalarHX,
            const double                                    scalarY,
            const double                                    scalarX,
            dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                              memorySpace> &dst,
            const bool onlyHPrimePartForFirstOrderDensityMatResponse,
            const bool skip1,
            const bool skip2,
            const bool skip3);

    /**
     * @brief Computing Y = scalarHX*M^{-1}HX + scalarX*X + scalarY*Y for a given X and Y in full precision
     *
     * @param src X vector
     * @param scalarHX scalar for HX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
     * @param skip1 flag to skip extraction
     * @param skip2 flag to skip nonLoal All Reduce
     * @param skip3 flag to skip local HX and Assembly
     */

    void
    HXCheby(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      const double scalarHX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false,
      const bool skip1                                         = false,
      const bool skip2                                         = false,
      const bool skip3                                         = false);

    using KohnShamDFTBaseOperator<memorySpace>::getInverseSqrtMassVector;
    using KohnShamDFTBaseOperator<memorySpace>::getSqrtMassVector;


  protected:
    using KohnShamDFTBaseOperator<
      memorySpace>::d_pseudopotentialNonLocalOperator;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_pseudopotentialNonLocalOperatorSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_BLASWrapperPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_basisOperationsPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_basisOperationsPtrHost;
    using KohnShamDFTBaseOperator<memorySpace>::d_pseudopotentialClassPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_excManagerPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_dftParamsPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_cellHamiltonianMatrix;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_cellHamiltonianMatrixSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_cellHamiltonianMatrixExtPot;
    using KohnShamDFTBaseOperator<memorySpace>::d_cellWaveFunctionMatrixSrc;
    using KohnShamDFTBaseOperator<memorySpace>::d_cellWaveFunctionMatrixDst;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_cellWaveFunctionMatrixSrcSinglePrec;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_cellWaveFunctionMatrixDstSinglePrec;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_pseudopotentialNonLocalProjectorTimesVectorBlock;
    using KohnShamDFTBaseOperator<memorySpace>::
      d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_tempBlockVectorOverlapInvX;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_tempBlockVectorOverlapInvXSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_VeffJxW;
    using KohnShamDFTBaseOperator<memorySpace>::d_VeffExtPotJxW;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_invJacderExcWithSigmaTimesGradRhoJxW;
    using KohnShamDFTBaseOperator<
      memorySpace>::inverseMassVectorScaledConstraintsNoneDataInfoPtr;
    using KohnShamDFTBaseOperator<
      memorySpace>::inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_kPointCoordinates;
    using KohnShamDFTBaseOperator<memorySpace>::d_kPointWeights;
    using KohnShamDFTBaseOperator<memorySpace>::tempHamMatrixRealBlock;
    using KohnShamDFTBaseOperator<memorySpace>::tempHamMatrixImagBlock;
    using KohnShamDFTBaseOperator<memorySpace>::d_densityQuadratureID;
    using KohnShamDFTBaseOperator<memorySpace>::d_lpspQuadratureID;
    using KohnShamDFTBaseOperator<memorySpace>::d_feOrderPlusOneQuadratureID;
    using KohnShamDFTBaseOperator<memorySpace>::d_kPointIndex;
    using KohnShamDFTBaseOperator<memorySpace>::d_spinIndex;
    using KohnShamDFTBaseOperator<memorySpace>::d_HamiltonianIndex;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_isExternalPotCorrHamiltonianComputed;
    using KohnShamDFTBaseOperator<memorySpace>::d_mpiCommParent;
    using KohnShamDFTBaseOperator<memorySpace>::d_mpiCommDomain;
    using KohnShamDFTBaseOperator<memorySpace>::n_mpi_processes;
    using KohnShamDFTBaseOperator<memorySpace>::this_mpi_process;
    using KohnShamDFTBaseOperator<
      memorySpace>::d_cellsBlockSizeHamiltonianConstruction;
    using KohnShamDFTBaseOperator<memorySpace>::d_cellsBlockSizeHX;
    using KohnShamDFTBaseOperator<memorySpace>::d_numVectorsInternal;
    using KohnShamDFTBaseOperator<memorySpace>::d_nOMPThreads;
    using KohnShamDFTBaseOperator<memorySpace>::pcout;
    using KohnShamDFTBaseOperator<memorySpace>::computing_timer;
    using KohnShamDFTBaseOperator<memorySpace>::d_hubbardClassPtr;
    using KohnShamDFTBaseOperator<memorySpace>::d_useHubbard;
    using KohnShamDFTBaseOperator<memorySpace>::d_srcNonLocalTemp;
    using KohnShamDFTBaseOperator<memorySpace>::d_dstNonLocalTemp;
    using KohnShamDFTBaseOperator<memorySpace>::d_srcNonLocalTempSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_dstNonLocalTempSinglePrec;
    using KohnShamDFTBaseOperator<memorySpace>::d_mapNodeIdToProcId;
    using KohnShamDFTBaseOperator<memorySpace>::reinitkPointSpinIndex;
    using KohnShamDFTBaseOperator<memorySpace>::reinitNumberWavefunctions;
  };
} // namespace dftfe
#endif
