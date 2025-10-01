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


#ifndef KohnShamDFTBaseOperatorClass_H_
#define KohnShamDFTBaseOperatorClass_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <headers.h>
#include <operator.h>
#include <BLASWrapper.h>
#include <FEBasisOperations.h>
#include <pseudopotentialBaseClass.h>
#include <AuxDensityMatrix.h>

#include "hubbardClass.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class KohnShamDFTBaseOperator : public operatorDFTClass<memorySpace>
  {
  public:
    KohnShamDFTBaseOperator(
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

    void
    init(const std::vector<double> &kPointCoordinates,
         const std::vector<double> &kPointWeights);

    /*
     * Sets the d_isExternalPotCorrHamiltonianComputed to false
     */
    void
    resetExtPotHamFlag();

    void
    resetKohnShamOp();


    const MPI_Comm &
    getMPICommunicatorDomain();

    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST> *
    getOverloadedConstraintMatrixHost() const;

    dftUtils::constraintMatrixInfo<memorySpace> *
    getOverloadedConstraintMatrix() const
    {
      return &(d_basisOperationsPtr
                 ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
    }

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &
    getScratchFEMultivector(const dftfe::uInt numVectors,
                            const dftfe::uInt index);


    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
    getScratchFEMultivectorSinglePrec(const dftfe::uInt numVectors,
                                      const dftfe::uInt index);


    /**
     * @brief Computes effective potential involving exchange-correlation functionals
     * @param auxDensityMatrixRepresentation core plus valence electron-density
     * @param phiValues electrostatic potential arising both from electron-density and nuclear charge
     */
    void
    computeVEff(
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCRepresentationPtr,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                       &phiValues,
      const dftfe::uInt spinIndex = 0);

    /**
     * @brief Sets the V-eff potential
     * @param vKS_quadValues the input V-KS values stored at the quadrature points
     * @param spinIndex spin index
     */
    void
    setVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                       &vKS_quadValues,
      const dftfe::uInt spinIndex);

    void
    computeVEffExternalPotCorr(
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues);

    void
    computeVEffPrime(
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCRepresentationPtr,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                       &phiPrimeValues,
      const dftfe::uInt spinIndex);

    /**
     * @brief sets the data member to appropriate kPoint and spin Index
     *
     * @param kPointIndex  k-point Index to set
     */
    void
    reinitkPointSpinIndex(const dftfe::uInt kPointIndex,
                          const dftfe::uInt spinIndex);

    void
    reinitNumberWavefunctions(const dftfe::uInt numWfc);

    const dftfe::utils::MemoryStorage<double, memorySpace> &
    getInverseSqrtMassVector();

    const dftfe::utils::MemoryStorage<double, memorySpace> &
    getSqrtMassVector();

    void
    computeCellHamiltonianMatrix(
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    computeCellHamiltonianMatrixExtPotContribution();

    /**
     * @brief Computing Y = scalarHX*HX + scalarX*X + scalarY*Y for a given X and Y in full precision
     *
     * @param src X vector
     * @param scalarHX scalar for HX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
     */
    void
    HX(dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
       const double scalarHX,
       const double scalarY,
       const double scalarX,
       dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse =
         false) override;

    /**
     * @brief Computing Y = scalarHX*HX + scalarX*X + scalarY*Y for a given X and Y in full precision
     *
     * @param src X vector
     * @param scalarHX scalar for HX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
     */
    void
    HX(dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                   &src,
       const double scalarHX,
       const double scalarY,
       const double scalarX,
       dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                 &dst,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse =
         false) override;
    /**
     * @brief Computing Y = scalarHX*M^{1/2}HM^{1/2}X + scalarX*X + scalarY*Y for a given X and Y in full precision. Used for TD-DFT and Inverse DFT calc.
     *
     * @param src X vector
     * @param scalarHX scalar for HX
     * @param scalarY scalar for Y
     * @param scalarX scalar for X
     * @param dst Y vector
     * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
     */
    void
    HXWithLowdinOrthonormalisedInput(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      const double scalarHX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    // /**
    //  * @brief Computing Y = scalarOX*OX + scalarX*X + scalarY*Y for a given X and Y in full precision
    //  *
    //  * @param src X vector
    //  * @param scalarHX scalar for OX
    //  * @param scalarY scalar for Y
    //  * @param scalarX scalar for X
    //  * @param dst Y vector
    //  * @param useApproximateMatrixEntries flag to use approximate overlap matrix
    //  */
    // void
    // overlapMatrixTimesX(
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    //   const double scalarOX,
    //   const double scalarY,
    //   const double scalarX,
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    //   const bool useApproximateMatrixEntries = true);



    // /**
    //  * @brief Computing Y = scalarOinvX*O^{-1}X + scalarX*X + scalarY*Y for a given X and Y in full precision
    //  *
    //  * @param src X vector
    //  * @param scalarOinvX scalar for O^{-1}X
    //  * @param scalarY scalar for Y
    //  * @param scalarX scalar for X
    //  * @param dst Y vector
    //  */
    // void
    // overlapInverseMatrixTimesX(
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    //   const double scalarOinvX,
    //   const double scalarY,
    //   const double scalarX,
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
    //   &dst);

    // /**
    //  * @brief Computing Y = scalarOinvX*O^{-1}X + scalarX*X + scalarY*Y for a given X and Y in Reduced precision
    //  *
    //  * @param src X vector
    //  * @param scalarOinvX scalar for O^{-1}X
    //  * @param scalarY scalar for Y
    //  * @param scalarX scalar for X
    //  * @param dst Y vector
    //  */
    // void
    // overlapInverseMatrixTimesX(
    //   dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
    //     &          src,
    //   const double scalarOinvX,
    //   const double scalarY,
    //   const double scalarX,
    //   dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
    //     &dst);

    // /**
    //  * @brief Computing Y = scalarHX*HM^{-1}X + scalarX*X + scalarY*Y for a given X and Y in reduced precision
    //  *
    //  * @param src X vector
    //  * @param scalarHX scalar for HX
    //  * @param scalarY scalar for Y
    //  * @param scalarX scalar for X
    //  * @param dst Y vector
    //  * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
    //  * @param skip1 flag to skip extraction
    //  * @param skip2 flag to skip nonLoal All Reduce
    //  * @param skip3 flag to skip local HX and Assembly
    //  */

    // void
    // HXCheby(dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
    //                                           memorySpace> &src,
    //         const double                                    scalarHX,
    //         const double                                    scalarY,
    //         const double                                    scalarX,
    //         dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
    //                                           memorySpace> &dst,
    //         const bool onlyHPrimePartForFirstOrderDensityMatResponse,
    //         const bool skip1,
    //         const bool skip2,
    //         const bool skip3);

    // /**
    //  * @brief Computing Y = scalarHX*M^{-1}HX + scalarX*X + scalarY*Y for a given X and Y in full precision
    //  *
    //  * @param src X vector
    //  * @param scalarHX scalar for HX
    //  * @param scalarY scalar for Y
    //  * @param scalarX scalar for X
    //  * @param dst Y vector
    //  * @param onlyHPrimePartForFirstOrderDensityMatResponse flag to compute only HPrime part for first order density matrix response
    //  * @param skip1 flag to skip extraction
    //  * @param skip2 flag to skip nonLoal All Reduce
    //  * @param skip3 flag to skip local HX and Assembly
    //  */

    // void
    // HXCheby(
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    //   const double scalarHX,
    //   const double scalarY,
    //   const double scalarX,
    //   dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    //   const bool onlyHPrimePartForFirstOrderDensityMatResponse = false,
    //   const bool skip1                                         = false,
    //   const bool skip2                                         = false,
    //   const bool skip3                                         = false);



    void
    setVEffExternalPotCorrToZero();

  protected:
    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
      d_pseudopotentialNonLocalOperator;


    /*
     * TODO  ------------------------------
     * TODO For debugging Purposes:  remove afterwards
     * TODO --------------------------------
     */

    // std::shared_ptr<
    //  AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
    //  d_HubbnonLocalOperator;

    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::numberFP32, memorySpace>>
      d_pseudopotentialNonLocalOperatorSinglePrec;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperationsPtr;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;
    std::shared_ptr<
      dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                             d_pseudopotentialClassPtr;
    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr;
    dftParameters                           *d_dftParamsPtr;

    std::vector<dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>>
      d_cellHamiltonianMatrix;
    std::vector<dftfe::utils::MemoryStorage<dataTypes::numberFP32, memorySpace>>
      d_cellHamiltonianMatrixSinglePrec;
    dftfe::utils::MemoryStorage<double, memorySpace>
      d_cellHamiltonianMatrixExtPot;


    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      d_cellWaveFunctionMatrixSrc;
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      d_cellWaveFunctionMatrixDst;

    dftfe::utils::MemoryStorage<dataTypes::numberFP32, memorySpace>
      d_cellWaveFunctionMatrixSrcSinglePrec;
    dftfe::utils::MemoryStorage<dataTypes::numberFP32, memorySpace>
      d_cellWaveFunctionMatrixDstSinglePrec;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_pseudopotentialNonLocalProjectorTimesVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
      d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_tempBlockVectorOverlapInvX;
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
      d_tempBlockVectorOverlapInvXSinglePrec;


    dftfe::utils::MemoryStorage<double, memorySpace> d_VeffJxW;
    dftfe::utils::MemoryStorage<double, memorySpace> d_VeffExtPotJxW;

    dftfe::utils::MemoryStorage<double, memorySpace>
      d_invJacderExcWithSigmaTimesGradRhoJxW;
    dftfe::utils::MemoryStorage<double, memorySpace>
      d_invJacinvJacderExcWithTauJxW;
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      d_invJacKPointTimesJxW;
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      d_halfKSquareTimesDerExcwithTauJxW;
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      d_derExcwithTauTimesinvJacKpointTimesJxW;
    // Constraints scaled with inverse sqrt diagonal Mass Matrix
    std::shared_ptr<dftUtils::constraintMatrixInfo<memorySpace>>
      inverseMassVectorScaledConstraintsNoneDataInfoPtr;
    std::shared_ptr<dftUtils::constraintMatrixInfo<memorySpace>>
      inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr;
    // kPoint cartesian coordinates
    std::vector<double> d_kPointCoordinates;
    // k point weights
    std::vector<double> d_kPointWeights;

    dftfe::utils::MemoryStorage<double, memorySpace> tempHamMatrixRealBlock;
    dftfe::utils::MemoryStorage<double, memorySpace> tempHamMatrixImagBlock;

    const dftfe::uInt          d_densityQuadratureID;
    const dftfe::uInt          d_lpspQuadratureID;
    const dftfe::uInt          d_feOrderPlusOneQuadratureID;
    dftfe::uInt                d_kPointIndex;
    dftfe::uInt                d_spinIndex;
    dftfe::uInt                d_HamiltonianIndex;
    bool                       d_isExternalPotCorrHamiltonianComputed;
    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             d_mpiCommDomain;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dftfe::uInt                d_cellsBlockSizeHamiltonianConstruction;
    dftfe::uInt                d_cellsBlockSizeHX;
    dftfe::uInt                d_numVectorsInternal;
    dftfe::uInt                d_nOMPThreads;
    dealii::ConditionalOStream pcout;

    // compute-time logger
    dealii::TimerOutput computing_timer;

    std::shared_ptr<hubbard<dataTypes::number, memorySpace>> d_hubbardClassPtr;
    bool                                                     d_useHubbard;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_srcNonLocalTemp;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_dstNonLocalTemp;

    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
      d_srcNonLocalTempSinglePrec;
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
      d_dstNonLocalTempSinglePrec;

    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> d_mapNodeIdToProcId;
  };
} // namespace dftfe
#endif
