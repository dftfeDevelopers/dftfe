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


#ifndef kohnShamHamiltonianOperatorClass_H_
#define kohnShamHamiltonianOperatorClass_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <headers.h>
#include <operator.h>
#include <BLASWrapper.h>
#include <FEBasisOperations.h>
#include <oncvClass.h>
#include <AuxDensityMatrix.h>

#include "hubbardClass.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class KohnShamHamiltonianOperator : public operatorDFTClass<memorySpace>
  {
  public:
    KohnShamHamiltonianOperator(
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
      std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                               oncvClassPtr,
      std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      dftParameters *                          dftParamsPtr,
      const unsigned int                       densityQuadratureID,
      const unsigned int                       lpspQuadratureID,
      const unsigned int                       feOrderPlusOneQuadratureID,
      const MPI_Comm &                         mpi_comm_parent,
      const MPI_Comm &                         mpi_comm_domain);

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
    getScratchFEMultivector(const unsigned int numVectors,
                            const unsigned int index);


    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
    getScratchFEMultivectorSinglePrec(const unsigned int numVectors,
                                      const unsigned int index);


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
        &                phiValues,
      const unsigned int spinIndex = 0);

    /**
     * @brief Sets the V-eff potential
     * @param vKS_quadValues the input V-KS values stored at the quadrature points
     * @param spinIndex spin index
     */
    void
    setVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                vKS_quadValues,
      const unsigned int spinIndex);

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
        &                phiPrimeValues,
      const unsigned int spinIndex);

    /**
     * @brief sets the data member to appropriate kPoint and spin Index
     *
     * @param kPointIndex  k-point Index to set
     */
    void
    reinitkPointSpinIndex(const unsigned int kPointIndex,
                          const unsigned int spinIndex);

    void
    reinitNumberWavefunctions(const unsigned int numWfc);

    const dftfe::utils::MemoryStorage<double, memorySpace> &
    getInverseSqrtMassVector();

    const dftfe::utils::MemoryStorage<double, memorySpace> &
    getSqrtMassVector();

    void
    computeCellHamiltonianMatrix(
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    computeCellHamiltonianMatrixExtPotContribution();

    void
    HX(dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
       const double scalarHX,
       const double scalarY,
       const double scalarX,
       dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

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

    void
    HXRR(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dstHX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dstMX,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

  private:
    void
    setVEffExternalPotCorrToZero();

    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
      d_ONCVnonLocalOperator;


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
      d_ONCVnonLocalOperatorSinglePrec;

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
    std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                             d_oncvClassPtr;
    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr;
    dftParameters *                          d_dftParamsPtr;

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
      d_ONCVNonLocalProjectorTimesVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                                                     d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec;
    dftfe::utils::MemoryStorage<double, memorySpace> d_VeffJxW;
    dftfe::utils::MemoryStorage<double, memorySpace> d_VeffExtPotJxW;

    dftfe::utils::MemoryStorage<double, memorySpace>
      d_invJacderExcWithSigmaTimesGradRhoJxW;
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      d_invJacKPointTimesJxW;
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

    const unsigned int         d_densityQuadratureID;
    const unsigned int         d_lpspQuadratureID;
    const unsigned int         d_feOrderPlusOneQuadratureID;
    unsigned int               d_kPointIndex;
    unsigned int               d_spinIndex;
    unsigned int               d_HamiltonianIndex;
    bool                       d_isExternalPotCorrHamiltonianComputed;
    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             d_mpiCommDomain;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    unsigned int               d_cellsBlockSizeHamiltonianConstruction;
    unsigned int               d_cellsBlockSizeHX;
    unsigned int               d_numVectorsInternal;
    unsigned int               d_nOMPThreads;
    dealii::ConditionalOStream pcout;

    // compute-time logger
    dealii::TimerOutput computing_timer;

    std::shared_ptr<hubbard<dataTypes::number, memorySpace>> d_hubbardClassPtr;
    bool                                                     d_useHubbard;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_srcNonLocalTemp;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      d_dstNonLocalTemp;

    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapNodeIdToProcId;
  };
} // namespace dftfe
#endif
