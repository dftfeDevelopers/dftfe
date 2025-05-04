// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXE_EXCDFTPLUSU_H
#define DFTFE_EXE_EXCDFTPLUSU_H



#include "ExcSSDFunctionalBaseClass.h"
#include "hubbardClass.h"
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class ExcDFTPlusU : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    ExcDFTPlusU(
      std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> excSSDObjPtr,
      dftfe::uInt                                             numSpins);

    ~ExcDFTPlusU();

    void
    applyWaveFunctionDependentFuncDerWrtPsi(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                        &src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex) override;

    /*
     * @brief The apply function that will be called in HXCheby() with single precision.
     * The distribute() and updateGhostValues() for src
     * has to be called before this function.
     * Similarly for dst, accumulateLocallyOwned() should be called in HX()
     * after this function is called. param[in] src The input vector param[out]
     * dst The output vector param[in] inputVecSize The size of the input vector
     * param[in] kPointIndex the k point for which the HX() is called
     * param[in] spinIndex the spin index for which the HX() is called
     */
    void
    applyWaveFunctionDependentFuncDerWrtPsi(
      const dftfe::linearAlgebra::MultiVector<
        typename dataTypes::singlePrecType<ValueType>::type,
        memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<
        typename dataTypes::singlePrecType<ValueType>::type,
        memorySpace>   &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex) override;

    void
    updateWaveFunctionDependentFuncDerWrtPsi(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
      const std::vector<double> &kPointWeights) override;

    void
    computeWaveFunctionDependentExcEnergy(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
      const std::vector<double> &kPointWeights) override;

    double
    getWaveFunctionDependentExcEnergy() override;

    double
    getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi() override;

    /**
     * x and c denotes exchange and correlation respectively
     */
    void
    computeRhoTauDependentXCData(
      AuxDensityMatrix<memorySpace>             &auxDensityMatrix,
      const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &xDataOut,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &cDataout) const override;
    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const override;

    void
    reinitKPointDependentVariables(dftfe::uInt kPointIndex) override;

    void
    initialiseHubbardClass(
      const MPI_Comm &mpi_comm_parent,
      const MPI_Comm &mpi_comm_domain,
      const MPI_Comm &mpi_comm_interPool,
      const MPI_Comm &mpi_comm_interBandGroup,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        basisOperationsMemPtr,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperMemPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                              BLASWrapperHostPtr,
      const dftfe::uInt                       matrixFreeVectorComponent,
      const dftfe::uInt                       densityQuadratureId,
      const dftfe::uInt                       sparsityPatternQuadratureId,
      const dftfe::uInt                       numberWaveFunctions,
      const dftfe::uInt                       numSpins,
      const dftParameters                    &dftParam,
      const std::string                      &scratchFolderName,
      const bool                              singlePrecNonLocalOperator,
      const bool                              updateNonlocalSparsity,
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &atomLocationsFrac,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      std::vector<double>                    &kPointCoordinates,
      const std::vector<double>              &kPointWeights,
      const std::vector<std::vector<double>> &domainBoundaries);

    std::shared_ptr<hubbard<ValueType, memorySpace>> &
    getHubbardClass();

  public:
    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> d_excSSDObjPtr;
    std::shared_ptr<hubbard<ValueType, memorySpace>>        d_hubbardClassPtr;
  };
} // namespace dftfe
#endif // DFTFE_EXE_EXCDFTPLUSU_H
