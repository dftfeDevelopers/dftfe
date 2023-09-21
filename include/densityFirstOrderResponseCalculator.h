// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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


#ifndef densityFirstOrderResponseCalculator_H_
#define densityFirstOrderResponseCalculator_H_

#include "headers.h"
#include "operator.h"
#include "dftParameters.h"

#if defined(DFTFE_WITH_DEVICE)
#  include "operatorDevice.h"
#  include "dftfeDataTypes.h"
#endif

namespace dftfe
{
  template <typename NumberType>
  void
  computeRhoFirstOrderResponseCPU(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);

  template <typename NumberType, typename NumberTypeLowPrec>
  void
  computeRhoFirstOrderResponseCPUMixedPrec(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);


#if defined(DFTFE_WITH_DEVICE)
  template <typename NumberType, typename NumberTypeLowPrec>
  void
  computeRhoFirstOrderResponseDevice(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
#endif
} // namespace dftfe
#endif
