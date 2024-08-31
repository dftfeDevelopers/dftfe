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
// @author Vishal Subramanian
//

#include "excDensityGGAClass.h"
#include "excDensityLDAClass.h"
#include "excDensityLLMGGAClass.h"
#include "ExcDFTPlusU.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<memorySpace>::ExcDFTPlusU(
    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> excSSDObjPtr,
    unsigned int                                            numSpins)
    : ExcSSDFunctionalBaseClass<memorySpace>(*(excSSDObjPtr.get()))
  {
    d_excSSDObjPtr = excSSDObjPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<memorySpace>::~ExcDFTPlusU()
  {}


  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::computeOutputXCData(
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    quadPoints,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &xDataOut,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &cDataOut) const
  {
    d_excSSDObjPtr->computeOutputXCData(auxDensityMatrix,
                                        quadPoints,
                                        xDataOut,
                                        cDataOut);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    d_excSSDObjPtr->checkInputOutputDataAttributesConsistency(
      outputDataAttributes);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::applyWaveFunctionDependentFuncDer(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      &                                                                src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const unsigned int inputVecSize,
    const double       factor,
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::updateWaveFunctionDependentFuncDer(
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    kPointWeights)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }
  template <dftfe::utils::MemorySpace memorySpace>
  double
  ExcDFTPlusU<memorySpace>::computeWaveFunctionDependentExcEnergy(
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    kPointWeights)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template class ExcDFTPlusU<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class ExcDFTPlusU<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
