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
// @author  Kartick Ramakrishnan
//

#ifndef DFTFE_PSEUDOBASECLASS_H
#define DFTFE_PSEUDOBASECLASS_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <excManager.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  enum class CouplingType
  {
    HamiltonianEntries,
    OverlapEntries,
    inverseOverlapEntries
  };

  // pseudopotentialBaseClass contains the getter functions to obtain the
  // pseudoPotential coupling matrix, single precision coupling matrix,
  // non-local operator and non-local operator single precision.

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class pseudopotentialBaseClass
  {
  public:
    virtual ~pseudopotentialBaseClass(){};

    virtual const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(
      CouplingType couplingtype = CouplingType::HamiltonianEntries) = 0;

    virtual const dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &
    getCouplingMatrixSinglePrec(
      CouplingType couplingtype = CouplingType::HamiltonianEntries) = 0;

    virtual const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator() = 0;

    virtual const std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
    getNonLocalOperatorSinglePrec();



  }; // end of class
} // end of namespace dftfe
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
