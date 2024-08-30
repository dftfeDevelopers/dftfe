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

#include <ExcSSDFunctionalBaseClass.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  ExcSSDFunctionalBaseClass<memorySpace>::ExcSSDFunctionalBaseClass(
    const ExcFamilyType     excFamType,
    const densityFamilyType densityFamType,
    const std::vector<DensityDescriptorDataAttributes>
      &densityDescriptorAttributesList)
    : d_ExcFamilyType(excFamType)
    , d_densityFamilyType(densityFamType)
    , d_densityDescriptorAttributesList(densityDescriptorAttributesList)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  ExcSSDFunctionalBaseClass<memorySpace>::~ExcSSDFunctionalBaseClass()
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  ExcFamilyType
  ExcSSDFunctionalBaseClass<memorySpace>::getExcFamilyType() const
  {
    return d_ExcFamilyType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  densityFamilyType
  ExcSSDFunctionalBaseClass<memorySpace>::getDensityBasedFamilyType() const
  {
    return d_densityFamilyType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<DensityDescriptorDataAttributes> &
  ExcSSDFunctionalBaseClass<memorySpace>::getDensityDescriptorAttributesList()
    const
  {
    return d_densityDescriptorAttributesList;
  }



} // namespace dftfe
