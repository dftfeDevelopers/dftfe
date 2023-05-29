// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXCWAVEFUNCTIONBASECLASS_H
#define DFTFE_EXCWAVEFUNCTIONBASECLASS_H

namespace dftfe
{
  enum class wavefunctionFamilyType
  {
    NONE,
    SCALED_FOCK,
    HUBBARD
  };

  class excWavefunctionBaseClass
  {
  public:
    excWavefunctionBaseClass(bool isSpinPolarized);

    virtual ~excWavefunctionBaseClass();

    virtual void
    applyWaveFunctionDependentVxc() const = 0;
    virtual void
    updateWaveFunctionDependentVxc() const = 0;
    virtual double
    computeWaveFunctionDependentExcEnergy() const = 0;

    wavefunctionFamilyType
    getWavefunctionBasedFamilyType() const;

  protected:
    wavefunctionFamilyType d_wavefunctionFamilyType;
    bool                   d_isSpinPolarized;
  };
} // namespace dftfe

#endif // DFTFE_EXCWAVEFUNCTIONBASECLASS_H
