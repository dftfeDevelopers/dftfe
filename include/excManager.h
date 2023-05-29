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

#ifndef DFTFE_EXCMANAGER_H
#define DFTFE_EXCMANAGER_H

#include <xc.h>
#include <excDensityBaseClass.h>
#include <excWavefunctionBaseClass.h>
namespace dftfe
{
  class excManager
  {
  public:
    /**
     * @brief Constructor
     *
     */
    excManager();

    /**
     * @brief  destructor
     */
    ~excManager();

    void
    clear();


    void
    init(unsigned int xc_id,
         bool         isSpinPolarized,
         unsigned int exxFactor,
         bool         scaleExchange,
         unsigned int scaleExchangeFactor,
         bool         computeCorrelation,
         std::string  modelXCInputFile);

    densityFamilyType
    getDensityBasedFamilyType() const;

    wavefunctionFamilyType
    getWavefunctionBasedFamilyType() const;


    excDensityBaseClass *
    getExcDensityObj();

    excWavefunctionBaseClass *
    getExcWavefunctionObj();


    const excDensityBaseClass *
    getExcDensityObj() const;

    const excWavefunctionBaseClass *
    getExcWavefunctionObj() const;


  private:
    /// objects for various exchange-correlations (from libxc package)
    xc_func_type *d_funcXPtr;
    xc_func_type *d_funcCPtr;

    excDensityBaseClass *     d_excDensityObjPtr;
    excWavefunctionBaseClass *d_excWavefunctionObjPtr;
  };
} // namespace dftfe

#endif // DFTFE_EXCMANAGER_H
