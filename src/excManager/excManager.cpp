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

#include <excManager.h>
#include <excWavefunctionNoneClass.h>
#include <excDensityGGAClass.h>
#include <excDensityLDAClass.h>

namespace dftfe
{
  excManager::excManager()
  {
    d_funcXPtr              = nullptr;
    d_funcCPtr              = nullptr;
    d_excDensityObjPtr      = nullptr;
    d_excWavefunctionObjPtr = nullptr;
  }

  excManager::~excManager()
  {
    clear();
  }

  void
  excManager::clear()
  {
    if (d_funcXPtr != nullptr)
      {
        xc_func_end(d_funcXPtr);
        delete d_funcXPtr;
      }

    if (d_funcCPtr != nullptr)
      {
        xc_func_end(d_funcCPtr);
        delete d_funcCPtr;
      }

    if (d_excDensityObjPtr != nullptr)
      delete d_excDensityObjPtr;

    if (d_excWavefunctionObjPtr != nullptr)
      delete d_excWavefunctionObjPtr;

    d_funcXPtr              = nullptr;
    d_funcCPtr              = nullptr;
    d_excDensityObjPtr      = nullptr;
    d_excWavefunctionObjPtr = nullptr;
  }



  void
  excManager::init(unsigned int xc_id,
                   bool         isSpinPolarized,
                   unsigned int exxFactor,
                   bool         scaleExchange,
                   unsigned int scaleExchangeFactor,
                   bool         computeCorrelation,
                   std::string  modelXCInputFile)
  {
    clear();

    d_funcXPtr = new xc_func_type;
    d_funcCPtr = new xc_func_type;


    int exceptParamX = -1, exceptParamC = -1;
    int isSpinPolarizedXC;
    if (isSpinPolarized)
      {
        isSpinPolarizedXC = XC_POLARIZED;
      }
    else
      {
        isSpinPolarizedXC = XC_UNPOLARIZED;
      }


    switch (xc_id)
      {
        case 1:
          exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_LDA_C_PZ, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityLDAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 2:
          exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_LDA_C_PW, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityLDAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 3:
          exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_LDA_C_VWN, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityLDAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);
          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 4:
          exceptParamX =
            xc_func_init(d_funcXPtr, XC_GGA_X_PBE, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_GGA_C_PBE, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityGGAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 5:
          exceptParamX =
            xc_func_init(d_funcXPtr, XC_GGA_X_RPBE, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_GGA_C_PBE, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityGGAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 6:
          exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_LDA_C_PW, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityLDAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      modelXCInputFile,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        case 7:
          exceptParamX =
            xc_func_init(d_funcXPtr, XC_GGA_X_PBE, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(d_funcCPtr, XC_GGA_C_PBE, isSpinPolarizedXC);
          d_excDensityObjPtr = new excDensityGGAClass(d_funcXPtr,
                                                      d_funcCPtr,
                                                      isSpinPolarized,
                                                      modelXCInputFile,
                                                      scaleExchange,
                                                      computeCorrelation,
                                                      scaleExchangeFactor);

          d_excWavefunctionObjPtr =
            new excWavefunctionNoneClass(isSpinPolarized);
          break;
        default:
          std::cout << "Error in xc code \n";
          break;

          if (exceptParamX != 0 || exceptParamC != 0)
            {
              std::cout << "-------------------------------------" << std::endl;
              std::cout << "Exchange or Correlation Functional not found"
                        << std::endl;
              std::cout << "-------------------------------------" << std::endl;
              exit(-1);
            }
      }
  }

  densityFamilyType
  excManager::getDensityBasedFamilyType() const
  {
    return d_excDensityObjPtr->getDensityBasedFamilyType();
  }

  wavefunctionFamilyType
  excManager::getWavefunctionBasedFamilyType() const
  {
    return d_excWavefunctionObjPtr->getWavefunctionBasedFamilyType();
  }


  excDensityBaseClass *
  excManager::getExcDensityObj()
  {
    return d_excDensityObjPtr;
  }

  excWavefunctionBaseClass *
  excManager::getExcWavefunctionObj()
  {
    return d_excWavefunctionObjPtr;
  }


  const excDensityBaseClass *
  excManager::getExcDensityObj() const
  {
    return d_excDensityObjPtr;
  }

  const excWavefunctionBaseClass *
  excManager::getExcWavefunctionObj() const
  {
    return d_excWavefunctionObjPtr;
  }


} // namespace dftfe
