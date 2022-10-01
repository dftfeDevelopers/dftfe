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

namespace dftfe
{
  void excManager::createExcClassObj(unsigned int xc_id,
                                           bool isSpinPolarized,
                                           unsigned int exxFactor,
                                           bool scaleExchange,
                                           unsigned int scaleExchangeFactor
                                           bool computeCorrelation,
                                excWavefunctionBaseClass *excClassPtr)
  {
    xc_func_type funcX, funcC;
    int exceptParamX, exceptParamC;
    switch (xc_id) {
        case 1 :
          exceptParamX = xc_func_init(&funcX, XC_LDA_X, isSpinPolarized);
          exceptParamC = xc_func_init(&funcC, XC_LDA_C_PZ, isSpinPolarized);
          excClassPtr = new excWavefunctionNoneClass (densityFamilyType::LDA,
                                                     funcX,
                                                     funcC,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break ;
        case 2 :
          exceptParamX = xc_func_init(&funcX, XC_LDA_X, isSpinPolarized);
          exceptParamC = xc_func_init(&funcC, XC_LDA_C_PW, isSpinPolarized);
          excClassPtr = new excWavefunctionNoneClass (densityFamilyType::LDA,
                                                     funcX,
                                                     funcC,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 3 :
          exceptParamX = xc_func_init(&funcX, XC_LDA_X, isSpinPolarized);
          exceptParamC = xc_func_init(&funcC, XC_LDA_C_VWN, isSpinPolarized);
          excClassPtr = new excWavefunctionNoneClass (densityFamilyType::LDA,
                                                     funcX,
                                                     funcC,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 4 :
          exceptParamX = xc_func_init(&funcX, XC_GGA_X_PBE, isSpinPolarized);
          exceptParamC = xc_func_init(&funcC, XC_GGA_C_PBE, isSpinPolarized);
          excClassPtr = new excWavefunctionNoneClass (densityFamilyType::GGA,
                                                     funcX,
                                                     funcC,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 5 :
          exceptParamX = xc_func_init(&funcX, XC_GGA_X_RPBE, isSpinPolarized);
          exceptParamC = xc_func_init(&funcC, XC_GGA_C_PBE, isSpinPolarized);
          excClassPtr = new excWavefunctionNoneClass (densityFamilyType::GGA,
                                                     funcX,
                                                     funcC,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case default :
          std::cout<< "Error in xc code \n";
          break;
      }

  }
}