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
  void
  excManager::createExcClassObj(unsigned int               xc_id,
                                bool                       isSpinPolarized,
                                unsigned int               exxFactor,
                                bool                       scaleExchange,
                                unsigned int               scaleExchangeFactor,
                                bool                       computeCorrelation,
                                xc_func_type *             funcXPtr,
                                xc_func_type *             funcCPtr,
                                excWavefunctionBaseClass *&excClassPtr)
  {
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
          exceptParamX = xc_func_init(funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC = xc_func_init(funcCPtr, XC_LDA_C_PZ, isSpinPolarizedXC);
          excClassPtr  = new excWavefunctionNoneClass(densityFamilyType::LDA,
                                                     funcXPtr,
                                                     funcCPtr,
                                                     0.0,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 2:
          exceptParamX = xc_func_init(funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC = xc_func_init(funcCPtr, XC_LDA_C_PW, isSpinPolarizedXC);
          excClassPtr  = new excWavefunctionNoneClass(densityFamilyType::LDA,
                                                     funcXPtr,
                                                     funcCPtr,
                                                     0.0,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 3:
          exceptParamX = xc_func_init(funcXPtr, XC_LDA_X, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(funcCPtr, XC_LDA_C_VWN, isSpinPolarizedXC);
          excClassPtr = new excWavefunctionNoneClass(densityFamilyType::LDA,
                                                     funcXPtr,
                                                     funcCPtr,
                                                     0.0,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 4:
          exceptParamX =
            xc_func_init(funcXPtr, XC_GGA_X_PBE, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(funcCPtr, XC_GGA_C_PBE, isSpinPolarizedXC);
          excClassPtr = new excWavefunctionNoneClass(densityFamilyType::GGA,
                                                     funcXPtr,
                                                     funcCPtr,
                                                     0.0,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
          break;
        case 5:
          exceptParamX =
            xc_func_init(funcXPtr, XC_GGA_X_RPBE, isSpinPolarizedXC);
          exceptParamC =
            xc_func_init(funcCPtr, XC_GGA_C_PBE, isSpinPolarizedXC);
          excClassPtr = new excWavefunctionNoneClass(densityFamilyType::GGA,
                                                     funcXPtr,
                                                     funcCPtr,
                                                     0.0,
                                                     scaleExchange,
                                                     computeCorrelation,
                                                     scaleExchangeFactor);
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
} // namespace dftfe
