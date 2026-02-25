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
// @author Vishal Subramanian
//

#include <excManager.h>
#include <excDensityGGAClass.h>
#include <excDensityLDAClass.h>
#include <excDensityLLMGGAClass.h>
#include <excTauMGGAClass.h>
#include "ExcDFTPlusU.h"
#include <excManagerKernels.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#endif
namespace dftfe
{
  namespace
  {
    std::string
    lastN(std::string input, dftfe::uInt n)
    {
      if (input.size() < n)
        return input;

      return input.substr(input.size() - n);
    }

    template <dftfe::utils::MemorySpace memorySpace>
    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>>
    initializeSSDPtr(std::string                   XCType,
                     std::shared_ptr<xc_func_type> funcXPtr,
                     std::shared_ptr<xc_func_type> funcCPtr,
                     std::string                   modelXCInputFile,
                     bool                          printXCInfo,
                     const bool                    useLibxc)
    {
      dftfe::Int exceptParamX = -1, exceptParamC = -1;

      int vmajor, vminor, vmicro;
      xc_version(&vmajor, &vminor, &vmicro);
      if (printXCInfo && useLibxc)
        printf("Libxc version: %d.%d.%d\n", vmajor, vminor, vmicro);

      std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> excObj;
      if (XCType == "LDA-PZ")
        {
          exceptParamX = xc_func_init(funcXPtr.get(), XC_LDA_X, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_LDA_C_PZ, XC_POLARIZED);
          excObj = std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "LDA-PW")
        {
          exceptParamX = xc_func_init(funcXPtr.get(), XC_LDA_X, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_LDA_C_PW, XC_POLARIZED);

          excObj = std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "LDA-VWN")
        {
          exceptParamX = xc_func_init(funcXPtr.get(), XC_LDA_X, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_LDA_C_VWN, XC_POLARIZED);
          excObj = std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "GGA-PBE")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_PBE, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);

          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "GGA-RPBE")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_RPBE, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);

          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "GGA-REVPBE")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_PBE_R, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);

          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "GGA-PBESOL")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_PBE_SOL, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE_SOL, XC_POLARIZED);

          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "GGA-LBxPBEc")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_LB, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);

          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                                     funcCPtr,
                                                                     useLibxc,
                                                                     XCType);
        }
      else if (XCType == "MLXC-NNLDA")
        {
          exceptParamX = xc_func_init(funcXPtr.get(), XC_LDA_X, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_LDA_C_PW, XC_POLARIZED);
          excObj = std::make_shared<excDensityLDAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, useLibxc, XCType);
        }
      else if (XCType == "MLXC-NNGGA")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_PBE, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);
          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, useLibxc, XCType);
        }
      else if (XCType == "MLXC-NNLLMGGA")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_GGA_X_PBE, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_GGA_C_PBE, XC_POLARIZED);
          excObj = std::make_shared<excDensityLLMGGAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, useLibxc);
        }

      else if (XCType == "MGGA-SCAN")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_MGGA_X_SCAN, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_MGGA_C_SCAN, XC_POLARIZED);
          excObj = std::make_shared<excTauMGGAClass<memorySpace>>(funcXPtr,
                                                                  funcCPtr,
                                                                  useLibxc,
                                                                  XCType);
        }
      else if (XCType == "MGGA-R2SCAN")
        {
          exceptParamX =
            xc_func_init(funcXPtr.get(), XC_MGGA_X_R2SCAN, XC_POLARIZED);
          exceptParamC =
            xc_func_init(funcCPtr.get(), XC_MGGA_C_R2SCAN, XC_POLARIZED);
          excObj = std::make_shared<excTauMGGAClass<memorySpace>>(funcXPtr,
                                                                  funcCPtr,
                                                                  useLibxc,
                                                                  XCType);
        }
      else
        {
          std::cout << "Error in xc code \n";
          if (exceptParamX != 0 || exceptParamC != 0)
            {
              std::cout << "-------------------------------------" << std::endl;
              std::cout << "Exchange or Correlation Functional not found"
                        << std::endl;
              std::cout << "-------------------------------------" << std::endl;
              exit(-1);
            }
        }

      if (printXCInfo)
        {
          for (int i = 0; i < 1; i++)
            if (funcXPtr->info->refs[i] != NULL)
              printf("X Functional: %s (DOI %s)\n",
                     funcXPtr->info->refs[i]->ref,
                     funcXPtr->info->refs[i]->doi);

          for (int i = 0; i < 1; i++)
            if (funcCPtr->info->refs[i] != NULL)
              printf("C Functional: %s (DOI %s)\n",
                     funcCPtr->info->refs[i]->ref,
                     funcCPtr->info->refs[i]->doi);
        }

      return excObj;
    }
  } // namespace

  namespace internal
  {
    template <>
    void
    fillRhoVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoVector)
    {
      for (dftfe::uInt iQuad = 0; iQuad < numQuadPoints; iQuad++)
        {
          rhoVector[2 * iQuad + 0] = densitySpinUp[iQuad];
          rhoVector[2 * iQuad + 1] = densitySpinDown[iQuad];
        }
    }

    template <>
    void
    fillRhoSigmaVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinDown,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradDensitySpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &sigmaVector)
    {
      for (dftfe::uInt iQuad = 0; iQuad < numQuadPoints; iQuad++)
        {
          rhoVector[2 * iQuad + 0] = densitySpinUp[iQuad];
          rhoVector[2 * iQuad + 1] = densitySpinDown[iQuad];
          for (dftfe::uInt j = 0; j < 3; j++)
            {
              sigmaVector[3 * iQuad + 0] += gradDensitySpinUp[3 * iQuad + j] *
                                            gradDensitySpinUp[3 * iQuad + j];
              sigmaVector[3 * iQuad + 1] += gradDensitySpinUp[3 * iQuad + j] *
                                            gradDensitySpinDown[3 * iQuad + j];
              sigmaVector[3 * iQuad + 2] += gradDensitySpinDown[3 * iQuad + j] *
                                            gradDensitySpinDown[3 * iQuad + j];
            }
        }
    }

    template <>
    void
    fillRhoSigmaTauVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densitySpinDown,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradDensitySpinDown,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tauSpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tauSpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &sigmaVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                  &tauVector,
      const double rhoThreshold,
      const double sigmaThreshold,
      const double tauThreshold)
    {
      for (dftfe::uInt iQuad = 0; iQuad < numQuadPoints; iQuad++)
        {
          rhoVector[2 * iQuad + 0] =
            std::max(std::abs(densitySpinUp[iQuad]), rhoThreshold);
          rhoVector[2 * iQuad + 1] =
            std::max(std::abs(densitySpinDown[iQuad]), rhoThreshold);
          for (dftfe::uInt j = 0; j < 3; j++)
            {
              sigmaVector[3 * iQuad + 0] += gradDensitySpinUp[3 * iQuad + j] *
                                            gradDensitySpinUp[3 * iQuad + j];
              sigmaVector[3 * iQuad + 1] += gradDensitySpinUp[3 * iQuad + j] *
                                            gradDensitySpinDown[3 * iQuad + j];
              sigmaVector[3 * iQuad + 2] += gradDensitySpinDown[3 * iQuad + j] *
                                            gradDensitySpinDown[3 * iQuad + j];
            }
          sigmaVector[3 * iQuad + 0] =
            std::max(std::abs(sigmaVector[3 * iQuad + 0]), sigmaThreshold);
          sigmaVector[3 * iQuad + 2] =
            std::max(std::abs(sigmaVector[3 * iQuad + 2]), sigmaThreshold);
          tauVector[2 * iQuad + 0] =
            std::max(std::abs(tauSpinUp[iQuad]), tauThreshold);
          tauVector[2 * iQuad + 1] =
            std::max(std::abs(tauSpinDown[iQuad]), tauThreshold);
        }
    }

  }; // namespace internal
  template <dftfe::utils::MemorySpace memorySpace>
  excManager<memorySpace>::excManager()
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  excManager<memorySpace>::~excManager()
  {
    //    clear();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excManager<memorySpace>::clear()
  {
    //    d_excObj.reset();
    //    if (d_funcXPtr.get() != nullptr)
    //      {
    //        xc_func_end(d_funcXPtr.get());
    //      }
    //
    //    if (d_funcCPtr.get() != nullptr)
    //      {
    //        xc_func_end(d_funcCPtr.get());
    //      }
    //
    //    d_funcXPtr.reset();
    //    d_funcCPtr.reset();
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excManager<memorySpace>::init(std::string XCType,
                                bool        isSpinPolarized,
                                std::string modelXCInputFile,
                                const bool  printXCInfo,
                                const bool  useLibxc)
  {
    clear();

    d_funcXPtr = std::make_shared<xc_func_type>();
    d_funcCPtr = std::make_shared<xc_func_type>();

    bool enableHubbard = false;

    if (lastN(XCType, 2) == "+U")
      {
        enableHubbard = true;
      }

    if (enableHubbard)
      {
        dftfe::uInt numSpin = 1;
        if (isSpinPolarized == true)
          numSpin = 2;

        std::string XCInput = "";
        if (XCType.size() > 2)
          XCInput = XCType.substr(0, XCType.size() - 2);
        d_excObj =
          std::make_shared<ExcDFTPlusU<dataTypes::number, memorySpace>>(
            initializeSSDPtr<memorySpace>(XCInput,
                                          d_funcXPtr,
                                          d_funcCPtr,
                                          modelXCInputFile,
                                          printXCInfo,
                                          useLibxc),
            numSpin);
      }
    else
      {
        d_excObj = initializeSSDPtr<memorySpace>(XCType,
                                                 d_funcXPtr,
                                                 d_funcCPtr,
                                                 modelXCInputFile,
                                                 printXCInfo,
                                                 useLibxc);
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  ExcSSDFunctionalBaseClass<memorySpace> *
  excManager<memorySpace>::getExcSSDFunctionalObj()
  {
    return d_excObj.get();
  }


  template <dftfe::utils::MemorySpace memorySpace>
  const ExcSSDFunctionalBaseClass<memorySpace> *
  excManager<memorySpace>::getExcSSDFunctionalObj() const
  {
    return d_excObj.get();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> &
  excManager<memorySpace>::getSSDSharedObj() const
  {
    return d_excObj;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> &
  excManager<memorySpace>::getSSDSharedObj()
  {
    return d_excObj;
  }


  template class excManager<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excManager<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
