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
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#  include <excManagerDeviceKernels.h>
#endif
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
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
    initializeSSDPtr(std::string                                 XCType,
                     std::vector<std::shared_ptr<xc_func_type>> &funcXPtr,
                     std::vector<std::shared_ptr<xc_func_type>> &funcCPtr,
                     std::string        modelXCInputFile,
                     bool               printXCInfo,
                     const unsigned int numThreads)
    {
      dftfe::Int exceptParamX = -1, exceptParamC = -1;

      int vmajor, vminor, vmicro;
      xc_version(&vmajor, &vminor, &vmicro);
      if (printXCInfo)
        printf("Libxc version: %d.%d.%d\n", vmajor, vminor, vmicro);

      std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> excObj;
      if (XCType == "LDA-PZ")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX =
                xc_func_init(funcXPtr[iThread].get(), XC_LDA_X, XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_LDA_C_PZ,
                                          XC_POLARIZED);
            }
          excObj =
            std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "LDA-PW")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX =
                xc_func_init(funcXPtr[iThread].get(), XC_LDA_X, XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_LDA_C_PW,
                                          XC_POLARIZED);
            }
          excObj =
            std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "LDA-VWN")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX =
                xc_func_init(funcXPtr[iThread].get(), XC_LDA_X, XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_LDA_C_VWN,
                                          XC_POLARIZED);
            }
          excObj =
            std::make_shared<excDensityLDAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "GGA-PBE")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_GGA_X_PBE,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_GGA_C_PBE,
                                          XC_POLARIZED);
            }

          excObj =
            std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "GGA-RPBE")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_GGA_X_RPBE,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_GGA_C_PBE,
                                          XC_POLARIZED);
            }
          excObj =
            std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "GGA-LBxPBEc")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_GGA_X_LB,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_GGA_C_PBE,
                                          XC_POLARIZED);
            }

          excObj =
            std::make_shared<excDensityGGAClass<memorySpace>>(funcXPtr,
                                                              funcCPtr,
                                                              numThreads);
        }
      else if (XCType == "MLXC-NNLDA")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX =
                xc_func_init(funcXPtr[iThread].get(), XC_LDA_X, XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_LDA_C_PW,
                                          XC_POLARIZED);
            }
          excObj = std::make_shared<excDensityLDAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, numThreads);
        }
      else if (XCType == "MLXC-NNGGA")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_GGA_X_PBE,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_GGA_C_PBE,
                                          XC_POLARIZED);
            }
          excObj = std::make_shared<excDensityGGAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, numThreads);
        }
      else if (XCType == "MLXC-NNLLMGGA")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_GGA_X_PBE,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_GGA_C_PBE,
                                          XC_POLARIZED);
            }
          excObj = std::make_shared<excDensityLLMGGAClass<memorySpace>>(
            funcXPtr, funcCPtr, modelXCInputFile, numThreads);
        }
      else if (XCType == "MGGA-SCAN")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_MGGA_X_SCAN,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_MGGA_C_SCAN,
                                          XC_POLARIZED);
            }
          excObj = std::make_shared<excTauMGGAClass<memorySpace>>(funcXPtr,
                                                                  funcCPtr,
                                                                  numThreads);
        }
      else if (XCType == "MGGA-R2SCAN")
        {
          for (dftfe::Int iThread = 0; iThread < numThreads; iThread++)
            {
              exceptParamX = xc_func_init(funcXPtr[iThread].get(),
                                          XC_MGGA_X_R2SCAN,
                                          XC_POLARIZED);
              exceptParamC = xc_func_init(funcCPtr[iThread].get(),
                                          XC_MGGA_C_R2SCAN,
                                          XC_POLARIZED);
            }
          excObj = std::make_shared<excTauMGGAClass<memorySpace>>(funcXPtr,
                                                                  funcCPtr,
                                                                  numThreads);
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
            if (funcXPtr[0]->info->refs[i] != NULL)
              printf("X Functional: %s (DOI %s)\n",
                     funcXPtr[0]->info->refs[i]->ref,
                     funcXPtr[0]->info->refs[i]->doi);

          for (int i = 0; i < 1; i++)
            if (funcCPtr[0]->info->refs[i] != NULL)
              printf("C Functional: %s (DOI %s)\n",
                     funcCPtr[0]->info->refs[i]->ref,
                     funcCPtr[0]->info->refs[i]->doi);
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
      const double tauThreshold)
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
          tauVector[2 * iQuad + 0] = std::max(tauSpinUp[iQuad], tauThreshold);
          tauVector[2 * iQuad + 1] = std::max(tauSpinDown[iQuad], tauThreshold);
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
                                const bool  printXCInfo)
  {
    clear();
    d_numThreads = 1;
#ifdef _OPENMP
    if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
      {
        try
          {
            d_numThreads = std::stoi(std::string(penv));
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                std::string(
                  "When specifying the <DFTFE_NUM_THREADS> environment "
                  "variable, it needs to be something that can be interpreted "
                  "as an integer. The text you have in the environment "
                  "variable is <") +
                penv + ">"));
          }

        AssertThrow(d_numThreads > 0,
                    dealii::ExcMessage(
                      "When specifying the <DFTFE_NUM_THREADS> environment "
                      "variable, it needs to be a positive number."));
      }
#endif
    for (dftfe::Int iThread = 0; iThread < d_numThreads; iThread++)
      {
        d_funcXPtr.push_back(std::make_shared<xc_func_type>());
        d_funcCPtr.push_back(std::make_shared<xc_func_type>());
      }

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
            initializeSSDPtr<memorySpace>(
              XCInput, d_funcXPtr, d_funcCPtr, modelXCInputFile, printXCInfo,d_numThreads),
            numSpin);
      }
    else
      {
        d_excObj = initializeSSDPtr<memorySpace>(XCType,
                                                 d_funcXPtr,
                                                 d_funcCPtr,
                                                 modelXCInputFile,
                                                 printXCInfo,
                                                 d_numThreads);
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
