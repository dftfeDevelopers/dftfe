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
// @author Srinibas Nandi, Vishal Subramanian, Sambit Das
//

#include "excTauMGGAClass.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>
#include <excManagerKernels.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#endif
#include <exchangeCorrelationFunctionalEvaluator.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::excTauMGGAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    const bool                     useLibxc)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::TauMGGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown},
        std::vector<WfcDescriptorDataAttributes>{
          WfcDescriptorDataAttributes::tauSpinUp,
          WfcDescriptorDataAttributes::tauSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_useLibxc = useLibxc;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::excTauMGGAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    std::string                    modelXCInputFile,
    const bool                     useLibxc)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::TauMGGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown},
        std::vector<WfcDescriptorDataAttributes>{
          WfcDescriptorDataAttributes::tauSpinUp,
          WfcDescriptorDataAttributes::tauSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_useLibxc = useLibxc;
  }
  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::~excTauMGGAClass()
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    const std::vector<xcRemainderOutputDataAttributes>
      allowedOutputDataAttributes = {
        xcRemainderOutputDataAttributes::e,
        xcRemainderOutputDataAttributes::pdeDensitySpinUp,
        xcRemainderOutputDataAttributes::pdeDensitySpinDown,
        xcRemainderOutputDataAttributes::pdeSigma,
        xcRemainderOutputDataAttributes::pdeTauSpinUp,
        xcRemainderOutputDataAttributes::pdeTauSpinDown};

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
          {
            if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
              isFound = true;
          }

        std::string errMsg =
          "xcRemainderOutputDataAttributes do not match with the allowed choices for the family type.";
        dftfe::utils::throwException(isFound, errMsg);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::computeRhoTauDependentXCData(
    AuxDensityMatrix<memorySpace>             &auxDensityMatrix,
    const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &xDataOut,
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &cDataOut) const
  {
    double tauThresholdMgga   = 1e-9;
    double rhoThresholdMgga   = 1e-9;
    double sigmaThresholdMgga = 1e-24;

    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;

    for (const auto &element : xDataOut)
      {
        outputDataAttributes.push_back(element.first);
      }

    checkInputOutputDataAttributesConsistency(outputDataAttributes);

    std::unordered_map<
      DensityDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityDescriptorData;
    std::unordered_map<
      WfcDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      wfcDescriptorData;

    for (size_t i = 0; i < this->d_densityDescriptorAttributesList.size(); i++)
      {
        if (this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinUp ||
            this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(nquad,
                                                                         0);
        else if (this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinUp ||
                 this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(
              3 * nquad, 0);
      }

    for (size_t i = 0; i < this->d_wfcDescriptorAttributesList.size(); i++)
      {
        if (this->d_wfcDescriptorAttributesList[i] ==
              WfcDescriptorDataAttributes::tauSpinUp ||
            this->d_wfcDescriptorAttributesList[i] ==
              WfcDescriptorDataAttributes::tauSpinDown)
          wfcDescriptorData[this->d_wfcDescriptorAttributesList[i]] =
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(nquad,
                                                                         0.0);
      }

    auxDensityMatrix.applyLocalOperations(quadIndexRange,
                                          densityDescriptorData);
    auxDensityMatrix.applyLocalOperations(quadIndexRange, wfcDescriptorData);


    auto &densityValuesSpinUp =
      densityDescriptorData.find(DensityDescriptorDataAttributes::valuesSpinUp)
        ->second;
    auto &densityValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::valuesSpinDown)
        ->second;
    auto &gradValuesSpinUp =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
        ->second;
    auto &gradValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinDown)
        ->second;
    auto &tauValuesSpinUp =
      wfcDescriptorData.find(WfcDescriptorDataAttributes::tauSpinUp)->second;
    auto &tauValuesSpinDown =
      wfcDescriptorData.find(WfcDescriptorDataAttributes::tauSpinDown)->second;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      densityValues(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      sigmaValues(3 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      tauValues(2 * nquad, 0);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      exValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      ecValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensityValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensityValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensitySpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensitySpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensitySpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensitySpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexSigmaValues(3 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecSigmaValues(3 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexTauValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecTauValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexTauSpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexTauSpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecTauSpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecTauSpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexLaplacianValues(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecLaplacianValues(2 * nquad, 0);

    dftfe::internal::fillRhoSigmaTauVector(nquad,
                                           densityValuesSpinUp,
                                           densityValuesSpinDown,
                                           gradValuesSpinUp,
                                           gradValuesSpinDown,
                                           tauValuesSpinUp,
                                           tauValuesSpinDown,
                                           densityValues,
                                           sigmaValues,
                                           tauValues,
                                           tauThresholdMgga);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      laplacianValues(2 * nquad, 0.0);

    if (d_useLibxc)
      {
        xc_mgga_exc_vxc(d_funcXPtr.get(),
                        nquad,
                        &densityValues[0],
                        &sigmaValues[0],
                        &laplacianValues[0],
                        &tauValues[0],
                        &exValues[0],
                        &pdexDensityValuesNonNN[0],
                        &pdexSigmaValues[0],
                        &pdexLaplacianValues[0],
                        &pdexTauValuesNonNN[0]);
        xc_mgga_exc_vxc(d_funcCPtr.get(),
                        nquad,
                        &densityValues[0],
                        &sigmaValues[0],
                        &laplacianValues[0],
                        &tauValues[0],
                        &ecValues[0],
                        &pdecDensityValuesNonNN[0],
                        &pdecSigmaValues[0],
                        &pdecLaplacianValues[0],
                        &pdecTauValuesNonNN[0]);
      }
    else
      {
        /*uncomment and modify the below part to get the parameters for  */
        
        // typedef struct
        // {
        //   double eta;
        // } mgga_c_r2scan_params;

        // mgga_x_scan_params *params;
        // mgga_c_r2scan_params *paramsC;

        // params = (mgga_x_scan_params *)d_funcXPtr->params;
        // paramsC = (mgga_c_r2scan_params *)d_funcCPtr->params;

        // std::cout << "c1: " << params->c1 << std::endl;
        // std::cout << "c2: " << params->c2 << std::endl;
        // std::cout << "d: " << params->d << std::endl;
        // std::cout << "k1: " << params->k1 << std::endl;

        // std::cout << "eta: " << paramsC->eta << std::endl;
        // std::cout << "dens_threshold: " << d_funcXPtr->dens_threshold
        //           << std::endl;
        // std::cout << "zeta_threshold: " << d_funcXPtr->zeta_threshold
        //           << std::endl;
        //         std::cout << "flags: " << d_funcXPtr->info->flags <<
        //         std::endl;
        // #ifndef XC_FLAGS_ENFORCE_FHC
        // #  define XC_FLAGS_NEEDS_TAU (1 << 16)
        // #  define XC_FLAGS_ENFORCE_FHC (1 << 17)
        // #endif
        //         std::cout << "XC_FLAGS_NEEDS_TAU: "
        //                   << (d_funcXPtr->info->flags & XC_FLAGS_NEEDS_TAU)
        //                   << std::endl;
        //         std::cout << "XC_FLAGS_ENFORCE_FHC: "
        //                   << (d_funcXPtr->info->flags & XC_FLAGS_ENFORCE_FHC)
        //                   << std::endl;
#if defined(DFTFE_WITH_DEVICE)
        dftfe::utils::MemoryStorage<double, memorySpace> densityValuesTemp,
          sigmaValuesTemp, tauValuesTemp;
        dftfe::utils::MemoryStorage<double, memorySpace> ecValuesTemp,
          exValuesTemp;
        dftfe::utils::MemoryStorage<double, memorySpace> pdecDensityTemp,
          pdexDensityTemp, pdecSigmaValuesTemp, pdexSigmaValuesTemp,
          pdexTauValuesTemp, pdecTauValuesTemp;

        densityValuesTemp.resize(densityValues.size());
        densityValuesTemp.copyFrom(densityValues);
        sigmaValuesTemp.resize(sigmaValues.size());
        sigmaValuesTemp.copyFrom(sigmaValues);
        tauValuesTemp.resize(tauValues.size());
        tauValuesTemp.copyFrom(tauValues);
        exValuesTemp.resize(exValues.size());
        ecValuesTemp.resize(ecValues.size());
        pdexDensityTemp.resize(pdexDensityValuesNonNN.size());
        pdecDensityTemp.resize(pdecDensityValuesNonNN.size());
        pdexSigmaValuesTemp.resize(pdexSigmaValues.size());
        pdecSigmaValuesTemp.resize(pdecSigmaValues.size());
        pdexTauValuesTemp.resize(pdexTauValuesNonNN.size());
        pdecTauValuesTemp.resize(pdecTauValuesNonNN.size());
#else
        auto &densityValuesTemp   = densityValues;
        auto &sigmaValuesTemp     = sigmaValues;
        auto &tauValuesTemp       = tauValues;
        auto &exValuesTemp        = exValues;
        auto &ecValuesTemp        = ecValues;
        auto &pdecDensityTemp     = pdecDensityValuesNonNN;
        auto &pdexDensityTemp     = pdexDensityValuesNonNN;
        auto &pdecSigmaValuesTemp = pdecSigmaValues;
        auto &pdexSigmaValuesTemp = pdexSigmaValues;
        auto &pdecTauValuesTemp   = pdecTauValuesNonNN;
        auto &pdexTauValuesTemp   = pdexTauValuesNonNN;
#endif
        if (d_funcXPtr->info->number == 497)
          {
            MGGAX_R2SCAN(nquad,
                         densityValuesTemp,
                         sigmaValuesTemp,
                         tauValuesTemp,
                         exValuesTemp,
                         pdexDensityTemp,
                         pdexSigmaValuesTemp,
                         pdexTauValuesTemp);
#if defined(DFTFE_WITH_DEVICE)
            exValues.copyFrom(exValuesTemp);
            pdexDensityValuesNonNN.copyFrom(pdexDensityTemp);
            pdexSigmaValues.copyFrom(pdexSigmaValuesTemp);
            pdexTauValuesNonNN.copyFrom(pdexTauValuesTemp);
#endif
          }
        if (d_funcCPtr->info->number == 498)
          {
            MGGAC_R2SCAN(nquad,
                         densityValuesTemp,
                         sigmaValuesTemp,
                         tauValuesTemp,
                         ecValuesTemp,
                         pdecDensityTemp,
                         pdecSigmaValuesTemp,
                         pdecTauValuesTemp);
#if defined(DFTFE_WITH_DEVICE)
            ecValues.copyFrom(ecValuesTemp);
            pdecDensityValuesNonNN.copyFrom(pdecDensityTemp);
            pdecSigmaValues.copyFrom(pdecSigmaValuesTemp);
            pdecTauValuesNonNN.copyFrom(pdecTauValuesTemp);
#endif
          }
        else
          {
            dftfe::utils::throwException(
              "xc_func_type name is not implemented in DFT-FE. Use LIBXC to compute the M-GGA functional.");
          }
        /////////////////////////////////////////////////////////////////////////
        // {
        //   double diff;
        //   for (int i = 0; i < nquad; i++)
        //     {
        // std::cout << "densityValues: " << densityValues[2 * i] <<
        // std::endl; std::cout << "sigmaValues: " << sigmaValues[3 * i] <<
        // std::endl; std::cout << "tauValues: " << tauValues[2 * i] <<
        // std::endl;
        // diff = ecValuesTemp[i] - ecValues[i];
        // if (std::abs(diff) > 1e-14)
        //   {
        //     std::cout << "alter exValues: " << ecValuesTemp[i] << std::endl;
        //     std::cout << "libxc exValues: " << ecValues[i] << std::endl;
        //     std::cout << "diff: " << diff << std::endl;
        //     std::cout << std::endl;
        //   }
        // diff = pdexDensityTemp[i] - pdexDensityValuesNonNN[i];
        // if (std::abs(diff) > 1e-14)
        //   {
        //     std::cout << "alter pdexrho: " << pdexDensityTemp[i]
        //               << std::endl;
        //     std::cout << "libxc pdexrho: " << pdexDensityValuesNonNN[i]
        //               << std::endl;
        //     std::cout << "diff: " << diff << std::endl;
        //     std::cout << std::endl;
        //   }
        // diff = pdexSigmaValuesTemp[i] - pdexSigmaValues[i];
        // if (std::abs(diff) > 1e-14)
        //   {
        //     std::cout << "alter pdexsigma: " << pdexSigmaValuesTemp[i]
        //               << std::endl;
        //     std::cout << "libxc pdexsigma: " << pdexSigmaValues[i]
        //               << std::endl;
        //     std::cout << "diff: " << diff << std::endl;
        //     std::cout << std::endl;
        //   }

        // diff = pdexTauValuesTemp[i] - pdexTauValuesNonNN[i];

        // if (std::abs(diff) > 1e-10)
        //   {
        //     std::cout << "alter pdextau: " << pdexTauValuesTemp[i]
        //               << std::endl;
        //     std::cout << "libxc pdextau: " << pdexTauValuesNonNN[i]
        //               << std::endl;
        //     std::cout << "diff: " << diff << std::endl;
        //     std::cout << std::endl;
        //   }
        //     }
        // }
        ///////////////////////////////////////////////////////////////////
      }

    for (size_t i = 0; i < nquad; i++)
      {
        if (std::abs(densityValues[2 * i + 0] + densityValues[2 * i + 1]) <=
              rhoThresholdMgga ||
            std::abs(tauValues[2 * i + 0] + tauValues[2 * i + 1]) <=
              tauThresholdMgga)
          {
            exValues[i]                       = 0.0;
            pdexDensityValuesNonNN[2 * i + 0] = 0.0;
            pdexSigmaValues[3 * i + 0]        = 0.0;
            pdexTauValuesNonNN[2 * i + 0]     = 0.0;

            pdexDensityValuesNonNN[2 * i + 1] = 0.0;
            pdexSigmaValues[3 * i + 1]        = 0.0;
            pdexSigmaValues[3 * i + 2]        = 0.0;
            pdexTauValuesNonNN[2 * i + 1]     = 0.0;

            ecValues[i]                       = 0.0;
            pdecDensityValuesNonNN[2 * i + 0] = 0.0;
            pdecSigmaValues[3 * i + 0]        = 0.0;
            pdecTauValuesNonNN[2 * i + 0]     = 0.0;

            pdecDensityValuesNonNN[2 * i + 1] = 0.0;
            pdecSigmaValues[3 * i + 1]        = 0.0;
            pdecSigmaValues[3 * i + 2]        = 0.0;
            pdecTauValuesNonNN[2 * i + 1]     = 0.0;
          }
      }

    for (size_t i = 0; i < nquad; i++)
      {
        // Evaluation of total exValue and ecValue per unit volume
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];

        pdexTauSpinUpValues[i]   = pdexTauValuesNonNN[2 * i + 0];
        pdexTauSpinDownValues[i] = pdexTauValuesNonNN[2 * i + 1];
        pdecTauSpinUpValues[i]   = pdecTauValuesNonNN[2 * i + 0];
        pdecTauSpinDownValues[i] = pdecTauValuesNonNN[2 * i + 1];
      }


    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        if (outputDataAttributes[i] == xcRemainderOutputDataAttributes::e)
          {
            xDataOut.find(outputDataAttributes[i])->second = exValues;

            cDataOut.find(outputDataAttributes[i])->second = ecValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeDensitySpinUp)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinUpValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinUpValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeDensitySpinDown)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinDownValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinDownValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeSigma)
          {
            xDataOut.find(outputDataAttributes[i])->second = pdexSigmaValues;

            cDataOut.find(outputDataAttributes[i])->second = pdecSigmaValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeTauSpinUp)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexTauSpinUpValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecTauSpinUpValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeTauSpinDown)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexTauSpinDownValues;
            cDataOut.find(outputDataAttributes[i])->second =
              pdecTauSpinDownValues;
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                      &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                                                                          &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double>                            &kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double>                            &kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excTauMGGAClass<memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excTauMGGAClass<
    memorySpace>::getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excTauMGGAClass<memorySpace>::reinitKPointDependentVariables(
    dftfe::uInt kPointIndex)
  {}

  template class excTauMGGAClass<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class excTauMGGAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
