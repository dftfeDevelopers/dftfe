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
// @author Vishal Subramanian, Sambit Das
//

#include "excDensityGGAClass.h"
#include "NNGGA.h"
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
  excDensityGGAClass<memorySpace>::excDensityGGAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    const bool                     useLibxc,
    std::string                    XCType)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::GGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_NNGGAPtr = nullptr;
    d_useLibXC = useLibxc;
    d_XCType   = XCType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::excDensityGGAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    std::string                    modelXCInputFile,
    const bool                     useLibxc,
    std::string                    XCType)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::GGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNGGAPtr = new NNGGA(modelXCInputFile, true);
#endif
    d_useLibXC = useLibxc;
    d_XCType   = XCType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::~excDensityGGAClass()
  {
    if (d_NNGGAPtr != nullptr)
      delete d_NNGGAPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    const std::vector<xcRemainderOutputDataAttributes>
      allowedOutputDataAttributes = {
        xcRemainderOutputDataAttributes::e,
        xcRemainderOutputDataAttributes::pdeDensitySpinUp,
        xcRemainderOutputDataAttributes::pdeDensitySpinDown,
        xcRemainderOutputDataAttributes::pdeSigma};

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
          {
            if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
              isFound = true;
          }


        std::string errMsg =
          "xcRemainderOutputDataAttributes do not match the allowed choices for the family type.";
        dftfe::utils::throwException(isFound, errMsg);
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::computeRhoTauDependentXCData(
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
    // double time1 = MPI_Wtime();
    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::unordered_map<
      DensityDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityDescriptorData;

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

    auxDensityMatrix.applyLocalOperations(quadIndexRange,
                                          densityDescriptorData);


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



    if (this->s_densityValues.size() != 2 * nquad)
      this->s_densityValues.resize(2 * nquad);

    if (this->s_sigmaValues.size() != 3 * nquad)
      this->s_sigmaValues.resize(3 * nquad);

    auto &densityValues = this->s_densityValues;
    auto &sigmaValues   = this->s_sigmaValues;
    sigmaValues.setValue(0.0);

    if (this->s_pdexDensityValuesNonNN.size() != 2 * nquad)
      this->s_pdexDensityValuesNonNN.resize(2 * nquad);
    if (this->s_pdecDensityValuesNonNN.size() != 2 * nquad)
      this->s_pdecDensityValuesNonNN.resize(2 * nquad);

    auto &pdexDensityValuesNonNN = this->s_pdexDensityValuesNonNN;
    auto &pdecDensityValuesNonNN = this->s_pdecDensityValuesNonNN;



    auto &exValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::e) != xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::e)->second :
        this->s_exValues;
    auto &ecValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::e) != cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::e)->second :
        this->s_ecValues;

    auto &pdexDensitySpinUpValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp) !=
       xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp)
          ->second :
        this->s_pdexDensitySpinUpValues;
    auto &pdexDensitySpinDownValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown) !=
       xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown)
          ->second :
        this->s_pdexDensitySpinDownValues;
    auto &pdecDensitySpinUpValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp) !=
       cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp)
          ->second :
        this->s_pdecDensitySpinUpValues;
    auto &pdecDensitySpinDownValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown) !=
       cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown)
          ->second :
        this->s_pdecDensitySpinDownValues;

    auto &pdexSigmaValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::pdeSigma) !=
       xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::pdeSigma)->second :
        this->s_pdexSigmaValues;
    auto &pdecSigmaValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::pdeSigma) !=
       cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::pdeSigma)->second :
        this->s_pdecSigmaValues;


    if (exValues.size() != nquad)
      exValues.resize(nquad);
    if (ecValues.size() != nquad)
      ecValues.resize(nquad);
    if (pdexDensitySpinUpValues.size() != nquad)
      pdexDensitySpinUpValues.resize(nquad);

    if (pdexDensitySpinDownValues.size() != nquad)
      pdexDensitySpinDownValues.resize(nquad);

    if (pdecDensitySpinUpValues.size() != nquad)
      pdecDensitySpinUpValues.resize(nquad);

    if (pdecDensitySpinDownValues.size() != nquad)
      pdecDensitySpinDownValues.resize(nquad);

    if (pdexSigmaValues.size() != 3 * nquad)
      pdexSigmaValues.resize(3 * nquad);

    if (pdecSigmaValues.size() != 3 * nquad)
      pdecSigmaValues.resize(3 * nquad);

    dftfe::internal::fillRhoSigmaVector(nquad,
                                        densityValuesSpinUp,
                                        densityValuesSpinDown,
                                        gradValuesSpinUp,
                                        gradValuesSpinDown,
                                        densityValues,
                                        sigmaValues);

    if (d_useLibXC)
      {
        // typedef struct
        // {
        //   double alpha, beta, gamma;
        // } gga_x_lb_params;

        // gga_x_lb_params *params;

        // params = (gga_x_lb_params *)d_funcXPtr->params;

        // std::cout << "d0: " << params->alpha << std::endl;
        // std::cout << "d1: " << params->beta << std::endl;
        // std::cout << "d2: " << params->gamma << std::endl;
        // std::cout << "dens_thresholdX: " << d_funcXPtr->dens_threshold
        //           << std::endl;
        // std::cout << "zeta_thresholdX: " << d_funcXPtr->zeta_threshold
        //           << std::endl;

        // std::cout << "sigma_thresholdX: " << d_funcXPtr->sigma_threshold
        //           << std::endl;

        exValues.setValue(0.0);
        ecValues.setValue(0.0);

        pdexDensityValuesNonNN.setValue(0.0);
        pdecDensityValuesNonNN.setValue(0.0);

        pdexSigmaValues.setValue(0.0);
        pdecSigmaValues.setValue(0.0);

        xc_gga_exc_vxc(d_funcXPtr.get(),
                       nquad,
                       densityValues.data(),
                       sigmaValues.data(),
                       exValues.data(),
                       pdexDensityValuesNonNN.data(),
                       pdexSigmaValues.data());
        xc_gga_exc_vxc(d_funcCPtr.get(),
                       nquad,
                       densityValues.data(),
                       sigmaValues.data(),
                       ecValues.data(),
                       pdecDensityValuesNonNN.data(),
                       pdecSigmaValues.data());
      }
    else
      {
#if defined(DFTFE_WITH_DEVICE)

        const std::size_t bytesDensity = densityValues.size() * sizeof(double);
        const std::size_t bytesSigma   = sigmaValues.size() * sizeof(double);
        const std::size_t bytesPhase1  = bytesDensity + bytesSigma;

        if (this->s_densityValuesTemp.size() != densityValues.size())
          {
            this->s_densityValuesTemp.resize(densityValues.size());
            this->s_sigmaValuesTemp.resize(sigmaValues.size());
            this->s_exValuesTemp.resize(exValues.size());
            this->s_ecValuesTemp.resize(ecValues.size());
            this->s_pdexDensityTemp.resize(pdexDensityValuesNonNN.size());
            this->s_pdecDensityTemp.resize(pdecDensityValuesNonNN.size());
            this->s_pdexSigmaValuesTemp.resize(pdexSigmaValues.size());
            this->s_pdecSigmaValuesTemp.resize(pdecSigmaValues.size());
          }

        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          {
            auto ensurePinnedCapacity = [&](std::size_t needBytes) {
              if (this->s_pinnedCap < needBytes)
                {
                  if (this->s_pinnedBuf)
                    dftfe::utils::deviceHostFree(this->s_pinnedBuf);
                  dftfe::utils::deviceHostMalloc(&this->s_pinnedBuf, needBytes);
                  this->s_pinnedCap = needBytes;
                }
            };
            ensurePinnedCapacity(bytesPhase1);
            double *p_density = static_cast<double *>(this->s_pinnedBuf);
            double *p_sigma   = p_density + densityValues.size();

            std::memcpy(p_density, &densityValues[0], bytesDensity);
            std::memcpy(p_sigma, &sigmaValues[0], bytesSigma);

            dftfe::utils::deviceMemcpyH2D(&this->s_densityValuesTemp[0],
                                          p_density,
                                          bytesDensity);
            dftfe::utils::deviceMemcpyH2D(&this->s_sigmaValuesTemp[0],
                                          p_sigma,
                                          bytesSigma);
          }
        else
          {
            std::memcpy(&this->s_densityValuesTemp[0],
                        &densityValues[0],
                        bytesDensity);
            std::memcpy(&this->s_sigmaValuesTemp[0],
                        &sigmaValues[0],
                        bytesSigma);
          }

        auto &densityValuesTemp   = this->s_densityValuesTemp;
        auto &sigmaValuesTemp     = this->s_sigmaValuesTemp;
        auto &exValuesTemp        = this->s_exValuesTemp;
        auto &ecValuesTemp        = this->s_ecValuesTemp;
        auto &pdecDensityTemp     = this->s_pdecDensityTemp;
        auto &pdexDensityTemp     = this->s_pdexDensityTemp;
        auto &pdecSigmaValuesTemp = this->s_pdecSigmaValuesTemp;
        auto &pdexSigmaValuesTemp = this->s_pdexSigmaValuesTemp;

#else
        auto &densityValuesTemp   = densityValues;
        auto &sigmaValuesTemp     = sigmaValues;
        auto &exValuesTemp        = exValues;
        auto &ecValuesTemp        = ecValues;
        auto &pdecDensityTemp     = pdecDensityValuesNonNN;
        auto &pdexDensityTemp     = pdexDensityValuesNonNN;
        auto &pdecSigmaValuesTemp = pdecSigmaValues;
        auto &pdexSigmaValuesTemp = pdexSigmaValues;
#endif
        if (d_XCType == "GGA-PBE")
          {
            GGAX_PBE(nquad,
                     densityValuesTemp,
                     sigmaValuesTemp,
                     exValuesTemp,
                     pdexDensityTemp,
                     pdexSigmaValuesTemp);
            GGAC_PBE(nquad,
                     densityValuesTemp,
                     sigmaValuesTemp,
                     ecValuesTemp,
                     pdecDensityTemp,
                     pdecSigmaValuesTemp);
          }
        else if (d_XCType == "GGA-RPBE")
          {
            GGAX_RPBE(nquad,
                      densityValuesTemp,
                      sigmaValuesTemp,
                      exValuesTemp,
                      pdexDensityTemp,
                      pdexSigmaValuesTemp);
            GGAC_PBE(nquad,
                     densityValuesTemp,
                     sigmaValuesTemp,
                     ecValuesTemp,
                     pdecDensityTemp,
                     pdecSigmaValuesTemp);
          }

        // else if (d_XCType == "GGA-LBxPBEc")
        //   {
        //     GGAX_LB(nquad,
        //             densityValuesTemp,
        //             sigmaValuesTemp,
        //             exValuesTemp,
        //             pdexDensityTemp,
        //             pdexSigmaValuesTemp);
        //     GGAC_PBE(nquad,
        //              densityValuesTemp,
        //              sigmaValuesTemp,
        //              ecValuesTemp,
        //              pdecDensityTemp,
        //              pdecSigmaValuesTemp);
        //   }
        else
          {
            dftfe::utils::throwException(
              "xc_func_type name is not implemented in DFT-FE. Use LIBXC to compute the LDA functional.");
          }
#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          {
            const std::size_t bytesEx = exValues.size() * sizeof(double);
            const std::size_t bytesEc = ecValues.size() * sizeof(double);
            const std::size_t bytesPxDensity =
              pdexDensityValuesNonNN.size() * sizeof(double);
            const std::size_t bytesPcDensity =
              pdecDensityValuesNonNN.size() * sizeof(double);

            const std::size_t bytesPxSigma =
              pdexSigmaValues.size() * sizeof(double);
            const std::size_t bytesPcSigma =
              pdecSigmaValues.size() * sizeof(double);

            if (bytesEx)
              dftfe::utils::deviceMemcpyD2H(&exValues[0],
                                            &exValuesTemp[0],
                                            bytesEx);
            if (bytesPxDensity)
              dftfe::utils::deviceMemcpyD2H(&pdexDensityValuesNonNN[0],
                                            &pdexDensityTemp[0],
                                            bytesPxDensity);
            if (bytesPxSigma)
              dftfe::utils::deviceMemcpyD2H(&pdexSigmaValues[0],
                                            &pdexSigmaValuesTemp[0],
                                            bytesPxSigma);

            if (bytesEc)
              dftfe::utils::deviceMemcpyD2H(&ecValues[0],
                                            &ecValuesTemp[0],
                                            bytesEc);
            if (bytesPcDensity)
              dftfe::utils::deviceMemcpyD2H(&pdecDensityValuesNonNN[0],
                                            &pdecDensityTemp[0],
                                            bytesPcDensity);
            if (bytesPcSigma)
              dftfe::utils::deviceMemcpyD2H(&pdecSigmaValues[0],
                                            &pdecSigmaValuesTemp[0],
                                            bytesPcSigma);
            dftfe::utils::deviceSynchronize();
          }
        else
          {
            exValues.copyFrom(exValuesTemp);
            pdexDensityValuesNonNN.copyFrom(pdexDensityTemp);
            pdexSigmaValues.copyFrom(pdexSigmaValuesTemp);

            ecValues.copyFrom(ecValuesTemp);
            pdecDensityValuesNonNN.copyFrom(pdecDensityTemp);
            pdecSigmaValues.copyFrom(pdecSigmaValuesTemp);
          }
#endif
      }
    for (size_t i = 0; i < nquad; i++)
      {
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }


#ifdef DFTFE_WITH_TORCH
    if (d_NNGGAPtr != nullptr)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     excValuesFromNN(nquad, 0);
        const size_t numDescriptors = 5;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexcDescriptorValuesFromNN(numDescriptors * nquad, 0);
        d_NNGGAPtr->evaluatevxc(&(densityValues[0]),
                                &sigmaValues[0],
                                nquad,
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (size_t i = 0; i < nquad; i++)
          {
            exValues[i] += excValuesFromNN[i] * (densityValues[2 * i + 0] +
                                                 densityValues[2 * i + 1]);
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
            pdexSigmaValues[3 * i + 0] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 2];
            pdexSigmaValues[3 * i + 1] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 3];
            pdexSigmaValues[3 * i + 2] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 4];
          }
      }
#endif
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                      &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                                                                          &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double>                            &kPointWeights)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double>                            &kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityGGAClass<memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityGGAClass<
    memorySpace>::getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::reinitKPointDependentVariables(
    dftfe::uInt kPointIndex)
  {}

  template class excDensityGGAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityGGAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
