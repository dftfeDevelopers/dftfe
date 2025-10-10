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
    const bool                     useLibxc,
    std::string                    XCType)
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
    d_funcXPtr    = funcXPtr;
    d_funcCPtr    = funcCPtr;
    d_useLibxc    = useLibxc;
    d_XCType      = XCType;
    d_enforceFHCX = (d_funcXPtr->info->flags & (1 << 16)) &&
                    (d_funcXPtr->info->flags & (1 << 17));
    d_enforceFHCC = (d_funcCPtr->info->flags & (1 << 16)) &&
                    (d_funcCPtr->info->flags & (1 << 17));

    d_tauNeededX = d_funcXPtr->info->flags & (1 << 16);
    d_tauNeededC = d_funcCPtr->info->flags & (1 << 16);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::excTauMGGAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    std::string                    modelXCInputFile,
    const bool                     useLibxc,
    std::string                    XCType)
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
    d_XCType   = XCType;
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
          "xcRemainderOutputDataAttributes do not match the allowed choices for the family type.";
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
    double rhoThresholdMgga   = 1e-12;
    double sigmaThresholdMgga = 1e-20;
    double tauThresholdMgga   = 1e-10;

    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;

    for (const auto &element : xDataOut)
      {
        outputDataAttributes.push_back(element.first);
      }

    checkInputOutputDataAttributesConsistency(outputDataAttributes);

    // see if we can use static variable for densityDescriptorData and
    // wfcDescriptorData as well.
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


    if (this->s_densityValues.size() != 2 * nquad)
      this->s_densityValues.resize(2 * nquad);

    if (this->s_sigmaValues.size() != 3 * nquad)
      this->s_sigmaValues.resize(3 * nquad);

    if (this->s_tauValues.size() != 2 * nquad)
      this->s_tauValues.resize(2 * nquad);

    auto &densityValues = this->s_densityValues;
    auto &sigmaValues   = this->s_sigmaValues;
    auto &tauValues     = this->s_tauValues;
    sigmaValues.setValue(0.0);

    if (this->s_pdexDensityValuesNonNN.size() != 2 * nquad)
      this->s_pdexDensityValuesNonNN.resize(2 * nquad);
    if (this->s_pdecDensityValuesNonNN.size() != 2 * nquad)
      this->s_pdecDensityValuesNonNN.resize(2 * nquad);

    auto &pdexDensityValuesNonNN = this->s_pdexDensityValuesNonNN;
    auto &pdecDensityValuesNonNN = this->s_pdecDensityValuesNonNN;

    if (this->s_pdexTauValuesNonNN.size() != 2 * nquad)
      this->s_pdexTauValuesNonNN.resize(2 * nquad);
    if (this->s_pdecTauValuesNonNN.size() != 2 * nquad)
      this->s_pdecTauValuesNonNN.resize(2 * nquad);

    auto &pdexTauValuesNonNN = this->s_pdexTauValuesNonNN;
    auto &pdecTauValuesNonNN = this->s_pdecTauValuesNonNN;


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

    auto &pdexTauSpinUpValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp) !=
       xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp)->second :
        this->s_pdexTauSpinUpValues;
    auto &pdexTauSpinDownValues =
      (xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown) !=
       xDataOut.end()) ?
        xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown)->second :
        this->s_pdexTauSpinDownValues;
    auto &pdecTauSpinUpValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp) !=
       cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp)->second :
        this->s_pdecTauSpinUpValues;
    auto &pdecTauSpinDownValues =
      (cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown) !=
       cDataOut.end()) ?
        cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown)->second :
        this->s_pdecTauSpinDownValues;

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


    if (pdexTauSpinUpValues.size() != nquad)
      pdexTauSpinUpValues.resize(nquad);

    if (pdexTauSpinDownValues.size() != nquad)
      pdexTauSpinDownValues.resize(nquad);

    if (pdecTauSpinUpValues.size() != nquad)
      pdecTauSpinUpValues.resize(nquad);

    if (pdecTauSpinDownValues.size() != nquad)
      pdecTauSpinDownValues.resize(nquad);

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
                                           rhoThresholdMgga,
                                           sigmaThresholdMgga,
                                           tauThresholdMgga);

    if (d_useLibxc)
      {
        // Allocate laplacian-related arrays only when using libxc
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          laplacianValues(2 * nquad, 0.0);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexLaplacianValues(2 * nquad, 0.0);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecLaplacianValues(2 * nquad, 0.0);

        exValues.setValue(0.0);
        ecValues.setValue(0.0);

        pdexDensityValuesNonNN.setValue(0.0);
        pdecDensityValuesNonNN.setValue(0.0);

        pdexSigmaValues.setValue(0.0);
        pdecSigmaValues.setValue(0.0);

        pdexTauValuesNonNN.setValue(0.0);
        pdecTauValuesNonNN.setValue(0.0);

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
        /*uncomment and modify the below part to get the parameters for  the
         * functionals*/

        // typedef struct
        // {
        //   double eta;
        // } mgga_c_scan_params;

        // typedef struct
        // {
        //   double c1, c2, d, k1;
        // } mgga_x_scan_params;

        // mgga_x_scan_params *params;
        // mgga_c_scan_params *paramsC;

        // params  = (mgga_x_scan_params *)d_funcXPtr->params;
        // paramsC = (mgga_c_scan_params *)d_funcCPtr->params;

        // std::cout << "c1: " << params->c1 << std::endl;
        // std::cout << "c2: " << params->c2 << std::endl;
        // std::cout << "d: " << params->d << std::endl;
        // std::cout << "k1: " << params->k1 << std::endl;

        // // std::cout << "eta: " << paramsC->eta << std::endl;
        // std::cout << "dens_thresholdX: " << d_funcXPtr->dens_threshold
        //           << std::endl;
        // std::cout << "zeta_thresholdX: " << d_funcXPtr->zeta_threshold
        //           << std::endl;
        // std::cout << "sigma_thresholdX: " << d_funcXPtr->sigma_threshold
        //           << std::endl;
        // std::cout << "tau_thresholdX: " << d_funcXPtr->tau_threshold
        //           << std::endl;

        // std::cout << std::endl;

        // std::cout << "dens_thresholdC: " << d_funcCPtr->dens_threshold
        //           << std::endl;
        // std::cout << "zeta_thresholdC: " << d_funcCPtr->zeta_threshold
        //           << std::endl;
        // std::cout << "sigma_thresholdC: " << d_funcCPtr->sigma_threshold
        //           << std::endl;
        // std::cout << "tau_thresholdC: " << d_funcCPtr->tau_threshold
        //           << std::endl;

#if defined(DFTFE_WITH_DEVICE)
        const std::size_t bytesDensity = densityValues.size() * sizeof(double);
        const std::size_t bytesSigma   = sigmaValues.size() * sizeof(double);
        const std::size_t bytesTau     = tauValues.size() * sizeof(double);
        const std::size_t bytesPhase1  = bytesDensity + bytesSigma + bytesTau;
        if (this->s_densityValuesTemp.size() != densityValues.size())
          {
            this->s_densityValuesTemp.resize(densityValues.size());
            this->s_sigmaValuesTemp.resize(sigmaValues.size());
            this->s_tauValuesTemp.resize(tauValues.size());
            this->s_exValuesTemp.resize(exValues.size());
            this->s_ecValuesTemp.resize(ecValues.size());
            this->s_pdexDensityTemp.resize(pdexDensityValuesNonNN.size());
            this->s_pdecDensityTemp.resize(pdecDensityValuesNonNN.size());
            this->s_pdexSigmaValuesTemp.resize(pdexSigmaValues.size());
            this->s_pdecSigmaValuesTemp.resize(pdecSigmaValues.size());
            this->s_pdexTauValuesTemp.resize(pdexTauValuesNonNN.size());
            this->s_pdecTauValuesTemp.resize(pdecTauValuesNonNN.size());
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
            double *p_tau     = p_sigma + sigmaValues.size();

            std::memcpy(p_density, &densityValues[0], bytesDensity);
            std::memcpy(p_sigma, &sigmaValues[0], bytesSigma);
            std::memcpy(p_tau, &tauValues[0], bytesTau);

            dftfe::utils::deviceMemcpyH2D(&this->s_densityValuesTemp[0],
                                          p_density,
                                          bytesDensity);
            dftfe::utils::deviceMemcpyH2D(&this->s_sigmaValuesTemp[0],
                                          p_sigma,
                                          bytesSigma);
            dftfe::utils::deviceMemcpyH2D(&this->s_tauValuesTemp[0],
                                          p_tau,
                                          bytesTau);
          }
        else
          {
            std::memcpy(&this->s_densityValuesTemp[0],
                        &densityValues[0],
                        bytesDensity);
            std::memcpy(&this->s_sigmaValuesTemp[0],
                        &sigmaValues[0],
                        bytesSigma);
            std::memcpy(&this->s_tauValuesTemp[0], &tauValues[0], bytesTau);
          }
        auto &densityValuesTemp   = this->s_densityValuesTemp;
        auto &sigmaValuesTemp     = this->s_sigmaValuesTemp;
        auto &tauValuesTemp       = this->s_tauValuesTemp;
        auto &exValuesTemp        = this->s_exValuesTemp;
        auto &ecValuesTemp        = this->s_ecValuesTemp;
        auto &pdecDensityTemp     = this->s_pdecDensityTemp;
        auto &pdexDensityTemp     = this->s_pdexDensityTemp;
        auto &pdecSigmaValuesTemp = this->s_pdecSigmaValuesTemp;
        auto &pdexSigmaValuesTemp = this->s_pdexSigmaValuesTemp;
        auto &pdecTauValuesTemp   = this->s_pdecTauValuesTemp;
        auto &pdexTauValuesTemp   = this->s_pdexTauValuesTemp;
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

        if (d_XCType == "MGGA-R2SCAN")
          {
            MGGAX_R2SCAN(nquad,
                         densityValuesTemp,
                         sigmaValuesTemp,
                         tauValuesTemp,
                         exValuesTemp,
                         pdexDensityTemp,
                         pdexSigmaValuesTemp,
                         pdexTauValuesTemp,
                         d_tauNeededX,
                         d_enforceFHCX);
            MGGAC_R2SCAN(nquad,
                         densityValuesTemp,
                         sigmaValuesTemp,
                         tauValuesTemp,
                         ecValuesTemp,
                         pdecDensityTemp,
                         pdecSigmaValuesTemp,
                         pdecTauValuesTemp,
                         d_tauNeededC,
                         d_enforceFHCC);
          }
        else if (d_XCType == "MGGA-SCAN")
          {
            MGGAX_SCAN(nquad,
                       densityValuesTemp,
                       sigmaValuesTemp,
                       tauValuesTemp,
                       exValuesTemp,
                       pdexDensityTemp,
                       pdexSigmaValuesTemp,
                       pdexTauValuesTemp,
                       d_tauNeededX,
                       d_enforceFHCX);
            MGGAC_SCAN(nquad,
                       densityValuesTemp,
                       sigmaValuesTemp,
                       tauValuesTemp,
                       ecValuesTemp,
                       pdecDensityTemp,
                       pdecSigmaValuesTemp,
                       pdecTauValuesTemp,
                       d_tauNeededC,
                       d_enforceFHCC);
          }
        else
          {
            dftfe::utils::throwException(
              "xc_func_type name is not implemented in DFT-FE. Use LIBXC to compute the M-GGA functional.");
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
            const std::size_t bytesPxTau =
              pdexTauValuesNonNN.size() * sizeof(double);
            const std::size_t bytesPcTau =
              pdecTauValuesNonNN.size() * sizeof(double);


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
            if (bytesPxTau)
              dftfe::utils::deviceMemcpyD2H(&pdexTauValuesNonNN[0],
                                            &pdexTauValuesTemp[0],
                                            bytesPxTau);

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
            if (bytesPcTau)
              dftfe::utils::deviceMemcpyD2H(&pdecTauValuesNonNN[0],
                                            &pdecTauValuesTemp[0],
                                            bytesPcTau);
          }
        else
          {
            exValues.copyFrom(exValuesTemp);
            pdexDensityValuesNonNN.copyFrom(pdexDensityTemp);
            pdexSigmaValues.copyFrom(pdexSigmaValuesTemp);
            pdexTauValuesNonNN.copyFrom(pdexTauValuesTemp);

            ecValues.copyFrom(ecValuesTemp);
            pdecDensityValuesNonNN.copyFrom(pdecDensityTemp);
            pdecSigmaValues.copyFrom(pdecSigmaValuesTemp);
            pdecTauValuesNonNN.copyFrom(pdecTauValuesTemp);
          }
#endif
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

        // Evaluation of total exValue and ecValue per unit volume
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
        pdexTauSpinUpValues[i]       = pdexTauValuesNonNN[2 * i + 0];
        pdexTauSpinDownValues[i]     = pdexTauValuesNonNN[2 * i + 1];
        pdecTauSpinUpValues[i]       = pdecTauValuesNonNN[2 * i + 0];
        pdecTauSpinDownValues[i]     = pdecTauValuesNonNN[2 * i + 1];
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
