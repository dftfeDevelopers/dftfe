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

#include <excDensityLDAClass.h>
#include <NNLDA.h>
#include <Exceptions.h>
#include <dftfeDataTypes.h>
#include <excManagerKernels.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#endif
#include <exchangeCorrelationFunctionalEvaluator.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::excDensityLDAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    const bool                     useLibxc,
    std::string                    XCType)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::LDA,
        densityFamilyType::LDA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_NNLDAPtr = nullptr;
    d_useLibXC = useLibxc;
    d_XCType   = XCType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::excDensityLDAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    std::string                    modelXCInputFile,
    const bool                     useLibxc,
    std::string                    XCType)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::LDA,
        densityFamilyType::LDA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNLDAPtr = new NNLDA(modelXCInputFile, true);
#endif
    d_useLibXC = useLibxc;
    d_XCType   = XCType;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::~excDensityLDAClass()
  {
    if (d_NNLDAPtr != nullptr)
      delete d_NNLDAPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    const std::vector<xcRemainderOutputDataAttributes>
      allowedOutputDataAttributes = {
        xcRemainderOutputDataAttributes::e,
        xcRemainderOutputDataAttributes::pdeDensitySpinUp,
        xcRemainderOutputDataAttributes::pdeDensitySpinDown};

    for (dftfe::uInt i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (dftfe::uInt j = 0; j < allowedOutputDataAttributes.size(); j++)
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
  excDensityLDAClass<memorySpace>::computeRhoTauDependentXCData(
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
    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::unordered_map<
      DensityDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityDescriptorData;

    for (dftfe::uInt i = 0; i < this->d_densityDescriptorAttributesList.size();
         i++)
      {
        densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            nquad, 0);
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



    if (this->s_densityValues.size() != 2 * nquad)
      this->s_densityValues.resize(2 * nquad);


    auto &densityValues = this->s_densityValues;

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

    dftfe::internal::fillRhoVector(nquad,
                                   densityValuesSpinUp,
                                   densityValuesSpinDown,
                                   densityValues);
    if (d_useLibXC)
      {
        /*uncomment and modify the below part to get the parameters for  the
         * functionals*/

        // typedef struct
        // {
        //   double gamma[2];
        //   double beta1[2];
        //   double beta2[2];
        //   double a[2], b[2], c[2], d[2];
        // } lda_c_pz_params;

        // lda_c_pz_params *params;

        // params = (lda_c_pz_params *)d_funcCPtr->params;

        // std::cout << "gamma0: " << params->gamma[0] << std::endl;
        // std::cout << "gamma1: " << params->gamma[1] << std::endl;
        // std::cout << "beta10: " << params->beta1[0] << std::endl;
        // std::cout << "beta11: " << params->beta1[1] << std::endl;
        // std::cout << "beta20: " << params->beta2[0] << std::endl;
        // std::cout << "beta21: " << params->beta2[1] << std::endl;
        // std::cout << "a0: " << params->a[0] << std::endl;
        // std::cout << "a1: " << params->a[1] << std::endl;
        // std::cout << "b0: " << params->b[0] << std::endl;
        // std::cout << "b1: " << params->b[1] << std::endl;
        // std::cout << "c0: " << params->c[0] << std::endl;
        // std::cout << "c1: " << params->c[1] << std::endl;
        // std::cout << "d0: " << params->d[0] << std::endl;
        // std::cout << "d1: " << params->d[1] << std::endl;
        // // std::cout << "dens_thresholdX: " << d_funcXPtr->dens_threshold
        // //           << std::endl;
        // // std::cout << "zeta_thresholdX: " << d_funcXPtr->zeta_threshold
        // //           << std::endl;

        // std::cout << std::endl;

        // std::cout << "dens_thresholdC: " << d_funcCPtr->dens_threshold
        //           << std::endl;
        // std::cout << "zeta_thresholdC: " << d_funcCPtr->zeta_threshold
        //           << std::endl;

        exValues.setValue(0.0);
        ecValues.setValue(0.0);

        pdexDensityValuesNonNN.setValue(0.0);
        pdecDensityValuesNonNN.setValue(0.0);

        xc_lda_exc_vxc(d_funcXPtr.get(),
                       nquad,
                       densityValues.data(),
                       exValues.data(),
                       pdexDensityValuesNonNN.data());
        xc_lda_exc_vxc(d_funcCPtr.get(),
                       nquad,
                       densityValues.data(),
                       ecValues.data(),
                       pdecDensityValuesNonNN.data());
      }
    else
      {
#if defined(DFTFE_WITH_DEVICE)

        const std::size_t bytesDensity = densityValues.size() * sizeof(double);
        const std::size_t bytesPhase1  = bytesDensity;

        if (this->s_densityValuesTemp.size() != densityValues.size())
          {
            this->s_densityValuesTemp.resize(densityValues.size());
            this->s_exValuesTemp.resize(exValues.size());
            this->s_ecValuesTemp.resize(ecValues.size());
            this->s_pdexDensityTemp.resize(pdexDensityValuesNonNN.size());
            this->s_pdecDensityTemp.resize(pdecDensityValuesNonNN.size());
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

            std::memcpy(p_density, &densityValues[0], bytesDensity);

            dftfe::utils::deviceMemcpyH2D(&this->s_densityValuesTemp[0],
                                          p_density,
                                          bytesDensity);
          }
        else
          {
            std::memcpy(&this->s_densityValuesTemp[0],
                        &densityValues[0],
                        bytesDensity);
          }

        auto &densityValuesTemp = this->s_densityValuesTemp;
        auto &exValuesTemp      = this->s_exValuesTemp;
        auto &ecValuesTemp      = this->s_ecValuesTemp;
        auto &pdecDensityTemp   = this->s_pdecDensityTemp;
        auto &pdexDensityTemp   = this->s_pdexDensityTemp;
#else
        auto &densityValuesTemp = densityValues;
        auto &exValuesTemp      = exValues;
        auto &ecValuesTemp      = ecValues;
        auto &pdecDensityTemp   = pdecDensityValuesNonNN;
        auto &pdexDensityTemp   = pdexDensityValuesNonNN;
#endif
        if (d_XCType == "LDA-PW")
          {
            LDAX_SLATER(nquad,
                        densityValuesTemp,
                        exValuesTemp,
                        pdexDensityTemp);
            LDAC_PW(nquad, densityValuesTemp, ecValuesTemp, pdecDensityTemp);
          }

        else if (d_XCType == "LDA-PZ")
          {
            LDAX_SLATER(nquad,
                        densityValuesTemp,
                        exValuesTemp,
                        pdexDensityTemp);
            LDAC_PZ(nquad, densityValuesTemp, ecValuesTemp, pdecDensityTemp);
          }

        else if (d_XCType == "LDA-VWN")
          {
            LDAX_SLATER(nquad,
                        densityValuesTemp,
                        exValuesTemp,
                        pdexDensityTemp);
            LDAC_VWN(nquad, densityValuesTemp, ecValuesTemp, pdecDensityTemp);
          }
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

            if (bytesEx)
              dftfe::utils::deviceMemcpyD2H(&exValues[0],
                                            &exValuesTemp[0],
                                            bytesEx);
            if (bytesPxDensity)
              dftfe::utils::deviceMemcpyD2H(&pdexDensityValuesNonNN[0],
                                            &pdexDensityTemp[0],
                                            bytesPxDensity);

            if (bytesEc)
              dftfe::utils::deviceMemcpyD2H(&ecValues[0],
                                            &ecValuesTemp[0],
                                            bytesEc);
            if (bytesPcDensity)
              dftfe::utils::deviceMemcpyD2H(&pdecDensityValuesNonNN[0],
                                            &pdecDensityTemp[0],
                                            bytesPcDensity);
          }
        else
          {
            exValues.copyFrom(exValuesTemp);
            pdexDensityValuesNonNN.copyFrom(pdexDensityTemp);

            ecValues.copyFrom(ecValuesTemp);
            pdecDensityValuesNonNN.copyFrom(pdecDensityTemp);
          }
#endif
      }

    for (dftfe::uInt i = 0; i < nquad; i++)
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
    if (d_NNLDAPtr != nullptr)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                          excValuesFromNN(nquad, 0);
        const dftfe::uInt numDescriptors = 2;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexcDescriptorValuesFromNN(numDescriptors * nquad, 0);
        d_NNLDAPtr->evaluatevxc(&(densityValues[0]),
                                nquad,
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (dftfe::uInt i = 0; i < nquad; i++)
          {
            exValues[i] += excValuesFromNN[i] * (densityValues[2 * i + 0] +
                                                 densityValues[2 * i + 1]);
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
          }
      }
#endif

    for (dftfe::uInt i = 0; i < outputDataAttributes.size(); i++)
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
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                      &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                                                                          &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double>                            &kPointWeights)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double>                            &kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityLDAClass<memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityLDAClass<
    memorySpace>::getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::reinitKPointDependentVariables(
    dftfe::uInt kPointIndex)
  {}

  template class excDensityLDAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityLDAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
