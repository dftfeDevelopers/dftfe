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
    const bool                     useLibxc)
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
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::excDensityLDAClass(
    std::shared_ptr<xc_func_type> &funcXPtr,
    std::shared_ptr<xc_func_type> &funcCPtr,
    std::string                    modelXCInputFile,
    const bool                     useLibxc)
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
          "xcRemainderOutputDataAttributes do not matched allowed choices for the family type.";
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



    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      densityValues(2 * nquad, 0);

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

    dftfe::internal::fillRhoVector(nquad,
                                   densityValuesSpinUp,
                                   densityValuesSpinDown,
                                   densityValues);
    if (d_useLibXC)
      {
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
        dftfe::utils::MemoryStorage<double, memorySpace> densityValuesTemp;
        dftfe::utils::MemoryStorage<double, memorySpace> ecValuesTemp,
          exValuesTemp;
        dftfe::utils::MemoryStorage<double, memorySpace> pdecDensityTemp,
          pdexDensityTemp;

        densityValuesTemp.resize(densityValues.size());
        densityValuesTemp.copyFrom(densityValues);
        ecValuesTemp.resize(ecValues.size());
        pdecDensityTemp.resize(pdecDensityValuesNonNN.size());
        exValuesTemp.resize(exValues.size());
        pdexDensityTemp.resize(pdexDensityValuesNonNN.size());
#else
        auto &densityValuesTemp = densityValues;
        auto &exValuesTemp      = exValues;
        auto &ecValuesTemp      = ecValues;
        auto &pdecDensityTemp   = pdecDensityValuesNonNN;
        auto &pdexDensityTemp   = pdexDensityValuesNonNN;
#endif
        if (d_funcXPtr->info->number == 1)
          {
            LDAX_PW(nquad, densityValuesTemp, exValuesTemp, pdexDensityTemp);
#if defined(DFTFE_WITH_DEVICE)
            exValues.copyFrom(exValuesTemp);
            pdexDensityValuesNonNN.copyFrom(pdexDensityTemp);
#endif
          }
        if (d_funcCPtr->info->number == 12)
          {
            LDAC_PW(nquad, densityValuesTemp, ecValuesTemp, pdecDensityTemp);
#if defined(DFTFE_WITH_DEVICE)
            ecValues.copyFrom(ecValuesTemp);
            pdecDensityValuesNonNN.copyFrom(pdecDensityTemp);
#endif
          }
        else
          {
            dftfe::utils::throwException(
              "xc_func_type name is not implemented in DFT-FE. Use LIBXC to compute the LDA functional.");
          }
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
