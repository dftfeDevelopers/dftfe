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
// @author Vishal Subramanian, Sambit Das
//

#include <excDensityLDAClass.h>
#include <NNLDA.h>
#include <Exceptions.h>
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::excDensityLDAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr)
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
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLDAClass<memorySpace>::excDensityLDAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr,
    std::string                   modelXCInputFile)
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

    for (unsigned int i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (unsigned int j = 0; j < allowedOutputDataAttributes.size(); j++)
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
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    quadPoints,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &xDataOut,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &cDataOut) const
  {
    const unsigned int                           nquad = quadPoints.size() / 3;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      densityDescriptorData;

    for (unsigned int i = 0; i < this->d_densityDescriptorAttributesList.size();
         i++)
      {
        densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
          std::vector<double>(nquad, 0);
      }

    auxDensityMatrix.applyLocalOperations(quadPoints, densityDescriptorData);


    auto &densityValuesSpinUp =
      densityDescriptorData.find(DensityDescriptorDataAttributes::valuesSpinUp)
        ->second;
    auto &densityValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::valuesSpinDown)
        ->second;



    std::vector<double> densityValues(2 * nquad, 0);

    std::vector<double> exValues(nquad, 0);
    std::vector<double> ecValues(nquad, 0);
    std::vector<double> pdexDensityValuesNonNN(2 * nquad, 0);
    std::vector<double> pdecDensityValuesNonNN(2 * nquad, 0);
    std::vector<double> pdexDensitySpinUpValues(nquad, 0);
    std::vector<double> pdexDensitySpinDownValues(nquad, 0);
    std::vector<double> pdecDensitySpinUpValues(nquad, 0);
    std::vector<double> pdecDensitySpinDownValues(nquad, 0);

    for (unsigned int i = 0; i < nquad; i++)
      {
        densityValues[2 * i + 0] = densityValuesSpinUp[i];
        densityValues[2 * i + 1] = densityValuesSpinDown[i];
      }

    xc_lda_exc_vxc(d_funcXPtr.get(),
                   nquad,
                   &densityValues[0],
                   &exValues[0],
                   &pdexDensityValuesNonNN[0]);
    xc_lda_exc_vxc(d_funcCPtr.get(),
                   nquad,
                   &densityValues[0],
                   &ecValues[0],
                   &pdecDensityValuesNonNN[0]);

    for (unsigned int i = 0; i < nquad; i++)
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
        std::vector<double> excValuesFromNN(nquad, 0);
        const unsigned int  numDescriptors =
          this->d_densityDescriptorAttributesList.size();
        std::vector<double> pdexcDescriptorValuesFromNN(numDescriptors * nquad,
                                                        0);
        d_NNLDAPtr->evaluatevxc(&(densityValues[0]),
                                nquad,
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (unsigned int i = 0; i < nquad; i++)
          {
            exValues[i] += excValuesFromNN[i];
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
          }
      }
#endif

    for (unsigned int i = 0; i < outputDataAttributes.size(); i++)
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
      &                                                                src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const unsigned int inputVecSize,
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double> &                           kPointWeights)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLDAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double> &                           kPointWeights)
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
    unsigned int kPointIndex)
  {}

  template class excDensityLDAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityLDAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
