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

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::excTauMGGAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr)
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
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excTauMGGAClass<memorySpace>::excTauMGGAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr,
    std::string                   modelXCInputFile)
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
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &xDataOut,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &cDataOut) const
  {
    double tauThresholdMgga = 1e-9;
    double rhoThresholdMgga = 1e-9;
    // double sigmaThresholdMgga   = 1e-24;

    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;

    for (const auto &element : xDataOut)
      {
        outputDataAttributes.push_back(element.first);
      }

    checkInputOutputDataAttributesConsistency(outputDataAttributes);

    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      densityDescriptorData;
    std::unordered_map<WfcDescriptorDataAttributes, std::vector<double>>
      wfcDescriptorData;

    for (size_t i = 0; i < this->d_densityDescriptorAttributesList.size(); i++)
      {
        if (this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinUp ||
            this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            std::vector<double>(nquad, 0);
        else if (this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinUp ||
                 this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            std::vector<double>(3 * nquad, 0);
      }

    for (size_t i = 0; i < this->d_wfcDescriptorAttributesList.size(); i++)
      {
        if (this->d_wfcDescriptorAttributesList[i] ==
              WfcDescriptorDataAttributes::tauSpinUp ||
            this->d_wfcDescriptorAttributesList[i] ==
              WfcDescriptorDataAttributes::tauSpinDown)
          wfcDescriptorData[this->d_wfcDescriptorAttributesList[i]] =
            std::vector<double>(nquad, 0.0);
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

    std::vector<double> densityValues(2 * nquad, 0);
    std::vector<double> sigmaValues(3 * nquad, 0);
    std::vector<double> tauValues(2 * nquad, 0);

    std::vector<double> exValues(nquad, 0);
    std::vector<double> ecValues(nquad, 0);
    std::vector<double> pdexDensityValuesNonNN(2 * nquad, 0);
    std::vector<double> pdecDensityValuesNonNN(2 * nquad, 0);
    std::vector<double> pdexDensitySpinUpValues(nquad, 0);
    std::vector<double> pdexDensitySpinDownValues(nquad, 0);
    std::vector<double> pdecDensitySpinUpValues(nquad, 0);
    std::vector<double> pdecDensitySpinDownValues(nquad, 0);
    std::vector<double> pdexSigmaValues(3 * nquad, 0);
    std::vector<double> pdecSigmaValues(3 * nquad, 0);
    std::vector<double> pdexTauValuesNonNN(2 * nquad, 0);
    std::vector<double> pdecTauValuesNonNN(2 * nquad, 0);
    std::vector<double> pdexTauSpinUpValues(nquad, 0);
    std::vector<double> pdexTauSpinDownValues(nquad, 0);
    std::vector<double> pdecTauSpinUpValues(nquad, 0);
    std::vector<double> pdecTauSpinDownValues(nquad, 0);
    std::vector<double> pdexLaplacianValues(2 * nquad, 0);
    std::vector<double> pdecLaplacianValues(2 * nquad, 0);

    for (size_t i = 0; i < nquad; i++)
      {
        densityValues[2 * i + 0] = std::abs(densityValuesSpinUp[i]);
        densityValues[2 * i + 1] = std::abs(densityValuesSpinDown[i]);
        for (size_t j = 0; j < 3; j++)
          {
            sigmaValues[3 * i + 0] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinUp[3 * i + j];
            sigmaValues[3 * i + 1] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinDown[3 * i + j];
            sigmaValues[3 * i + 2] +=
              gradValuesSpinDown[3 * i + j] * gradValuesSpinDown[3 * i + j];
          }
        // sigmaValues[3 * i + 0] =
        //   std::max(sigmaValues[3 * i + 0], sigmaThresholdMgga);
        // sigmaValues[3 * i + 2] =
        //   std::max(sigmaValues[3 * i + 2], sigmaThresholdMgga);

        tauValues[2 * i + 0] = std::max(tauValuesSpinUp[i], tauThresholdMgga);
        tauValues[2 * i + 1] = std::max(tauValuesSpinDown[i], tauThresholdMgga);
      }

    std::vector<double> laplacianValues(2 * nquad, 0.0);

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
