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

#include "excDensityTauMGGAClass.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityMGGAClass<memorySpace>::excDensityMGGAClass(
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
    // d_NNGGAPtr = nullptr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityMGGAClass<memorySpace>::excDensityMGGAClass(
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
    // d_NNGGAPtr = nullptr;
#ifdef DFTFE_WITH_TORCH
    std::string errMsg ="NNMGGA is not implemented yet.";
    dftfe::utils::throwException(false, errMsg);
#endif
  }
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityMGGAClass<memorySpace>::~excDensityMGGAClass()
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityMGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
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
  excDensityMGGAClass<memorySpace>::computeRhoTauDependentXCData(
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
          {
            wfcDescriptorData[this->d_wfcDescriptorAttributesList[i]] =
              std::vector<double>(nquad, 0.0);
          }
      }

    auxDensityMatrix.applyLocalOperations(quadPoints, densityDescriptorData);
    auxDensityMatrix.applyLocalOperations(quadPoints, wfcDescriptorData);

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
    std::vector<double> pdexLaplacianValues(nquad, 0);
    std::vector<double> pdecLaplacianValues(nquad, 0);

    for (size_t i = 0; i < nquad; i++)
      {
        densityValues[2 * i + 0] = densityValuesSpinUp[i];
        densityValues[2 * i + 1] = densityValuesSpinDown[i];
        for (size_t j = 0; j < 3; j++)
          {
            sigmaValues[3 * i + 0] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinUp[3 * i + j];
            sigmaValues[3 * i + 1] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinDown[3 * i + j];
            sigmaValues[3 * i + 2] +=
              gradValuesSpinDown[3 * i + j] * gradValuesSpinDown[3 * i + j];
          }

        tauValues[2 * i + 0] = tauValuesSpinUp[i];
        tauValues[2 * i + 1] = tauValuesSpinDown[i];
      }

    std::vector<double> laplacianValues;
    laplacianValues.resize(0.0, 2 * nquad);

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
        // Evaluation of total exValue and ecValue per unit volume
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];

        pdexTauSpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexTauSpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecTauSpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecTauSpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }

#ifdef DFTFE_WITH_TORCH
    std::string errMsg = "NNMGGA is not implemented yet.";
    dftfe::utils::throwException(false,errMsg);
#endif

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



} // namespace dftfe
