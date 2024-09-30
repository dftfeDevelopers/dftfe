#include <excDensityLLMGGAClass.h>
#include <NNLLMGGA.h>
#include <algorithm>
#include <cmath>
#include "Exceptions.h"
#include "FiniteDifference.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLLMGGAClass<memorySpace>::excDensityLLMGGAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::LLMGGA,
        densityFamilyType::LLMGGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown,
          DensityDescriptorDataAttributes::laplacianSpinUp,
          DensityDescriptorDataAttributes::laplacianSpinDown})
  {
    d_funcXPtr    = funcXPtr;
    d_funcCPtr    = funcCPtr;
    d_NNLLMGGAPtr = nullptr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLLMGGAClass<memorySpace>::excDensityLLMGGAClass(
    std::shared_ptr<xc_func_type> funcXPtr,
    std::shared_ptr<xc_func_type> funcCPtr,
    std::string                   modelXCInputFile)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::LLMGGA,
        densityFamilyType::LLMGGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown,
          DensityDescriptorDataAttributes::laplacianSpinUp,
          DensityDescriptorDataAttributes::laplacianSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNLLMGGAPtr = new NNLLMGGA(modelXCInputFile, true);
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityLLMGGAClass<memorySpace>::~excDensityLLMGGAClass()
  {
    if (d_NNLLMGGAPtr != nullptr)
      delete d_NNLLMGGAPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLLMGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    const std::vector<xcRemainderOutputDataAttributes>
      allowedOutputDataAttributes = {
        xcRemainderOutputDataAttributes::e,
        xcRemainderOutputDataAttributes::vSpinUp,
        xcRemainderOutputDataAttributes::vSpinDown,
        xcRemainderOutputDataAttributes::pdeDensitySpinUp,
        xcRemainderOutputDataAttributes::pdeDensitySpinDown,
        xcRemainderOutputDataAttributes::pdeSigma,
        xcRemainderOutputDataAttributes::pdeLaplacianSpinUp,
        xcRemainderOutputDataAttributes::pdeLaplacianSpinDown};

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
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
  excDensityLLMGGAClass<memorySpace>::computeRhoTauDependentXCData(
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

    // d_densityDescriptorAttributesList not defined
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

    bool isVxcBeingComputed = false;
    if (std::find(outputDataAttributes.begin(),
                  outputDataAttributes.end(),
                  xcRemainderOutputDataAttributes::vSpinUp) !=
          outputDataAttributes.end() ||
        std::find(outputDataAttributes.begin(),
                  outputDataAttributes.end(),
                  xcRemainderOutputDataAttributes::vSpinDown) !=
          outputDataAttributes.end())
      isVxcBeingComputed = true;

    auxDensityMatrix.applyLocalOperations(quadPoints, densityDescriptorData);

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
    auto &laplacianValuesSpinUp =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::laplacianSpinUp)
        ->second;
    auto &laplacianValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::laplacianSpinDown)
        ->second;


    std::vector<double> densityValues(2 * nquad, 0);
    std::vector<double> sigmaValues(3 * nquad, 0);
    std::vector<double> laplacianValues(2 * nquad, 0);

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

        laplacianValues[2 * i + 0] = laplacianValuesSpinUp[i];
        laplacianValues[2 * i + 1] = laplacianValuesSpinDown[i];
      }

    xc_gga_exc_vxc(d_funcXPtr.get(),
                   nquad,
                   &densityValues[0],
                   &sigmaValues[0],
                   &exValues[0],
                   &pdexDensityValuesNonNN[0],
                   &pdexSigmaValues[0]);
    xc_gga_exc_vxc(d_funcCPtr.get(),
                   nquad,
                   &densityValues[0],
                   &sigmaValues[0],
                   &ecValues[0],
                   &pdecDensityValuesNonNN[0],
                   &pdecSigmaValues[0]);

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
    if (d_NNLLMGGAPtr != nullptr)
      {
        std::vector<double> excValuesFromNN(nquad, 0);
        const size_t        numDescriptors =
          this->d_densityDescriptorAttributesList.size();
        std::vector<double> pdexcDescriptorValuesFromNN(numDescriptors * nquad,
                                                        0);

        d_NNLLMGGAPtr->evaluatevxc(&(densityValues[0]),
                                   &sigmaValues[0],
                                   &laplacianValues[0],
                                   nquad,
                                   &excValuesFromNN[0],
                                   &pdexcDescriptorValuesFromNN[0]);

        for (size_t i = 0; i < nquad; i++)
          {
            exValues[i] += excValuesFromNN[i];
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
          }
      }
#endif

    std::vector<double> vxValuesSpinUp(nquad, 0);
    std::vector<double> vcValuesSpinUp(nquad, 0);
    std::vector<double> vxValuesSpinDown(nquad, 0);
    std::vector<double> vcValuesSpinDown(nquad, 0);

    if (isVxcBeingComputed)
      {
        std::vector<double> pdexGradDensityidimSpinUpStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinUpStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinUp(
          3, std::vector<double>(nquad, 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinUp(
          3, std::vector<double>(nquad, 0));
        std::vector<double> pdexLapDensityidimSpinUpStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecLapDensityidimSpinUpStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> laplacianTermsPdexLapDensitySpinUp(
          3, std::vector<double>(nquad, 0));
        std::vector<std::vector<double>> laplacianTermsPdecLapDensitySpinUp(
          3, std::vector<double>(nquad, 0));

        std::vector<double> pdexGradDensityidimSpinDownStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinDownStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinDown(
          3, std::vector<double>(nquad, 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinDown(
          3, std::vector<double>(nquad, 0));
        std::vector<double> pdexLapDensityidimSpinDownStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecLapDensityidimSpinDownStencil(
          nquad * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> laplacianTermsPdexLapDensitySpinDown(
          3, std::vector<double>(nquad, 0));
        std::vector<std::vector<double>> laplacianTermsPdecLapDensitySpinDown(
          3, std::vector<double>(nquad, 0));


        std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
          densityDescriptorDataForFD;

        std::vector<double> densityValuesFD(2 * nquad, 0);
        std::vector<double> sigmaValuesFD(3 * nquad, 0);
        std::vector<double> laplacianValuesFD(2 * nquad, 0);

        std::vector<double> exValuesFD(nquad, 0);
        std::vector<double> ecValuesFD(nquad, 0);
        std::vector<double> pdexDensityValuesNonNNFD(2 * nquad,
                                                     0); // not used
        std::vector<double> pdecDensityValuesNonNNFD(2 * nquad,
                                                     0); // not used
        std::vector<double> pdexSigmaValuesFD(3 * nquad, 0);
        std::vector<double> pdecSigmaValuesFD(3 * nquad, 0);
        std::vector<double> pdexLaplacianValuesFD(2 * nquad, 0);
        std::vector<double> pdecLaplacianValuesFD(2 * nquad,
                                                  0); // not used

        std::vector<double> d_spacingFDStencil(nquad, 1e-4);

        for (size_t idim = 0; idim < 3; idim++)
          {
            for (size_t istencil = 0;
                 istencil < d_vxcDivergenceTermFDStencilSize;
                 istencil++)
              {
                std::vector<double> quadShiftedFD = quadPoints;
                for (size_t igrid = 0; igrid < nquad; igrid++)
                  {
                    // create FD grid
                    quadShiftedFD[3 * igrid + idim] =
                      quadPoints[3 * igrid + idim] +
                      (-std::floor(d_vxcDivergenceTermFDStencilSize / 2) *
                         d_spacingFDStencil[igrid] +
                       igrid * d_spacingFDStencil[igrid]);
                  }

                auxDensityMatrix.applyLocalOperations(
                  quadShiftedFD, densityDescriptorDataForFD);

                auto &densityValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinUp)
                    ->second;
                auto &densityValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinDown)
                    ->second;
                auto &gradValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
                    ->second;
                auto &gradValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValuesSpinDown)
                    ->second;
                auto &laplacianValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::laplacianSpinUp)
                    ->second;
                auto &laplacianValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::laplacianSpinDown)
                    ->second;

                for (size_t i = 0; i < nquad; i++)
                  {
                    densityValuesFD[2 * i + 0] = densityValuesSpinUpFD[i];
                    densityValuesFD[2 * i + 1] = densityValuesSpinDownFD[i];

                    sigmaValuesFD[3 * i + 0] = 0;
                    sigmaValuesFD[3 * i + 1] = 0;
                    sigmaValuesFD[3 * i + 2] = 0;

                    for (size_t j = 0; j < 3; j++)
                      {
                        sigmaValuesFD[3 * i + 0] +=
                          gradValuesSpinUpFD[3 * i + j] *
                          gradValuesSpinUpFD[3 * i + j];
                        sigmaValuesFD[3 * i + 1] +=
                          gradValuesSpinUpFD[3 * i + j] *
                          gradValuesSpinDownFD[3 * i + j];
                        sigmaValuesFD[3 * i + 2] +=
                          gradValuesSpinDownFD[3 * i + j] *
                          gradValuesSpinDownFD[3 * i + j];
                      }

                    laplacianValuesFD[2 * i + 0] = laplacianValuesSpinUp[i];
                    laplacianValuesFD[2 * i + 1] = laplacianValuesSpinDown[i];
                  }

                xc_gga_exc_vxc(d_funcXPtr.get(),
                               nquad,
                               &densityValuesFD[0],
                               &sigmaValuesFD[0],
                               &exValuesFD[0],
                               &pdexDensityValuesNonNNFD[0],
                               &pdexSigmaValuesFD[0]);
                xc_gga_exc_vxc(d_funcCPtr.get(),
                               nquad,
                               &densityValuesFD[0],
                               &sigmaValuesFD[0],
                               &ecValuesFD[0],
                               &pdecDensityValuesNonNNFD[0],
                               &pdecSigmaValuesFD[0]);

#ifdef DFTFE_WITH_TORCH
                if (d_NNLLMGGAPtr != nullptr)
                  {
                    std::vector<double> excValuesFromNNFD(nquad, 0);
                    const size_t        numDescriptors =
                      this->d_densityDescriptorAttributesList.size();
                    std::vector<double> pdexcDescriptorValuesFromNNFD(
                      numDescriptors * nquad, 0);


                    d_NNLLMGGAPtr->evaluatevxc(
                      &(densityValuesFD[0]),
                      &sigmaValuesFD[0],
                      &laplacianValuesFD[0],
                      nquad,
                      &excValuesFromNNFD[0],
                      &pdexcDescriptorValuesFromNNFD[0]);

                    for (size_t i = 0; i < nquad; i++)
                      {
                        pdexSigmaValuesFD[3 * i + 0] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 2];
                        pdexSigmaValuesFD[3 * i + 1] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 3];
                        pdexSigmaValuesFD[3 * i + 2] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 4];
                        pdexLaplacianValuesFD[2 * i + 0] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 5];
                        pdexLaplacianValuesFD[2 * i + 1] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 6];
                      }
                  }
#endif

                for (size_t igrid = 0; igrid < nquad; igrid++)
                  {
                    pdexGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        gradValuesSpinUpFD[3 * igrid + idim] *
                          (2.0 * pdexSigmaValuesFD[3 * igrid] +
                           pdexSigmaValuesFD[3 * igrid + 1]) +
                        gradValuesSpinDownFD[3 * igrid + idim] *
                          (2.0 * pdexSigmaValuesFD[3 * igrid + 2] +
                           pdexSigmaValuesFD[3 * igrid + 1]);

                    pdecGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        gradValuesSpinUpFD[3 * igrid + idim] *
                          (2.0 * pdecSigmaValuesFD[3 * igrid] +
                           pdecSigmaValuesFD[3 * igrid + 1]) +
                        gradValuesSpinDownFD[3 * igrid + idim] *
                          (2.0 * pdecSigmaValuesFD[3 * igrid + 2] +
                           pdecSigmaValuesFD[3 * igrid + 1]);

                    pdexLapDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdexLaplacianValuesFD[2 * igrid];

                    pdecLapDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdecLaplacianValuesFD[2 * igrid];

                    pdexGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        gradValuesSpinUpFD[3 * igrid + idim] *
                          (2.0 * pdexSigmaValuesFD[3 * igrid] +
                           pdexSigmaValuesFD[3 * igrid + 1]) +
                        gradValuesSpinDownFD[3 * igrid + idim] *
                          (2.0 * pdexSigmaValuesFD[3 * igrid + 2] +
                           pdexSigmaValuesFD[3 * igrid + 1]);

                    pdecGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        gradValuesSpinUpFD[3 * igrid + idim] *
                          (2.0 * pdecSigmaValuesFD[3 * igrid] +
                           pdecSigmaValuesFD[3 * igrid + 1]) +
                        gradValuesSpinDownFD[3 * igrid + idim] *
                          (2.0 * pdecSigmaValuesFD[3 * igrid + 2] +
                           pdecSigmaValuesFD[3 * igrid + 1]);

                    pdexLapDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdexLaplacianValuesFD[2 * igrid + 1];

                    pdecLapDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdecLaplacianValuesFD[2 * igrid + 1];
                  }
              } // stencil grid filling loop

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdexGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdexGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdecGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdecGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdexGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdexGradDensitySpinDown[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdecGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdecGradDensitySpinDown[idim][0]));


            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdexLapDensityidimSpinUpStencil[0]),
              &(laplacianTermsPdexLapDensitySpinUp[idim][0]));

            /*
            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
                    d_vxcDivergenceTermFDStencilSize,
                    &(d_spacingFDStencil[0]),
                    nquad,
                    &(pdecLapDensityidimSpinUpStencil[0]),
                    &(laplacianTermsPdecLapDensitySpinUp[idim][0]));
            */

            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              &(d_spacingFDStencil[0]),
              nquad,
              &(pdexLapDensityidimSpinDownStencil[0]),
              &(laplacianTermsPdexLapDensitySpinDown[idim][0]));

            /*
            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
                    d_vxcDivergenceTermFDStencilSize,
                    &(d_spacingFDStencil[0]),
                    nquad,
                    &(pdecLapDensityidimSpinDownStencil[0]),
                    &(laplacianTermsPdecLapDensitySpinDown[idim][0]));
            */

          } // dim loop

        for (size_t igrid = 0; igrid < nquad; igrid++)
          {
            vxValuesSpinUp[igrid] =
              pdexDensitySpinUpValues[igrid] -
              (divergenceTermsPdexGradDensitySpinUp[0][igrid] +
               divergenceTermsPdexGradDensitySpinUp[1][igrid] +
               divergenceTermsPdexGradDensitySpinUp[2][igrid]) +
              (laplacianTermsPdexLapDensitySpinUp[0][igrid] +
               laplacianTermsPdexLapDensitySpinUp[1][igrid] +
               laplacianTermsPdexLapDensitySpinUp[2][igrid]);

            vcValuesSpinUp[igrid] =
              pdecDensitySpinUpValues[igrid] -
              (divergenceTermsPdecGradDensitySpinUp[0][igrid] +
               divergenceTermsPdecGradDensitySpinUp[1][igrid] +
               divergenceTermsPdecGradDensitySpinUp[2][igrid]) +
              (laplacianTermsPdecLapDensitySpinUp[0][igrid] +
               laplacianTermsPdecLapDensitySpinUp[1][igrid] +
               laplacianTermsPdecLapDensitySpinUp[2][igrid]);


            vxValuesSpinDown[igrid] =
              pdexDensitySpinDownValues[igrid] -
              (divergenceTermsPdexGradDensitySpinDown[0][igrid] +
               divergenceTermsPdexGradDensitySpinDown[1][igrid] +
               divergenceTermsPdexGradDensitySpinDown[2][igrid]) +
              (laplacianTermsPdexLapDensitySpinDown[0][igrid] +
               laplacianTermsPdexLapDensitySpinDown[1][igrid] +
               laplacianTermsPdexLapDensitySpinDown[2][igrid]);

            vcValuesSpinDown[igrid] =
              pdecDensitySpinDownValues[igrid] -
              (divergenceTermsPdecGradDensitySpinDown[0][igrid] +
               divergenceTermsPdecGradDensitySpinDown[1][igrid] +
               divergenceTermsPdecGradDensitySpinDown[2][igrid]) +
              (laplacianTermsPdecLapDensitySpinDown[0][igrid] +
               laplacianTermsPdecLapDensitySpinDown[1][igrid] +
               laplacianTermsPdecLapDensitySpinDown[2][igrid]);
          }
      } // VxcCompute check
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLLMGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      &                                                                src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const unsigned int inputVecSize,
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLLMGGAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double> &                           kPointWeights)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLLMGGAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double> &                           kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityLLMGGAClass<memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityLLMGGAClass<
    memorySpace>::getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityLLMGGAClass<memorySpace>::reinitKPointDependentVariables(
    unsigned int kPointIndex)
  {}

  template class excDensityLLMGGAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityLLMGGAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
