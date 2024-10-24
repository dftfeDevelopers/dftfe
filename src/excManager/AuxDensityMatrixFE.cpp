//
// Created by Sambit Das.
//

#include "AuxDensityMatrixFE.h"
#include <Exceptions.h>
#include <iostream>

namespace dftfe
{
  namespace
  {
    void
    fillDensityAttributeData(
      std::vector<double> &                        attributeData,
      const std::vector<double> &                  values,
      const std::pair<unsigned int, unsigned int> &indexRange)
    {
      unsigned int startIndex = indexRange.first;
      unsigned int endIndex   = indexRange.second;

      attributeData.resize(endIndex - startIndex);
      if (startIndex > endIndex || endIndex > values.size())
        {
          std::cout << "CHECK1: " << startIndex << std::endl;
          std::cout << "CHECK1: " << endIndex << std::endl;
          throw std::invalid_argument("Invalid index range for densityData");
        }

      for (unsigned int i = startIndex; i < endIndex; ++i)
        {
          attributeData[i - startIndex] = values[i];
        }
    }
  } // namespace


  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::applyLocalOperations(
    const std::vector<double> &points,
    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      &densityData)
  {
    std::pair<unsigned int, unsigned int> indexRangeVal;
    std::pair<unsigned int, unsigned int> indexRangeGrad;

    unsigned int minIndex = 0;
    for (unsigned int i = 0; i < d_quadWeightsAll.size(); i++)
      {
        if ((std::abs(points[0] - d_quadPointsAll[3 * i + 0]) +
             std::abs(points[1] - d_quadPointsAll[3 * i + 1]) +
             std::abs(points[2] - d_quadPointsAll[3 * i + 2])) < 1e-6)
          {
            minIndex = i;
            break;
          }
      }


    indexRangeVal.first  = minIndex;
    indexRangeVal.second = minIndex + points.size() / 3;

    indexRangeGrad.first  = minIndex * 3;
    indexRangeGrad.second = minIndex * 3 + points.size();

    if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesTotal],
          d_densityValsTotalAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinUp],
          d_densityValsSpinUpAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinDown],
          d_densityValsSpinDownAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinUp) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValuesSpinUp],
          d_gradDensityValsSpinUpAllQuads,
          indexRangeGrad);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinDown) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValuesSpinDown],
          d_gradDensityValsSpinDownAllQuads,
          indexRangeGrad);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::setDensityMatrixComponents(
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      &                                     eigenVectorsFlattenedMemSpace,
    const std::vector<std::vector<double>> &fractionalOccupancies)
  {
    d_eigenVectorsFlattenedMemSpacePtr = &(eigenVectorsFlattenedMemSpace);
    d_fractionalOccupancies            = &fractionalOccupancies;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::evalOverlapMatrixStart(
    const std::vector<double> &quadpts,
    const std::vector<double> &quadWt)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::evalOverlapMatrixEnd(const MPI_Comm &mpiComm)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::projectDensityMatrixStart(
    const std::unordered_map<std::string, std::vector<dataTypes::number>>
      &projectionInputsDataType,
    const std::unordered_map<std::string, std::vector<double>>
      &       projectionInputsReal,
    const int iSpin)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::projectDensityMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::projectDensityStart(
    const std::unordered_map<std::string, std::vector<double>>
      &projectionInputs)
  {
    d_quadPointsAll  = projectionInputs.find("quadpts")->second;
    d_quadWeightsAll = projectionInputs.find("quadWt")->second;
    const std::vector<double> &densityVals =
      projectionInputs.find("densityFunc")->second;
    const unsigned int nQ = d_quadWeightsAll.size();
    d_densityValsTotalAllQuads.resize(nQ, 0);
    d_densityValsSpinUpAllQuads.resize(nQ, 0);
    d_densityValsSpinDownAllQuads.resize(nQ, 0);
    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinUpAllQuads[iquad] = densityVals[iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinDownAllQuads[iquad] = densityVals[nQ + iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsTotalAllQuads[iquad] = d_densityValsSpinUpAllQuads[iquad] +
                                          d_densityValsSpinDownAllQuads[iquad];

    if (projectionInputs.find("gradDensityFunc") != projectionInputs.end())
      {
        const std::vector<double> &gradDensityVals =
          projectionInputs.find("gradDensityFunc")->second;
        d_gradDensityValsSpinUpAllQuads.resize(nQ * 3, 0);
        d_gradDensityValsSpinDownAllQuads.resize(nQ * 3, 0);

        for (unsigned int iquad = 0; iquad < nQ; iquad++)
          for (unsigned int idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinUpAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * iquad + idim];

        for (unsigned int iquad = 0; iquad < nQ; iquad++)
          for (unsigned idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinDownAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * nQ + 3 * iquad + idim];
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> *
  AuxDensityMatrixFE<memorySpace>::getDensityMatrixComponents_occupancies()
    const
  {
    return d_fractionalOccupancies;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> *
  AuxDensityMatrixFE<memorySpace>::getDensityMatrixComponents_wavefunctions()
    const
  {
    return d_eigenVectorsFlattenedMemSpacePtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixFE<memorySpace>::projectDensityEnd(const MPI_Comm &mpiComm)
  {}

  template class AuxDensityMatrixFE<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class AuxDensityMatrixFE<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
