//
// Created by Arghadwip Paul, Sambit Das
//
#include "AuxDensityMatrixAtomicBasis.h"
#include "SlaterBasis.h"
#include "GaussianBasis.h"
#include <stdexcept>
#include <cmath>
// #  include <linearAlgebraOperationsCPU.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::reinit(
    const AtomicBasis::BasisType                                    basisType,
    const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
    const std::unordered_map<std::string, std::string> &atomBasisFileNames,
    const int                                           nSpin,
    const int                                           maxDerOrder)
  {
    if (basisType == AtomicBasis::BasisType::SLATER)
      d_atomicBasisPtr = std::make_unique<SlaterBasis>();
    else if (basisType == AtomicBasis::BasisType::GAUSSIAN)
      // d_atomicBasisPtr= std::make_unique<GaussianBasis>();

      d_atomicBasisPtr->constructBasisSet(atomCoords, atomBasisFileNames);
    d_nBasis      = d_atomicBasisPtr->getNumBasis();
    d_nSpin       = nSpin;
    d_maxDerOrder = maxDerOrder;
    d_DM.assign(d_nSpin * d_nBasis * d_nBasis, 0.0);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::evalOverlapMatrixStart(
    const std::vector<double> &quadpts,
    const std::vector<double> &quadWt)
  {
    d_atomicBasisData.evalBasisData(quadpts, *d_atomicBasisPtr, 0);
    d_SMatrix = std::vector<double>(d_nBasis * d_nBasis, 0.0);
    const std::vector<double> &basisValues = d_atomicBasisData.getBasisValues();
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = i; j < d_nBasis; ++j)
          {
            double sum = 0.0;
            for (int iQuad = 0; iQuad < quadWt.size(); iQuad++)
              {
                sum += basisValues[iQuad * d_nBasis + i] *
                       basisValues[iQuad * d_nBasis + j] * quadWt[iQuad];
              }
            d_SMatrix[i * d_nBasis + j] = sum;
            if (i != j)
              {
                d_SMatrix[j * d_nBasis + i] = sum;
              }
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::evalOverlapMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    // MPI All Reduce
    MPI_Allreduce(d_SMatrix.data(),
                  d_SMatrix.data(),
                  d_SMatrix.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiComm);
    evalOverlapMatrixInv();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::evalOverlapMatrixInv()
  {
    d_SMatrixInv = d_SMatrix;
    // Invert using Cholesky
    int info;
    // Compute the Cholesky factorization of SMatrix
    dftfe::dpotrf_("L",
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   d_SMatrixInv.data(),
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   &info);

    if (info != 0)
      {
        throw std::runtime_error("Error in dpotrf."
                                 "Cholesky Factorization failed."
                                 " Matrix may not be positive definite.");
      }

    // Compute the inverse of SMatrix using the Cholesky factorization
    dftfe::dpotri_("L",
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   d_SMatrixInv.data(),
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   &info);
    if (info != 0)
      {
        throw std::runtime_error("Error in dpotri."
                                 "Inversion using Cholesky Factors failed."
                                 "Matrix may not be positive definite.");
      }

    // Copy the upper triangular part to the lower triangular part
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = i + 1; j < d_nBasis; ++j)
          {
            d_SMatrixInv[j * d_nBasis + i] = d_SMatrixInv[i * d_nBasis + j];
          }
      }

    // Multiply SMatrix and SMatrixInv using dgemm_
    double              alpha = 1.0;
    double              beta  = 0.0;
    std::vector<double> CMatrix(d_nBasis * d_nBasis, 0.0);
    dftfe::dgemm_("N",
                  "N",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &alpha,
                  d_SMatrix.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  d_SMatrixInv.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  CMatrix.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));



    double frobenius_norm = 0.0;
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = 0; j < d_nBasis; j++)
          {
            if (i == j)
              {
                frobenius_norm += (CMatrix[i * d_nBasis + j] - 1.0) *
                                  (CMatrix[i * d_nBasis + j] - 1.0);
              }
            else
              {
                frobenius_norm +=
                  CMatrix[i * d_nBasis + j] * CMatrix[i * d_nBasis + j];
              }
          }
      }
    frobenius_norm = std::sqrt(frobenius_norm / d_nBasis);
    if (frobenius_norm > 1E-8)
      {
        std::cout << "frobenius norm of inversion : " << frobenius_norm
                  << std::endl;
        throw std::runtime_error("SMatrix Inversion"
                                 "using Cholesky Factors failed"
                                 "to pass check!");
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::vector<double> &
  AuxDensityMatrixAtomicBasis<memorySpace>::getOverlapMatrixInv()
  {
    return d_SMatrixInv;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::projectDensityMatrixStart(
    const std::unordered_map<std::string, std::vector<dataTypes::number>>
      &projectionInputsDataType,
    const std::unordered_map<std::string, std::vector<double>>
      &       projectionInputsReal,
    const int iSpin)
  {
// FIXME: extend implementation to complex datatype
#ifndef USE_COMPLEX
    // eval <AtomicBasis|WFC> matrix d_basisWFCInnerProducts
    d_iSpin       = iSpin;
    auto &psiFunc = projectionInputsDataType.find("psiFunc")->second;
    auto &quadpts = projectionInputsReal.find("quadpts")->second;
    auto &quadWt  = projectionInputsReal.find("quadWt")->second;
    d_fValues     = projectionInputsReal.find("fValues")->second;
    d_atomicBasisData.evalBasisData(quadpts, *d_atomicBasisPtr, 0);
    std::vector<double> basisValsWeighted = d_atomicBasisData.getBasisValues();

    d_nQuad                 = quadWt.size();
    d_nWFC                  = psiFunc.size() / d_nQuad;
    d_basisWFCInnerProducts = std::vector<double>(d_nBasis * d_nWFC, 0.0);


    for (int i = 0; i < d_nQuad; ++i)
      {
        for (int j = 0; j < d_nBasis; ++j)
          {
            basisValsWeighted[i * d_nBasis + j] *= quadWt[i];
          }
      }

    double alpha = 1.0;
    double beta  = 0.0;
    dftfe::dgemm_("N",
                  "T",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  reinterpret_cast<const unsigned int *>(&d_nQuad),
                  &alpha,
                  basisValsWeighted.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  psiFunc.data(),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  &beta,
                  d_basisWFCInnerProducts.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::projectDensityMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    // MPI All Reduce d_basisWFCInnerProducts
    MPI_Allreduce(d_basisWFCInnerProducts.data(),
                  d_basisWFCInnerProducts.data(),
                  d_basisWFCInnerProducts.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiComm);

    auto overlapInv = this->getOverlapMatrixInv();
    // Multiply S^(-1) * <S|W> = BB
    std::vector<double> BB(d_nBasis * d_nWFC, 0.0);
    double              alpha = 1.0;
    double              beta  = 0.0;
    dftfe::dgemm_("N",
                  "N",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &alpha,
                  overlapInv.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  d_basisWFCInnerProducts.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  BB.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));

    // BF = BB * F^(1/2)
    std::vector<double> BF = BB;
    for (int i = 0; i < d_nWFC; i++)
      {
        for (int j = 0; j < d_nBasis; j++)
          {
            BF[i * d_nBasis + j] *= std::sqrt(d_fValues[i]);
          }
      }

    // D = BF * BF^T
    dftfe::dgemm_("N",
                  "T",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  &alpha,
                  BF.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  BF.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  d_DM.data() + d_iSpin * d_nBasis * d_nBasis,
                  reinterpret_cast<const unsigned int *>(&d_nBasis));
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::applyLocalOperations(
    const std::vector<double> &quadpts,
    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      &densityData)
  {
    const unsigned int DMSpinOffset = d_nBasis * d_nBasis;
    const unsigned int nQuad        = quadpts.size() / 3;
    d_atomicBasisData.evalBasisData(quadpts, *d_atomicBasisPtr, d_maxDerOrder);

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        std::vector<double> &rhoUp =
          densityData[DensityDescriptorDataAttributes::valuesSpinUp];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              rhoUp[iQuad] += d_DM[i * d_nBasis + j] *
                              basisValues[iQuad * d_nBasis + i] *
                              basisValues[iQuad * d_nBasis + j];
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        std::vector<double> &rhoDown =
          densityData[DensityDescriptorDataAttributes::valuesSpinDown];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              rhoDown[iQuad] += d_DM[DMSpinOffset + i * d_nBasis + j] *
                                basisValues[iQuad * d_nBasis + i] *
                                basisValues[iQuad * d_nBasis + j];
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        std::vector<double> &rho =
          densityData[DensityDescriptorDataAttributes::valuesTotal];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              rho[iQuad] += (d_DM[i * d_nBasis + j] +
                             d_DM[DMSpinOffset + i * d_nBasis + j]) *
                            basisValues[iQuad * d_nBasis + i] *
                            basisValues[iQuad * d_nBasis + j];
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinUp) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        const std::vector<double> &basisGradValues =
          d_atomicBasisData.getBasisGradValues();

        std::vector<double> &gradRhoUp =
          densityData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              for (int derIndex = 0; derIndex < 3; derIndex++)
                gradRhoUp[3 * iQuad + derIndex] +=
                  d_DM[i * d_nBasis + j] *
                  (basisGradValues[iQuad * d_nBasis * 3 + 3 * i + derIndex] *
                     basisValues[iQuad * d_nBasis + j] +
                   basisGradValues[iQuad * d_nBasis * 3 + 3 * j + derIndex] *
                     basisValues[iQuad * d_nBasis + i]);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinDown) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        const std::vector<double> &basisGradValues =
          d_atomicBasisData.getBasisGradValues();

        std::vector<double> &gradRhoDown =
          densityData[DensityDescriptorDataAttributes::gradValuesSpinDown];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              for (int derIndex = 0; derIndex < 3; derIndex++)
                gradRhoDown[3 * iQuad + derIndex] +=
                  d_DM[DMSpinOffset + i * d_nBasis + j] *
                  (basisGradValues[iQuad * d_nBasis * 3 + 3 * i + derIndex] *
                     basisValues[iQuad * d_nBasis + j] +
                   basisGradValues[iQuad * d_nBasis * 3 + 3 * j + derIndex] *
                     basisValues[iQuad * d_nBasis + i]);
      }


    if (densityData.find(DensityDescriptorDataAttributes::laplacianSpinUp) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        const std::vector<double> &basisGradValues =
          d_atomicBasisData.getBasisGradValues();
        const std::vector<double> &basisLaplacianValues =
          d_atomicBasisData.getBasisLaplacianValues();

        std::vector<double> &laplacianRhoUp =
          densityData[DensityDescriptorDataAttributes::laplacianSpinUp];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              laplacianRhoUp[iQuad] +=
                d_DM[i * d_nBasis + j] *
                (2.0 * (basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 0] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 0] +
                        basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 1] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 1] +
                        basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 2] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 2]) +
                 basisValues[iQuad * d_nBasis + i] *
                   basisLaplacianValues[iQuad * d_nBasis + j] +
                 basisLaplacianValues[iQuad * d_nBasis + i] *
                   basisValues[iQuad * d_nBasis + j]);
      }

    if (densityData.find(DensityDescriptorDataAttributes::laplacianSpinDown) !=
        densityData.end())
      {
        const std::vector<double> &basisValues =
          d_atomicBasisData.getBasisValues();
        const std::vector<double> &basisGradValues =
          d_atomicBasisData.getBasisGradValues();
        const std::vector<double> &basisLaplacianValues =
          d_atomicBasisData.getBasisLaplacianValues();

        std::vector<double> &laplacianRhoDown =
          densityData[DensityDescriptorDataAttributes::laplacianSpinDown];
        for (int iQuad = 0; iQuad < nQuad; iQuad++)
          for (int i = 0; i < d_nBasis; i++)
            for (int j = 0; j < d_nBasis; j++)
              laplacianRhoDown[iQuad] +=
                d_DM[DMSpinOffset + i * d_nBasis + j] *
                (2.0 * (basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 0] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 0] +
                        basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 1] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 1] +
                        basisGradValues[iQuad * d_nBasis * 3 + 3 * i + 2] *
                          basisGradValues[iQuad * d_nBasis * 3 + 3 * j + 2]) +
                 basisValues[iQuad * d_nBasis + i] *
                   basisLaplacianValues[iQuad * d_nBasis + j] +
                 basisLaplacianValues[iQuad * d_nBasis + i] *
                   basisValues[iQuad * d_nBasis + j]);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::projectDensityStart(
    const std::unordered_map<std::string, std::vector<double>>
      &projectionInputs)
  {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixAtomicBasis<memorySpace>::projectDensityEnd(
    const MPI_Comm &mpiComm)
  {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
  }

  template class AuxDensityMatrixAtomicBasis<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class AuxDensityMatrixAtomicBasis<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
