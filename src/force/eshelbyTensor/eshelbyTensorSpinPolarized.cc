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
// @author Sambit Das (2017)
//
#include <dftUtils.h>
#include <eshelbyTensorSpinPolarized.h>

namespace dftfe
{
  namespace eshelbyTensorSP
  {
    Tensor<2, 3, VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const VectorizedArray<double> &              rho,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoSpin1,
      const VectorizedArray<double> &              exc,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin1)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        -outer_product(derExcGradRhoSpin0, gradRhoSpin0) -
        outer_product(derExcGradRhoSpin1, gradRhoSpin1);
      VectorizedArray<double> identityTensorFactor = exc * rho;
      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
      getELocWfcEshelbyTensorPeriodicKPoints(
        dealii::AlignedVector<
          Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin0Begin,
        dealii::AlignedVector<
          Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin1Begin,
        dealii::AlignedVector<
          Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
          gradPsiSpin0Begin,
        dealii::AlignedVector<
          Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                                gradPsiSpin1Begin,
        const std::vector<double> &             kPointCoordinates,
        const std::vector<double> &             kPointWeights,
        const std::vector<std::vector<double>> &eigenValues_,
        const double                            fermiEnergy_,
        const double                            fermiEnergyUp_,
        const double                            fermiEnergyDown_,
        const double                            tVal,
        const bool                              constraintMagnetization)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          eshelbyTensor[idim][jdim] = make_vectorized_array(0.0);

      VectorizedArray<double> identityTensorFactor = make_vectorized_array(0.0);

      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin0 =
        psiSpin0Begin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        it2Spin0 = gradPsiSpin0Begin;
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin1 =
        psiSpin1Begin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                         it2Spin1       = gradPsiSpin1Begin;
      const unsigned int numEigenValues = eigenValues_[0].size() / 2;

      Tensor<1, 3, VectorizedArray<double>> kPointCoord;
      for (unsigned int ik = 0; ik < kPointWeights.size(); ++ik)
        {
          kPointCoord[0] = make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
          kPointCoord[1] = make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
          kPointCoord[2] = make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
          for (unsigned int eigenIndex = 0; eigenIndex < numEigenValues;
               ++it1Spin0, ++it2Spin0, ++it1Spin1, ++it2Spin1, ++eigenIndex)
            {
              const Tensor<1, 2, VectorizedArray<double>> &psiSpin0 = *it1Spin0;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &gradPsiSpin0                                       = *it2Spin0;
              const Tensor<1, 2, VectorizedArray<double>> &psiSpin1 = *it1Spin1;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &gradPsiSpin1 = *it2Spin1;

              double partOccSpin0 = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex], fermiEnergy_, C_kb, tVal);
              double partOccSpin1 = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex + numEigenValues],
                fermiEnergy_,
                C_kb,
                tVal);

              if (constraintMagnetization)
                {
                  partOccSpin0 = 1.0, partOccSpin1 = 1.0;
                  if (eigenValues_[ik][eigenIndex + numEigenValues] >
                      fermiEnergyDown_)
                    partOccSpin1 = 0.0;
                  if (eigenValues_[ik][eigenIndex] > fermiEnergyUp_)
                    partOccSpin0 = 0.0;
                }

              VectorizedArray<double> identityTensorFactorContributionSpin0 =
                make_vectorized_array(0.0);
              VectorizedArray<double> identityTensorFactorContributionSpin1 =
                make_vectorized_array(0.0);
              const VectorizedArray<double> fnkSpin0 =
                make_vectorized_array(partOccSpin0 * kPointWeights[ik]);
              const VectorizedArray<double> fnkSpin1 =
                make_vectorized_array(partOccSpin1 * kPointWeights[ik]);

              identityTensorFactorContributionSpin0 +=
                (scalar_product(gradPsiSpin0[0], gradPsiSpin0[0]) +
                 scalar_product(gradPsiSpin0[1], gradPsiSpin0[1]));
              identityTensorFactorContributionSpin0 +=
                make_vectorized_array(2.0) *
                (psiSpin0[0] * scalar_product(kPointCoord, gradPsiSpin0[1]) -
                 psiSpin0[1] * scalar_product(kPointCoord, gradPsiSpin0[0]));
              identityTensorFactorContributionSpin0 +=
                (scalar_product(kPointCoord, kPointCoord) -
                 make_vectorized_array(2.0 * eigenValues_[ik][eigenIndex])) *
                (psiSpin0[0] * psiSpin0[0] + psiSpin0[1] * psiSpin0[1]);

              identityTensorFactorContributionSpin1 +=
                (scalar_product(gradPsiSpin1[0], gradPsiSpin1[0]) +
                 scalar_product(gradPsiSpin1[1], gradPsiSpin1[1]));
              identityTensorFactorContributionSpin1 +=
                make_vectorized_array(2.0) *
                (psiSpin1[0] * scalar_product(kPointCoord, gradPsiSpin1[1]) -
                 psiSpin1[1] * scalar_product(kPointCoord, gradPsiSpin1[0]));
              identityTensorFactorContributionSpin1 +=
                (scalar_product(kPointCoord, kPointCoord) -
                 make_vectorized_array(
                   2.0 * eigenValues_[ik][eigenIndex + numEigenValues])) *
                (psiSpin1[0] * psiSpin1[0] + psiSpin1[1] * psiSpin1[1]);

              identityTensorFactor +=
                (identityTensorFactorContributionSpin0 * fnkSpin0 +
                 identityTensorFactorContributionSpin1 * fnkSpin1) *
                make_vectorized_array(0.5);

              eshelbyTensor -=
                fnkSpin0 *
                (outer_product(gradPsiSpin0[0], gradPsiSpin0[0]) +
                 outer_product(gradPsiSpin0[1], gradPsiSpin0[1]) +
                 psiSpin0[0] * outer_product(gradPsiSpin0[1], kPointCoord) -
                 psiSpin0[1] * outer_product(gradPsiSpin0[0], kPointCoord));
              eshelbyTensor -=
                fnkSpin1 *
                (outer_product(gradPsiSpin1[0], gradPsiSpin1[0]) +
                 outer_product(gradPsiSpin1[1], gradPsiSpin1[1]) +
                 psiSpin1[0] * outer_product(gradPsiSpin1[1], kPointCoord) -
                 psiSpin1[1] * outer_product(gradPsiSpin1[0], kPointCoord));
            }
        }

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getELocWfcEshelbyTensorNonPeriodic(
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator
        psiSpin0Begin,
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator
        psiSpin1Begin,
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
        const_iterator gradPsiSpin0Begin,
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
        const_iterator           gradPsiSpin1Begin,
      const std::vector<double> &eigenValues_,
      const double               fermiEnergy_,
      const double               fermiEnergyUp_,
      const double               fermiEnergyDown_,
      const double               tVal,
      const bool                 constraintMagnetization)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          eshelbyTensor[idim][jdim] = make_vectorized_array(0.0);

      VectorizedArray<double> identityTensorFactor = make_vectorized_array(0.0);

      dealii::AlignedVector<VectorizedArray<double>>::const_iterator it1Spin0 =
        psiSpin0Begin;
      dealii::AlignedVector<
        Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2Spin0 =
        gradPsiSpin0Begin;
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator it1Spin1 =
        psiSpin1Begin;
      dealii::AlignedVector<
        Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2Spin1 =
        gradPsiSpin1Begin;
      const unsigned int numEigenValues = eigenValues_.size() / 2;
      for (unsigned int eigenIndex = 0; eigenIndex < numEigenValues;
           ++it1Spin0, ++it2Spin0, ++it1Spin1, ++it2Spin1, ++eigenIndex)
        {
          const VectorizedArray<double> &              psiSpin0     = *it1Spin0;
          const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin0 = *it2Spin0;
          const VectorizedArray<double> &              psiSpin1     = *it1Spin1;
          const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin1 = *it2Spin1;

          double partOccSpin0 = dftUtils::getPartialOccupancy(
            eigenValues_[eigenIndex], fermiEnergy_, C_kb, tVal);

          double partOccSpin1 = dftUtils::getPartialOccupancy(
            eigenValues_[eigenIndex + numEigenValues],
            fermiEnergy_,
            C_kb,
            tVal);

          if (constraintMagnetization)
            {
              partOccSpin0 = 1.0, partOccSpin1 = 1.0;
              if (eigenValues_[eigenIndex + numEigenValues] > fermiEnergyDown_)
                partOccSpin1 = 0.0;
              if (eigenValues_[eigenIndex] > fermiEnergyUp_)
                partOccSpin0 = 0.0;
            }

          identityTensorFactor +=
            make_vectorized_array(0.5 * partOccSpin0) *
              scalar_product(gradPsiSpin0, gradPsiSpin0) -
            make_vectorized_array(partOccSpin0 * eigenValues_[eigenIndex]) *
              psiSpin0 * psiSpin0;
          identityTensorFactor +=
            make_vectorized_array(0.5 * partOccSpin1) *
              scalar_product(gradPsiSpin1, gradPsiSpin1) -
            make_vectorized_array(partOccSpin1 *
                                  eigenValues_[eigenIndex + numEigenValues]) *
              psiSpin1 * psiSpin1;
          eshelbyTensor -= make_vectorized_array(partOccSpin0) *
                           outer_product(gradPsiSpin0, gradPsiSpin0);
          eshelbyTensor -= make_vectorized_array(partOccSpin1) *
                           outer_product(gradPsiSpin1, gradPsiSpin1);
        }

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }



    Tensor<2, 3, VectorizedArray<double>> getEKStress(
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin0Begin,
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin1Begin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        gradPsiSpin0Begin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                              gradPsiSpin1Begin,
      const std::vector<double> &             kPointCoordinates,
      const std::vector<double> &             kPointWeights,
      const std::vector<std::vector<double>> &eigenValues_,
      const double                            fermiEnergy_,
      const double                            fermiEnergyUp_,
      const double                            fermiEnergyDown_,
      const double                            tVal,
      const bool                              constraintMagnetization)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        {
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              eshelbyTensor[idim][jdim] = make_vectorized_array(0.0);
            }
        }

      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin0 =
        psiSpin0Begin;
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin1 =
        psiSpin1Begin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        it2Spin0 = gradPsiSpin0Begin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                         it2Spin1       = gradPsiSpin1Begin;
      const unsigned int numEigenValues = eigenValues_[0].size() / 2;

      Tensor<1, 3, VectorizedArray<double>> kPointCoord;
      for (unsigned int ik = 0; ik < kPointWeights.size(); ++ik)
        {
          kPointCoord[0] = make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
          kPointCoord[1] = make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
          kPointCoord[2] = make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
          for (unsigned int eigenIndex = 0; eigenIndex < numEigenValues;
               ++it1Spin0, ++it1Spin1, ++it2Spin0, ++it2Spin1, ++eigenIndex)
            {
              const Tensor<1, 2, VectorizedArray<double>> &psiSpin0 = *it1Spin0;
              const Tensor<1, 2, VectorizedArray<double>> &psiSpin1 = *it1Spin1;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &gradPsiSpin0 = *it2Spin0;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &gradPsiSpin1 = *it2Spin1;

              double partOccSpin0 = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex], fermiEnergy_, C_kb, tVal);
              double partOccSpin1 = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex + numEigenValues],
                fermiEnergy_,
                C_kb,
                tVal);

              if (constraintMagnetization)
                {
                  partOccSpin0 = 1.0, partOccSpin1 = 1.0;
                  if (eigenValues_[ik][eigenIndex + numEigenValues] >
                      fermiEnergyDown_)
                    partOccSpin1 = 0.0;
                  if (eigenValues_[ik][eigenIndex] > fermiEnergyUp_)
                    partOccSpin0 = 0.0;
                }

              VectorizedArray<double> fnkSpin0 =
                make_vectorized_array(partOccSpin0 * kPointWeights[ik]);
              VectorizedArray<double> fnkSpin1 =
                make_vectorized_array(partOccSpin1 * kPointWeights[ik]);

              eshelbyTensor +=
                fnkSpin0 *
                (psiSpin0[1] * outer_product(kPointCoord, gradPsiSpin0[0]) -
                 psiSpin0[0] * outer_product(kPointCoord, gradPsiSpin0[1]) -
                 outer_product(kPointCoord, kPointCoord) *
                   (psiSpin0[0] * psiSpin0[0] + psiSpin0[1] * psiSpin0[1]));
              eshelbyTensor +=
                fnkSpin1 *
                (psiSpin1[1] * outer_product(kPointCoord, gradPsiSpin1[0]) -
                 psiSpin1[0] * outer_product(kPointCoord, gradPsiSpin1[1]) -
                 outer_product(kPointCoord, kPointCoord) *
                   (psiSpin1[0] * psiSpin1[0] + psiSpin1[1] * psiSpin1[1]));
            }
        }

      return eshelbyTensor;
    }

    // multiple k point and complex mode
    Tensor<2, 3, VectorizedArray<double>>
    getEnlStress(
      const dealii::AlignedVector<dealii::AlignedVector<dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>>>
        &zetalmDeltaVlProductDistImageAtoms,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiSpin0TimesVTimesPartOcc,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiSpin1TimesVTimesPartOcc,
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin0Begin,
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiSpin1Begin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        gradPsiSpin0Begin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                       gradPsiSpin1Begin,
      const std::vector<double> &      kPointWeights,
      const std::vector<double> &      kPointCoordinates,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList,
      const unsigned int               numBlockedEigenvectors)
    {
      Tensor<2, 3, VectorizedArray<double>> E;
      VectorizedArray<double>               two = make_vectorized_array(2.0);

      for (unsigned int iAtomNonLocal = 0;
           iAtomNonLocal < zetalmDeltaVlProductDistImageAtoms.size();
           ++iAtomNonLocal)
        {
          bool isCellInCompactSupport = false;
          for (unsigned int i = 0; i < nonlocalAtomsCompactSupportList.size();
               i++)
            if (nonlocalAtomsCompactSupportList[i] == iAtomNonLocal)
              {
                isCellInCompactSupport = true;
                break;
              }

          if (!isCellInCompactSupport)
            continue;

          const int numberPseudoWaveFunctions =
            zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal].size();
          const int numKPoints = kPointWeights.size();

          dealii::AlignedVector<
            Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin0 =
            psiSpin0Begin;
          dealii::AlignedVector<
            Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1Spin1 =
            psiSpin1Begin;
          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
            it2Spin0 = gradPsiSpin0Begin;
          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
            it2Spin1 = gradPsiSpin1Begin;

          Tensor<1, 3, VectorizedArray<double>> kPointCoord;
          for (unsigned int ik = 0; ik < numKPoints; ++ik)
            {
              kPointCoord[0] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
              kPointCoord[1] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
              kPointCoord[2] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
              const VectorizedArray<double> fnkSpin0 =
                make_vectorized_array(kPointWeights[ik]);
              const VectorizedArray<double> fnkSpin1 =
                make_vectorized_array(kPointWeights[ik]);
              for (unsigned int eigenIndex = 0;
                   eigenIndex < numBlockedEigenvectors;
                   ++it1Spin0, ++it1Spin1, ++it2Spin0, ++it2Spin1, ++eigenIndex)
                {
                  const Tensor<1, 2, VectorizedArray<double>> &psiSpin0 =
                    *it1Spin0;
                  const Tensor<1, 2, VectorizedArray<double>> &psiSpin1 =
                    *it1Spin1;
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsiSpin0 = *it2Spin0;
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsiSpin1 = *it2Spin1;
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
                      const VectorizedArray<double> CRealSpin0 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin0TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .real());
                      const VectorizedArray<double> CImagSpin0 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin0TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .imag());
                      const VectorizedArray<double> CRealSpin1 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin1TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .real());
                      const VectorizedArray<double> CImagSpin1 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin1TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .imag());
                      const Tensor<1, 3, VectorizedArray<double>> zdvR =
                        zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                          [iPseudoWave][ik][0];
                      const Tensor<1, 3, VectorizedArray<double>> zdvI =
                        zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                          [iPseudoWave][ik][1];
                      E -= two * fnkSpin0 *
                           ((outer_product(gradPsiSpin0[0], zdvR) +
                             outer_product(gradPsiSpin0[1], zdvI)) *
                              CRealSpin0 -
                            (outer_product(gradPsiSpin0[0], zdvI) -
                             outer_product(gradPsiSpin0[1], zdvR)) *
                              CImagSpin0 +
                            outer_product(
                              ((-psiSpin0[1] * zdvR + psiSpin0[0] * zdvI) *
                                 CRealSpin0 +
                               (psiSpin0[0] * zdvR + psiSpin0[1] * zdvI) *
                                 CImagSpin0),
                              kPointCoord));
                      E -= two * fnkSpin1 *
                           ((outer_product(gradPsiSpin1[0], zdvR) +
                             outer_product(gradPsiSpin1[1], zdvI)) *
                              CRealSpin1 -
                            (outer_product(gradPsiSpin1[0], zdvI) -
                             outer_product(gradPsiSpin1[1], zdvR)) *
                              CImagSpin1 +
                            outer_product(
                              ((-psiSpin1[1] * zdvR + psiSpin1[0] * zdvI) *
                                 CRealSpin1 +
                               (psiSpin1[0] * zdvR + psiSpin1[1] * zdvI) *
                                 CImagSpin1),
                              kPointCoord));
                    }
                }
            }
        }
      return E;
    }

    // Gamma point case
    Tensor<2, 3, VectorizedArray<double>>
    getEnlStress(
      const dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>>
        &zetalmDeltaVlProductDistImageAtoms,
      const std::vector<std::vector<std::vector<double>>>
        &projectorKetTimesPsiSpin0TimesVTimesPartOcc,
      const std::vector<std::vector<std::vector<double>>>
        &projectorKetTimesPsiSpin1TimesVTimesPartOcc,
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator
        psiSpin0Begin,
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator
        psiSpin1Begin,
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
        const_iterator gradPsiSpin0Begin,
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
        const_iterator                 gradPsiSpin1Begin,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList,
      const unsigned int               numBlockedEigenvectors)
    {
      Tensor<2, 3, VectorizedArray<double>> E;
      VectorizedArray<double>               two = make_vectorized_array(2.0);

      for (unsigned int iAtomNonLocal = 0;
           iAtomNonLocal < zetalmDeltaVlProductDistImageAtoms.size();
           ++iAtomNonLocal)
        {
          bool isCellInCompactSupport = false;
          for (unsigned int i = 0; i < nonlocalAtomsCompactSupportList.size();
               i++)
            if (nonlocalAtomsCompactSupportList[i] == iAtomNonLocal)
              {
                isCellInCompactSupport = true;
                break;
              }

          if (!isCellInCompactSupport)
            continue;

          const int numberPseudoWaveFunctions =
            zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal].size();

          dealii::AlignedVector<VectorizedArray<double>>::const_iterator
            it1Spin0 = psiSpin0Begin;
          dealii::AlignedVector<VectorizedArray<double>>::const_iterator
            it1Spin1 = psiSpin1Begin;
          dealii::AlignedVector<
            Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2Spin0 =
            gradPsiSpin0Begin;
          dealii::AlignedVector<
            Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2Spin1 =
            gradPsiSpin1Begin;

          for (unsigned int eigenIndex = 0; eigenIndex < numBlockedEigenvectors;
               ++it1Spin0, ++it1Spin1, ++it2Spin0, ++it2Spin1, ++eigenIndex)
            {
              const VectorizedArray<double> &              psiSpin0 = *it1Spin0;
              const VectorizedArray<double> &              psiSpin1 = *it1Spin1;
              const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin0 =
                *it2Spin0;
              const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin1 =
                *it2Spin1;
              for (unsigned int iPseudoWave = 0;
                   iPseudoWave < numberPseudoWaveFunctions;
                   ++iPseudoWave)
                {
                  const VectorizedArray<double> CRealSpin0 =
                    make_vectorized_array(
                      projectorKetTimesPsiSpin0TimesVTimesPartOcc
                        [0][iAtomNonLocal]
                        [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]);
                  const VectorizedArray<double> CRealSpin1 =
                    make_vectorized_array(
                      projectorKetTimesPsiSpin1TimesVTimesPartOcc
                        [0][iAtomNonLocal]
                        [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]);
                  const Tensor<1, 3, VectorizedArray<double>> zdvR =
                    zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                      [iPseudoWave];
                  E -= two * ((outer_product(gradPsiSpin0, zdvR)) * CRealSpin0);
                  E -= two * ((outer_product(gradPsiSpin1, zdvR)) * CRealSpin1);
                }
            }
        }
      return E;
    }

    Tensor<1, 3, VectorizedArray<double>>
    getFnl(
      const dealii::AlignedVector<dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
        &zetaDeltaV,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiSpin0TimesVTimesPartOcc,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiSpin1TimesVTimesPartOcc,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        gradPsiSpin0Begin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                       gradPsiSpin1Begin,
      const std::vector<double> &      kPointWeights,
      const unsigned int               numBlockedEigenvectors,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> Fnl = zeroTensor;
      VectorizedArray<double>               two = make_vectorized_array(2.0);

      for (unsigned int iAtomNonLocal = 0; iAtomNonLocal < zetaDeltaV.size();
           ++iAtomNonLocal)
        {
          bool isCellInCompactSupport = false;
          for (unsigned int i = 0; i < nonlocalAtomsCompactSupportList.size();
               i++)
            if (nonlocalAtomsCompactSupportList[i] == iAtomNonLocal)
              {
                isCellInCompactSupport = true;
                break;
              }

          if (!isCellInCompactSupport)
            continue;

          const int numberPseudoWaveFunctions =
            zetaDeltaV[iAtomNonLocal].size();
          const int numKPoints = kPointWeights.size();

          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
            it1Spin0 = gradPsiSpin0Begin;
          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
            it1Spin1 = gradPsiSpin1Begin;
          for (unsigned int ik = 0; ik < numKPoints; ++ik)
            {
              Tensor<1, 3, VectorizedArray<double>> tempF = zeroTensor;
              VectorizedArray<double>       tempE = make_vectorized_array(0.0);
              const VectorizedArray<double> fnkSpin0 =
                make_vectorized_array(kPointWeights[ik]);
              const VectorizedArray<double> fnkSpin1 =
                make_vectorized_array(kPointWeights[ik]);
              for (unsigned int eigenIndex = 0;
                   eigenIndex < numBlockedEigenvectors;
                   ++it1Spin0, ++it1Spin1, ++eigenIndex)
                {
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsiSpin0 = *it1Spin0;
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsiSpin1 = *it1Spin1;
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
                      const VectorizedArray<double> CRealSpin0 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin0TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .real());
                      const VectorizedArray<double> CImagSpin0 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin0TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .imag());
                      const VectorizedArray<double> CRealSpin1 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin1TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .real());
                      const VectorizedArray<double> CImagSpin1 =
                        make_vectorized_array(
                          projectorKetTimesPsiSpin1TimesVTimesPartOcc
                            [ik][iAtomNonLocal]
                            [numberPseudoWaveFunctions * eigenIndex +
                             iPseudoWave]
                              .imag());

                      const VectorizedArray<double> zdvR =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
                      const VectorizedArray<double> zdvI =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];

                      tempF +=
                        fnkSpin0 *
                        ((gradPsiSpin0[0] * zdvR + gradPsiSpin0[1] * zdvI) *
                           CRealSpin0 -
                         (gradPsiSpin0[0] * zdvI - gradPsiSpin0[1] * zdvR) *
                           CImagSpin0);
                      tempF +=
                        fnkSpin1 *
                        ((gradPsiSpin1[0] * zdvR + gradPsiSpin1[1] * zdvI) *
                           CRealSpin1 -
                         (gradPsiSpin1[0] * zdvI - gradPsiSpin1[1] * zdvR) *
                           CImagSpin1);
                    }
                }
              Fnl -= two * tempF;
            }
        }
      return Fnl;
    }

    Tensor<1, 3, VectorizedArray<double>>
    getFnl(const dealii::AlignedVector<
             dealii::AlignedVector<VectorizedArray<double>>> &zetaDeltaV,
           const std::vector<std::vector<double>>
             &projectorKetTimesPsiSpin0TimesVTimesPartOcc,
           const std::vector<std::vector<double>>
             &projectorKetTimesPsiSpin1TimesVTimesPartOcc,
           dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
             const_iterator gradPsiSpin0Begin,
           dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
             const_iterator                 gradPsiSpin1Begin,
           const unsigned int               numBlockedEigenvectors,
           const std::vector<unsigned int> &nonlocalAtomsCompactSupportList)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> Fnl = zeroTensor;
      VectorizedArray<double>               two = make_vectorized_array(2.0);

      for (unsigned int iAtomNonLocal = 0; iAtomNonLocal < zetaDeltaV.size();
           ++iAtomNonLocal)
        {
          bool isCellInCompactSupport = false;
          for (unsigned int i = 0; i < nonlocalAtomsCompactSupportList.size();
               i++)
            if (nonlocalAtomsCompactSupportList[i] == iAtomNonLocal)
              {
                isCellInCompactSupport = true;
                break;
              }

          if (!isCellInCompactSupport)
            continue;

          const int numberPseudoWaveFunctions =
            zetaDeltaV[iAtomNonLocal].size();
          const std::vector<double>
            &projectorKetTimesPsiSpin0TimesVTimesPartOccAtom =
              projectorKetTimesPsiSpin0TimesVTimesPartOcc[iAtomNonLocal];
          const std::vector<double>
            &projectorKetTimesPsiSpin1TimesVTimesPartOccAtom =
              projectorKetTimesPsiSpin1TimesVTimesPartOcc[iAtomNonLocal];
          const dealii::AlignedVector<VectorizedArray<double>> &zetaDeltaVAtom =
            zetaDeltaV[iAtomNonLocal];
          Tensor<1, 3, VectorizedArray<double>> tempF = zeroTensor;

          dealii::AlignedVector<
            Tensor<1, 3, VectorizedArray<double>>>::const_iterator it1Spin0 =
            gradPsiSpin0Begin;
          dealii::AlignedVector<
            Tensor<1, 3, VectorizedArray<double>>>::const_iterator it1Spin1 =
            gradPsiSpin1Begin;
          for (unsigned int eigenIndex = 0; eigenIndex < numBlockedEigenvectors;
               ++it1Spin0, ++it1Spin1, ++eigenIndex)
            {
              const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin0 =
                *it1Spin0;
              const Tensor<1, 3, VectorizedArray<double>> &gradPsiSpin1 =
                *it1Spin1;
              for (unsigned int iPseudoWave = 0;
                   iPseudoWave < numberPseudoWaveFunctions;
                   ++iPseudoWave)
                {
                  tempF += (make_vectorized_array(
                              projectorKetTimesPsiSpin0TimesVTimesPartOccAtom
                                [numberPseudoWaveFunctions * eigenIndex +
                                 iPseudoWave]) *
                              gradPsiSpin0 +
                            make_vectorized_array(
                              projectorKetTimesPsiSpin1TimesVTimesPartOccAtom
                                [numberPseudoWaveFunctions * eigenIndex +
                                 iPseudoWave]) *
                              gradPsiSpin1) *
                           zetaDeltaVAtom[iPseudoWave];
                }
            }
          Fnl -= two * tempF;
        }
      return Fnl;
    }

    Tensor<1, 3, VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const VectorizedArray<double> &              vxcSpin0,
      const VectorizedArray<double> &              vxcSpin1,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin1,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoCore,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoCore,
      const bool                                   isXCGGA)
    {
      Tensor<1, 3, VectorizedArray<double>> temp;
      for (unsigned int i = 0; i < 3; i++)
        temp[i] = make_vectorized_array(0.0);

      if (isXCGGA)
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              temp[i] += (derExcGradRhoSpin0[j] + derExcGradRhoSpin1[j]) *
                         hessianRhoCore[j][i];
        }

      temp += (vxcSpin0 + vxcSpin1) * gradRhoCore;
      return temp;
    }
  } // namespace eshelbyTensorSP

} // namespace dftfe
