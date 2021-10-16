// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
#include "../../../include/eshelbyTensor.h"

#include "../../../include/dftUtils.h"

namespace dftfe
{
  namespace eshelbyTensor
  {
    Tensor<2, 3, VectorizedArray<double>>
    getPhiExtEshelbyTensor(
      const VectorizedArray<double> &              phiExt,
      const Tensor<1, 3, VectorizedArray<double>> &gradPhiExt)
    {
      Tensor<2, 3, VectorizedArray<double>> identityTensor;
      identityTensor[0][0] = make_vectorized_array(1.0);
      identityTensor[1][1] = make_vectorized_array(1.0);
      identityTensor[2][2] = make_vectorized_array(1.0);



      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        make_vectorized_array(1.0 / (4.0 * M_PI)) *
          outer_product(gradPhiExt, gradPhiExt) -
        make_vectorized_array(1.0 / (8.0 * M_PI)) *
          scalar_product(gradPhiExt, gradPhiExt) * identityTensor;

      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getVselfBallEshelbyTensor(
      const Tensor<1, 3, VectorizedArray<double>> &gradVself)
    {
      Tensor<2, 3, VectorizedArray<double>> identityTensor;
      identityTensor[0][0] = make_vectorized_array(1.0);
      identityTensor[1][1] = make_vectorized_array(1.0);
      identityTensor[2][2] = make_vectorized_array(1.0);



      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        make_vectorized_array(1.0 / (8.0 * M_PI)) *
          scalar_product(gradVself, gradVself) * identityTensor -
        make_vectorized_array(1.0 / (4.0 * M_PI)) *
          outer_product(gradVself, gradVself);

      return eshelbyTensor;
    }


    Tensor<2, 3, double>
    getVselfBallEshelbyTensor(const Tensor<1, 3, double> &gradVself)
    {
      double identityTensorFactor =
        1.0 / (8.0 * M_PI) * scalar_product(gradVself, gradVself);
      Tensor<2, 3, double> eshelbyTensor =
        -1.0 / (4.0 * M_PI) * outer_product(gradVself, gradVself);

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;

      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
      getELocWfcEshelbyTensorPeriodicKPoints(
        dealii::AlignedVector<
          Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiBegin,
        dealii::AlignedVector<
          Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                                gradPsiBegin,
        const std::vector<double> &             kPointCoordinates,
        const std::vector<double> &             kPointWeights,
        const std::vector<std::vector<double>> &eigenValues_,
        const double                            fermiEnergy_,
        const double                            tVal)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        {
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              eshelbyTensor[idim][jdim] = make_vectorized_array(0.0);
            }
        }
      VectorizedArray<double> identityTensorFactor = make_vectorized_array(0.0);

      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1 = psiBegin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        it2 = gradPsiBegin;

      Tensor<1, 3, VectorizedArray<double>> kPointCoord;
      for (unsigned int ik = 0; ik < eigenValues_.size(); ++ik)
        {
          kPointCoord[0] = make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
          kPointCoord[1] = make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
          kPointCoord[2] = make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
          for (unsigned int eigenIndex = 0; eigenIndex < eigenValues_[0].size();
               ++it1, ++it2, ++eigenIndex)
            {
              const Tensor<1, 2, VectorizedArray<double>> &psi = *it1;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &          gradPsi = *it2;
              const double partOcc = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex], fermiEnergy_, C_kb, tVal);
              VectorizedArray<double> identityTensorFactorContribution =
                make_vectorized_array(0.0);
              VectorizedArray<double> fnk =
                make_vectorized_array(partOcc * kPointWeights[ik]);
              identityTensorFactorContribution +=
                (scalar_product(gradPsi[0], gradPsi[0]) +
                 scalar_product(gradPsi[1], gradPsi[1]));
              identityTensorFactorContribution +=
                make_vectorized_array(2.0) *
                (psi[0] * scalar_product(kPointCoord, gradPsi[1]) -
                 psi[1] * scalar_product(kPointCoord, gradPsi[0]));
              identityTensorFactorContribution +=
                (scalar_product(kPointCoord, kPointCoord) -
                 make_vectorized_array(2.0 * eigenValues_[ik][eigenIndex])) *
                (psi[0] * psi[0] + psi[1] * psi[1]);
              identityTensorFactorContribution *= fnk;
              identityTensorFactor += identityTensorFactorContribution;

              eshelbyTensor -=
                make_vectorized_array(2.0) * fnk *
                (outer_product(gradPsi[0], gradPsi[0]) +
                 outer_product(gradPsi[1], gradPsi[1]) +
                 psi[0] * outer_product(gradPsi[1], kPointCoord) -
                 psi[1] * outer_product(gradPsi[0], kPointCoord));
            }
        }

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getELocWfcEshelbyTensorNonPeriodic(
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator psiBegin,
      dealii::AlignedVector<
        Tensor<1, 3, VectorizedArray<double>>>::const_iterator gradPsiBegin,
      const std::vector<double> &                              eigenValues_,
      const std::vector<double> &partialOccupancies_)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          eshelbyTensor[idim][jdim] = make_vectorized_array(0.0);

      VectorizedArray<double> identityTensorFactor = make_vectorized_array(0.0);

      dealii::AlignedVector<VectorizedArray<double>>::const_iterator it1 =
        psiBegin;
      dealii::AlignedVector<
        Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2 =
        gradPsiBegin;
      for (unsigned int eigenIndex = 0; eigenIndex < eigenValues_.size();
           ++it1, ++it2, ++eigenIndex)
        {
          const VectorizedArray<double> &              psi     = *it1;
          const Tensor<1, 3, VectorizedArray<double>> &gradPsi = *it2;
          identityTensorFactor +=
            make_vectorized_array(partialOccupancies_[eigenIndex]) *
              scalar_product(gradPsi, gradPsi) -
            make_vectorized_array(2 * partialOccupancies_[eigenIndex] *
                                  eigenValues_[eigenIndex]) *
              psi * psi;
          eshelbyTensor -=
            make_vectorized_array(2.0 * partialOccupancies_[eigenIndex]) *
            outer_product(gradPsi, gradPsi);
        }

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }


    Tensor<2, 3, VectorizedArray<double>>
    getEElectroEshelbyTensor(
      const VectorizedArray<double> &              phiTot,
      const Tensor<1, 3, VectorizedArray<double>> &gradPhiTot,
      const VectorizedArray<double> &              rho)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        make_vectorized_array(1.0 / (4.0 * M_PI)) *
        outer_product(gradPhiTot, gradPhiTot);
      VectorizedArray<double> identityTensorFactor =
        make_vectorized_array(-1.0 / (8.0 * M_PI)) *
          scalar_product(gradPhiTot, gradPhiTot) +
        rho * phiTot;

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const VectorizedArray<double> &              rho,
      const Tensor<1, 3, VectorizedArray<double>> &gradRho,
      const VectorizedArray<double> &              exc,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRho)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        -outer_product(gradRho, derExcGradRho);
      VectorizedArray<double> identityTensorFactor = exc * rho;


      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getShadowPotentialForceRhoDiffXcEshelbyTensor(
      const VectorizedArray<double> &shadowKSRhoMinMinusRho,
      const Tensor<1, 3, VectorizedArray<double>>
        &shadowKSGradRhoMinMinusGradRho,
      const Tensor<1, 3, VectorizedArray<double>> &gradRho,
      const VectorizedArray<double> &              vxc,
      const Tensor<1, 3, VectorizedArray<double>> &derVxcGradRho,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRho,
      const Tensor<2, 3, VectorizedArray<double>> &der2ExcGradRho)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor =
        -outer_product(derVxcGradRho, gradRho) * shadowKSRhoMinMinusRho -
        outer_product(shadowKSGradRhoMinMinusGradRho * der2ExcGradRho,
                      gradRho) -
        outer_product(derExcGradRho, shadowKSGradRhoMinMinusGradRho);
      VectorizedArray<double> identityTensorFactor =
        vxc * shadowKSRhoMinMinusRho +
        scalar_product(derExcGradRho, shadowKSGradRhoMinMinusGradRho);


      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    Tensor<2, 3, VectorizedArray<double>>
    getELocPspEshelbyTensor(const VectorizedArray<double> &rho,
                            const VectorizedArray<double> &pseudoVLoc,
                            const VectorizedArray<double> &phiExt)
    {
      Tensor<2, 3, VectorizedArray<double>> eshelbyTensor;
      VectorizedArray<double>               identityTensorFactor =
        (pseudoVLoc - phiExt) * rho;


      eshelbyTensor[0][0] = identityTensorFactor;
      eshelbyTensor[1][1] = identityTensorFactor;
      eshelbyTensor[2][2] = identityTensorFactor;
      return eshelbyTensor;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFnl(
      const dealii::AlignedVector<dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
        &zetaDeltaV,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiTimesVTimesPartOcc,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                       gradPsiBegin,
      const std::vector<double> &      kPointWeights,
      const unsigned int               numBlockedEigenvectors,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> Fnl  = zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);

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
            it1 = gradPsiBegin;
          for (unsigned int ik = 0; ik < numKPoints; ++ik)
            {
              Tensor<1, 3, VectorizedArray<double>> tempF = zeroTensor;
              VectorizedArray<double>               fnk =
                make_vectorized_array(kPointWeights[ik]);
              for (unsigned int eigenIndex = 0;
                   eigenIndex < numBlockedEigenvectors;
                   ++it1, ++eigenIndex)
                {
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsi = *it1;
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
                      VectorizedArray<double> CReal = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .real());
                      VectorizedArray<double> CImag = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .imag());
                      VectorizedArray<double> zdvR =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
                      VectorizedArray<double> zdvI =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
                      tempF -=
                        ((gradPsi[0] * zdvR + gradPsi[1] * zdvI) * CReal -
                         (gradPsi[0] * zdvI - gradPsi[1] * zdvR) * CImag);
                    }
                }
              Fnl += four * fnk * tempF;
            }
        }
      return Fnl;
    }

    Tensor<1, 3, VectorizedArray<double>>
    getFnl(const dealii::AlignedVector<
             dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>
             &zetaDeltaV,
           const dealii::AlignedVector<
             Tensor<1, 3, Tensor<1, 2, VectorizedArray<double>>>>
             &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
           const std::vector<bool> &        isAtomInCell,
           const std::vector<unsigned int> &nonlocalPseudoWfcsAccum)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> Fnl  = zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);


      for (unsigned int iAtomNonLocal = 0; iAtomNonLocal < zetaDeltaV.size();
           ++iAtomNonLocal)
        {
          if (!isAtomInCell[iAtomNonLocal])
            continue;

          const int numberPseudoWaveFunctions =
            zetaDeltaV[iAtomNonLocal].size();
          const dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>
            &zetaDeltaVAtom = zetaDeltaV[iAtomNonLocal];

          Tensor<1, 3, VectorizedArray<double>> tempF = zeroTensor;
          for (unsigned int iPseudoWave = 0;
               iPseudoWave < numberPseudoWaveFunctions;
               ++iPseudoWave)
            {
              Tensor<1, 3, VectorizedArray<double>>
                pKetPsiContractionGradPsiReal;
              Tensor<1, 3, VectorizedArray<double>>
                pKetPsiContractionGradPsiImag;
              for (unsigned int idim = 0; idim < 3; ++idim)
                {
                  pKetPsiContractionGradPsiReal[idim] =
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                      [nonlocalPseudoWfcsAccum[iAtomNonLocal] + iPseudoWave]
                      [idim][0];
                  pKetPsiContractionGradPsiImag[idim] =
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                      [nonlocalPseudoWfcsAccum[iAtomNonLocal] + iPseudoWave]
                      [idim][1];
                }
              const VectorizedArray<double> zdvR =
                zetaDeltaV[iAtomNonLocal][iPseudoWave][0];
              const VectorizedArray<double> zdvI =
                zetaDeltaV[iAtomNonLocal][iPseudoWave][1];

              tempF -= (pKetPsiContractionGradPsiReal * zdvR -
                        pKetPsiContractionGradPsiImag * zdvI);
            }
          Fnl += four * tempF;
        }
      return Fnl;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFnl(const dealii::AlignedVector<
             dealii::AlignedVector<VectorizedArray<double>>> &zetaDeltaV,
           const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
             &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
           const std::vector<bool> &        isAtomInCell,
           const std::vector<unsigned int> &nonlocalPseudoWfcsAccum)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> Fnl  = zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);

      for (unsigned int iAtomNonLocal = 0; iAtomNonLocal < zetaDeltaV.size();
           ++iAtomNonLocal)
        {
          if (!isAtomInCell[iAtomNonLocal])
            continue;

          const int numberPseudoWaveFunctions =
            zetaDeltaV[iAtomNonLocal].size();
          const dealii::AlignedVector<VectorizedArray<double>> &zetaDeltaVAtom =
            zetaDeltaV[iAtomNonLocal];

          Tensor<1, 3, VectorizedArray<double>> tempF = zeroTensor;
          for (unsigned int iPseudoWave = 0;
               iPseudoWave < numberPseudoWaveFunctions;
               ++iPseudoWave)
            tempF -= projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                       [nonlocalPseudoWfcsAccum[iAtomNonLocal] + iPseudoWave] *
                     zetaDeltaVAtom[iPseudoWave];
          Fnl += four * tempF;
        }
      return Fnl;
    }

    Tensor<1, 3, VectorizedArray<double>>
    getFnlAtom(
      const dealii::AlignedVector<VectorizedArray<double>> &zetaDeltaV,
      const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
      const unsigned int startingId)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> F    = zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);


      const unsigned int numberPseudoWaveFunctions = zetaDeltaV.size();
      for (unsigned int iPseudoWave = 0;
           iPseudoWave < numberPseudoWaveFunctions;
           ++iPseudoWave)
        F -=
          four * zetaDeltaV[iPseudoWave] *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi[startingId +
                                                                   iPseudoWave];

      return F;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFnlAtom(
      const dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>
        &zetaDeltaV,
      const dealii::AlignedVector<
        Tensor<1, 3, Tensor<1, 2, VectorizedArray<double>>>>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
      const dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>
        &projectorKetTimesPsiTimesVTimesPartOccContractionPsi,
      const Tensor<1, 3, VectorizedArray<double>> kcoord,
      const unsigned int                          startingId)
    {
      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        zeroTensor[idim] = make_vectorized_array(0.0);

      Tensor<1, 3, VectorizedArray<double>> F    = zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);


      const unsigned int numberPseudoWaveFunctions = zetaDeltaV.size();
      for (unsigned int iPseudoWave = 0;
           iPseudoWave < numberPseudoWaveFunctions;
           ++iPseudoWave)
        {
          Tensor<1, 3, VectorizedArray<double>> pKetPsiContractionGradPsiReal;
          Tensor<1, 3, VectorizedArray<double>> pKetPsiContractionGradPsiImag;
          for (unsigned int idim = 0; idim < 3; ++idim)
            {
              pKetPsiContractionGradPsiReal[idim] =
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                  [startingId + iPseudoWave][idim][0];
              pKetPsiContractionGradPsiImag[idim] =
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                  [startingId + iPseudoWave][idim][1];
            }

          const VectorizedArray<double> pKetPsiContractionPsiReal =
            projectorKetTimesPsiTimesVTimesPartOccContractionPsi[startingId +
                                                                 iPseudoWave]
                                                                [0];
          const VectorizedArray<double> pKetPsiContractionPsiImag =
            projectorKetTimesPsiTimesVTimesPartOccContractionPsi[startingId +
                                                                 iPseudoWave]
                                                                [1];
          const VectorizedArray<double> zdvR = zetaDeltaV[iPseudoWave][0];
          const VectorizedArray<double> zdvI = zetaDeltaV[iPseudoWave][1];

          F -= four * ((pKetPsiContractionGradPsiReal * zdvR -
                        pKetPsiContractionGradPsiImag * zdvI) +
                       (pKetPsiContractionPsiReal * zdvI +
                        pKetPsiContractionPsiImag * zdvR) *
                         kcoord);
        }

      return F;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFnlAtom(
      const dealii::AlignedVector<dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
        &zetaDeltaV,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiTimesVTimesPartOcc,
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiBegin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                 gradPsiBegin,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      const unsigned int         numBlockedEigenvectors)
    {
      Tensor<1, 3, VectorizedArray<double>> F;
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1 = psiBegin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                            it2  = gradPsiBegin;
      VectorizedArray<double>               four = make_vectorized_array(4.0);
      const int                             numKPoints = kPointWeights.size();
      Tensor<1, 3, VectorizedArray<double>> kcoord;
      for (unsigned int ik = 0; ik < numKPoints; ++ik)
        {
          kcoord[0] = make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
          kcoord[1] = make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
          kcoord[2] = make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
          for (unsigned int eigenIndex = 0; eigenIndex < numBlockedEigenvectors;
               ++it1, ++it2, ++eigenIndex)
            {
              const Tensor<1, 2, VectorizedArray<double>> &psi = *it1;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &                     gradPsi = *it2;
              VectorizedArray<double> fnk =
                make_vectorized_array(kPointWeights[ik]);
              for (unsigned int iAtomNonLocal = 0;
                   iAtomNonLocal < zetaDeltaV.size();
                   ++iAtomNonLocal)
                {
                  const int numberPseudoWaveFunctions =
                    zetaDeltaV[iAtomNonLocal].size();
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
                      VectorizedArray<double> CReal = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .real());
                      VectorizedArray<double> CImag = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .imag());
                      VectorizedArray<double> zdvR =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
                      VectorizedArray<double> zdvI =
                        zetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
                      F -= four * fnk *
                           (((gradPsi[0] * zdvR + gradPsi[1] * zdvI) * CReal -
                             (gradPsi[0] * zdvI - gradPsi[1] * zdvR) * CImag) +
                            ((-psi[1] * zdvR + psi[0] * zdvI) * CReal +
                             (psi[0] * zdvR + psi[1] * zdvI) * CImag) *
                              kcoord);
                    }
                }
            }
        }

      return F;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFPSPLocal(const VectorizedArray<double>                rho,
                 const Tensor<1, 3, VectorizedArray<double>> &gradPseudoVLoc,
                 const Tensor<1, 3, VectorizedArray<double>> &gradPhiExt)

    {
      return rho * (gradPseudoVLoc - gradPhiExt);
    }

    Tensor<1, 3, VectorizedArray<double>>
    getNonSelfConsistentForce(
      const VectorizedArray<double> &              vEffRhoIn,
      const VectorizedArray<double> &              vEffRhoOut,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoOut,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoIn,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoOut,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoOut)
    {
      return (vEffRhoOut - vEffRhoIn) * gradRhoOut +
             (derExchCorrEnergyWithGradRhoOut -
              derExchCorrEnergyWithGradRhoIn) *
               hessianRhoOut;
    }


    Tensor<2, 3, VectorizedArray<double>> getEKStress(
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiBegin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                              gradPsiBegin,
      const std::vector<double> &             kPointCoordinates,
      const std::vector<double> &             kPointWeights,
      const std::vector<std::vector<double>> &eigenValues_,
      const double                            fermiEnergy_,
      const double                            tVal)
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
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1 = psiBegin;
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
        it2 = gradPsiBegin;

      Tensor<1, 3, VectorizedArray<double>> kPointCoord;
      for (unsigned int ik = 0; ik < eigenValues_.size(); ++ik)
        {
          kPointCoord[0] = make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
          kPointCoord[1] = make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
          kPointCoord[2] = make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
          for (unsigned int eigenIndex = 0; eigenIndex < eigenValues_[0].size();
               ++it1, ++it2, ++eigenIndex)
            {
              const Tensor<1, 2, VectorizedArray<double>> &psi = *it1;
              const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                &          gradPsi = *it2;
              const double partOcc = dftUtils::getPartialOccupancy(
                eigenValues_[ik][eigenIndex], fermiEnergy_, C_kb, tVal);
              VectorizedArray<double> fnk =
                make_vectorized_array(2.0 * partOcc * kPointWeights[ik]);
              eshelbyTensor +=
                fnk * (psi[1] * outer_product(kPointCoord, gradPsi[0]) -
                       psi[0] * outer_product(kPointCoord, gradPsi[1]) -
                       outer_product(kPointCoord, kPointCoord) *
                         (psi[0] * psi[0] + psi[1] * psi[1]));
            }
        }

      return eshelbyTensor;
    }

    // for complex mode
    Tensor<2, 3, VectorizedArray<double>>
    getEnlStress(
      const dealii::AlignedVector<dealii::AlignedVector<dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>>>
        &zetalmDeltaVlProductDistImageAtoms,
      const std::vector<std::vector<std::vector<std::complex<double>>>>
        &projectorKetTimesPsiTimesVTimesPartOcc,
      dealii::AlignedVector<
        Tensor<1, 2, VectorizedArray<double>>>::const_iterator psiBegin,
      dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                       gradPsiBegin,
      const std::vector<double> &      kPointWeights,
      const std::vector<double> &      kPointCoordinates,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList,
      const unsigned int               numBlockedEigenvectors)
    {
      Tensor<2, 3, VectorizedArray<double>> E;
      VectorizedArray<double>               four = make_vectorized_array(4.0);

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
            Tensor<1, 2, VectorizedArray<double>>>::const_iterator it1 =
            psiBegin;
          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>::const_iterator
                                                it2 = gradPsiBegin;
          Tensor<1, 3, VectorizedArray<double>> kPointCoord;
          for (unsigned int ik = 0; ik < numKPoints; ++ik)
            {
              kPointCoord[0] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 0]);
              kPointCoord[1] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 1]);
              kPointCoord[2] =
                make_vectorized_array(kPointCoordinates[ik * 3 + 2]);
              VectorizedArray<double> fnk =
                make_vectorized_array(kPointWeights[ik]);
              for (unsigned int eigenIndex = 0;
                   eigenIndex < numBlockedEigenvectors;
                   ++it1, ++it2, ++eigenIndex)
                {
                  const Tensor<1, 2, VectorizedArray<double>> &psi = *it1;
                  const Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>
                    &gradPsi = *it2;
                  for (unsigned int iPseudoWave = 0;
                       iPseudoWave < numberPseudoWaveFunctions;
                       ++iPseudoWave)
                    {
                      VectorizedArray<double> CReal = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .real());
                      VectorizedArray<double> CImag = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOcc
                          [ik][iAtomNonLocal]
                          [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]
                            .imag());
                      Tensor<1, 3, VectorizedArray<double>> zdvR =
                        zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                          [iPseudoWave][ik][0];
                      Tensor<1, 3, VectorizedArray<double>> zdvI =
                        zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                          [iPseudoWave][ik][1];
                      E -= four * fnk *
                           ((outer_product(gradPsi[0], zdvR) +
                             outer_product(gradPsi[1], zdvI)) *
                              CReal -
                            (outer_product(gradPsi[0], zdvI) -
                             outer_product(gradPsi[1], zdvR)) *
                              CImag +
                            outer_product(
                              ((-psi[1] * zdvR + psi[0] * zdvI) * CReal +
                               (psi[0] * zdvR + psi[1] * zdvI) * CImag),
                              kPointCoord));
                    }
                }
            }
        }
      return E;
    }

    // for real mode
    Tensor<2, 3, VectorizedArray<double>>
    getEnlStress(
      const dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>>
        &zetalmDeltaVlProductDistImageAtoms,
      const std::vector<std::vector<std::vector<double>>>
        &projectorKetTimesPsiTimesVTimesPartOcc,
      dealii::AlignedVector<VectorizedArray<double>>::const_iterator psiBegin,
      dealii::AlignedVector<
        Tensor<1, 3, VectorizedArray<double>>>::const_iterator gradPsiBegin,
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList,
      const unsigned int               numBlockedEigenvectors)
    {
      Tensor<2, 3, VectorizedArray<double>> E;
      VectorizedArray<double>               four = make_vectorized_array(4.0);

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

          dealii::AlignedVector<VectorizedArray<double>>::const_iterator it1 =
            psiBegin;
          dealii::AlignedVector<
            Tensor<1, 3, VectorizedArray<double>>>::const_iterator it2 =
            gradPsiBegin;
          for (unsigned int eigenIndex = 0; eigenIndex < numBlockedEigenvectors;
               ++it1, ++it2, ++eigenIndex)
            {
              const VectorizedArray<double> &              psi     = *it1;
              const Tensor<1, 3, VectorizedArray<double>> &gradPsi = *it2;
              for (unsigned int iPseudoWave = 0;
                   iPseudoWave < numberPseudoWaveFunctions;
                   ++iPseudoWave)
                {
                  VectorizedArray<double> CReal = make_vectorized_array(
                    projectorKetTimesPsiTimesVTimesPartOcc
                      [0][iAtomNonLocal]
                      [numberPseudoWaveFunctions * eigenIndex + iPseudoWave]);
                  Tensor<1, 3, VectorizedArray<double>> zdvR =
                    zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal]
                                                      [iPseudoWave];
                  E -= four * outer_product(gradPsi, zdvR) * CReal;
                }
            }
        }
      return E;
    }

    // for real mode
    Tensor<2, 3, VectorizedArray<double>>
    getEnlStress(
      const dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>>
        &zetalmDeltaVlProductDistImageAtoms,
      const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
      const std::vector<bool> &        isAtomInCell,
      const std::vector<unsigned int> &nonlocalPseudoWfcsAccum)
    {
      Tensor<2, 3, VectorizedArray<double>> zeroTensor;
      VectorizedArray<double>               four = make_vectorized_array(4.0);

      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          zeroTensor[idim][jdim] = make_vectorized_array(0.0);

      Tensor<2, 3, VectorizedArray<double>> E = zeroTensor;

      for (unsigned int iAtomNonLocal = 0;
           iAtomNonLocal < zetalmDeltaVlProductDistImageAtoms.size();
           ++iAtomNonLocal)
        {
          if (!isAtomInCell[iAtomNonLocal])
            continue;

          const int numberPseudoWaveFunctions =
            zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal].size();
          const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            &zetalmDeltaVlProductDistAtom =
              zetalmDeltaVlProductDistImageAtoms[iAtomNonLocal];

          Tensor<2, 3, VectorizedArray<double>> tempE = zeroTensor;
          for (unsigned int iPseudoWave = 0;
               iPseudoWave < numberPseudoWaveFunctions;
               ++iPseudoWave)
            tempE += outer_product(
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                [nonlocalPseudoWfcsAccum[iAtomNonLocal] + iPseudoWave],
              zetalmDeltaVlProductDistAtom[iPseudoWave]);
          E -= four * tempE;
        }
      return E;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const VectorizedArray<double> &              vxc,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoCore)

    {
      return vxc * gradRhoCore;
    }


    Tensor<1, 3, VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRho,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoCore)

    {
      Tensor<1, 3, VectorizedArray<double>> temp;
      for (unsigned int i = 0; i < 3; i++)
        {
          temp[i] = make_vectorized_array(0.0);
          for (unsigned int j = 0; j < 3; j++)
            temp[i] += derExcGradRho[j] * hessianRhoCore[j][i];
        }

      return temp;
      // return hessianRhoCore*derExcGradRho;
    }
  } // namespace eshelbyTensor

} // namespace dftfe
