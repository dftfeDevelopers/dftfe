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
#ifndef eshelbySP_H_
#define eshelbySP_H_
#include "constants.h"
#include "headers.h"

namespace dftfe
{
  using namespace dealii;
  /**
   * @brief The functions in this namespace contain the expressions for the various terms of the configurational force (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
   * for both periodic and non-periodic case.
   *
   * The functions in this namespace are similar to the ones in eshelbyTensor.h
   * except the ones here are specialized
   * for spin polarized case. Spin0 and Spin1 refer to up and down spins
   * respectively. General nomenclature of the input arguments: a) phiTot- total
   * electrostatic potential b) phiExt- sum of electrostatic potential from all
   * nuclear charges c) rho- electron density d) gradRho- gradient of electron
   * density e) exc- exchange correlation energy f) derExcGradRho- derivative of
   * exc with gradient of rho g) psiBegin- begin iterator to vector eigenvectors
   * stored as a flattened array over k points and number of eigenvectors for
   * each k point (periodic case has complex valued eigenvectors which is why
   * Tensor<1,2,VectorizedArray<double> is used in functions for periodic case)
   * h) gradPsiBegin- gradient of eigenvectors
   * i) eigenValues- Kohn sham grounstate eigenvalues stored in a vector. For
   * periodic problems with multiple k points the outer vector should be over k
   * points j) tVal- smearing temperature in K k) pseudoVLoc- local part of the
   * pseuodopotential l) gradPseudoVLoc- gradient of local part of
   * pseudopotential m) ZetaDeltaV- nonlocal pseudowavefunctions times deltaV
   * (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132) n)
   * gradZetaDeltaV- gradient of ZetaDeltaV o) projectorKetTimesPsiTimesV-
   * nonlocal pseudopotential projector ket times eigenvectors which are
   * precomputed. The nonlocal pseudopotential constants are also multiplied to
   * this quantity. (see Eq. 11 in
   * https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
   *
   * @author Sambit Das
   */
  namespace eshelbyTensorSP
  {
    /// Local part of the Eshelby tensor for periodic case (only considers terms
    /// which are summed over k points)
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
        const double                            tVal);

    /// Local part of the Eshelby tensor for non-periodic case
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
      const double               tVal);

    /// exchange-correlation and psp part of the ELoc Eshelby tensor
    Tensor<2, 3, VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const VectorizedArray<double> &              rho,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoSpin1,
      const VectorizedArray<double> &              exc,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin1);


    /// Nonlocal pseudopotential force contribution (for non periodic case)
    Tensor<1, 3, VectorizedArray<double>>
    getFnlAtom(const dealii::AlignedVector<
                 dealii::AlignedVector<VectorizedArray<double>>> &zetaDeltaV,
               const std::vector<std::vector<double>>
                 &projectorKetTimesPsiSpin0TimesVTimesPartOcc,
               const std::vector<std::vector<double>>
                 &projectorKetTimesPsiSpin1TimesVTimesPartOcc,
               dealii::AlignedVector<VectorizedArray<double>>::const_iterator
                 psiSpin0Begin,
               dealii::AlignedVector<VectorizedArray<double>>::const_iterator
                 psiSpin1Begin,
               dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
                 const_iterator gradPsiSpin0Begin,
               dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>::
                 const_iterator   gradPsiSpin1Begin,
               const unsigned int numBlockedEigenvectors);
    /// Nonlocal pseudopotential force contribution (for periodic case)
    Tensor<1, 3, VectorizedArray<double>>
    getFnlAtom(
      const dealii::AlignedVector<dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
        &zetaDeltaV,
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
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      const unsigned int         numBlockedEigenvectors);


    /// Nonlocal pseudopotential force contribution (for periodic case)
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
      const std::vector<unsigned int> &nonlocalAtomsCompactSupportList);

    /// Nonlocal pseudopotential force contribution (for non periodic case)
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
           const std::vector<unsigned int> &nonlocalAtomsCompactSupportList);

    /// Force contribution due to the numerical difference between the input and
    /// output electron density (rhoIn and rhoOut) of the final scf iteration.
    /// vEff denotes the Kohn-Sham effective potential.
    Tensor<1, 3, VectorizedArray<double>>
    getNonSelfConsistentForce(
      const VectorizedArray<double> &              vEffRhoInSpin0,
      const VectorizedArray<double> &              vEffRhoOutSpin0,
      const VectorizedArray<double> &              vEffRhoInSpin1,
      const VectorizedArray<double> &              vEffRhoOutSpin1,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoOutSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoOutSpin1,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoInSpin0,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoInSpin1,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoOutSpin0,
      const Tensor<1, 3, VectorizedArray<double>>
        &derExchCorrEnergyWithGradRhoOutSpin1,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoOutSpin0,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoOutSpin1);

    /// EK Eshelby tensor (used only for stress computation)
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
      const double                            tVal);

    /// Nonlocal pseudopotential Eshelby tensor (used only for stress
    /// computation), multiple k point and complex mode
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
      const unsigned int               numBlockedEigenvectors);

    /// Nonlocal pseudopotential Eshelby tensor (used only for stress
    /// computation), Gamma point case
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
      const unsigned int               numBlockedEigenvectors);

    /// Nonlocal core correction pseudopotential force contribution
    Tensor<1, 3, VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const VectorizedArray<double> &              vxcSpin0,
      const VectorizedArray<double> &              vxcSpin1,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin0,
      const Tensor<1, 3, VectorizedArray<double>> &derExcGradRhoSpin1,
      const Tensor<1, 3, VectorizedArray<double>> &gradRhoCore,
      const Tensor<2, 3, VectorizedArray<double>> &hessianRhoCore,
      const bool                                   isXCGGA);
  }; // namespace eshelbyTensorSP

} // namespace dftfe
#endif
