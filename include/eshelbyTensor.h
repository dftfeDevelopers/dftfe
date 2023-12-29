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



#ifndef eshelby_H_
#define eshelby_H_
#include "constants.h"
#include "headers.h"

namespace dftfe
{
  /**
   * @brief The functions in this namespace contain the expressions for the various terms of the configurational force (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
   * for both periodic (see Eq. 38) and non-periodic (see Eqs. 28-29) case.
   *
   * Basically, the configurational force is the Gateaux derivative
   * of the Kohn-Sham saddle point problem with respect to perturbations of the
   * underlying space due to generic generator (which can be affine perturbation
   * in case of stress computation or an atom centered generator with a compact
   * support for computing the forces). The terms in the configurational force
   * can be grouped into two types: one type can be written as contraction of
   * Eshelby tensors (second order tensor) with the gradient of the Generator.
   * Another type involves contraction of first order tensors with the
   * Generator. The functions in this class provide expressions for the left
   * side of the contraction operation- the second order Eshelby tensors
   * (denoted by E*) and the first order force tensors (denoted by F*). General
   * nomenclature of the input arguments: a) phiTot- total electrostatic
   * potential b) phiExt- sum of electrostatic potential from all nuclear
   * charges c) rho- electron density d) gradRho- gradient of electron density
   * e) exc- exchange correlation energy
   * f) derExcGradRho- derivative of exc with gradient of rho
   * g) psiBegin- begin iterator to vector eigenvectors stored as a flattened
   * array over k points and number of eigenvectors for each k point (periodic
   * case has complex valued eigenvectors which is why
   * dealii::Tensor<1,2,dealii::VectorizedArray<double> is used in functions for
   * periodic case) h) gradPsiBegin- gradient of eigenvectors i) eigenValues-
   * Kohn sham grounstate eigenvalues stored in a vector. For periodic problems
   * with multiple k points the outer vector should be over k points j) tVal-
   * smearing temperature in K k) pseudoVLoc- local part of the pseudopotential
   * l) gradPseudoVLoc- gradient of local part of pseudopotential m) ZetaDeltaV-
   * nonlocal pseudowavefunctions times deltaV (see Eq. 11 in
   * https://link.aps.org/doi/10.1103/PhysRevB.97.165132) n) gradZetaDeltaV-
   * gradient of ZetaDeltaV o) projectorKetTimesPsiTimesV- nonlocal
   * pseudopotential projector ket times eigenvectors which are precomputed. The
   * nonlocal pseudopotential constants are also multiplied to this quantity.
   * (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
   *
   * @author Sambit Das
   */
  namespace eshelbyTensor
  {
    /// Eshelby tensor from sum of electrostatic potential from all nuclear
    /// charges (only used for testing purpose)
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getPhiExtEshelbyTensor(
      const dealii::VectorizedArray<double> &                      phiExt,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiExt);

    /// Eshelby tensor corresponding to nuclear self energy (only used for
    /// testing purpose)
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getVselfBallEshelbyTensor(
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradVself);

    /// Eshelby tensor corresponding to nuclear self energy
    dealii::Tensor<2, 3, double>
    getVselfBallEshelbyTensor(const dealii::Tensor<1, 3, double> &gradVself);



    /// All-electron electrostatic part of the Eshelby tensor
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getEElectroEshelbyTensor(
      const dealii::VectorizedArray<double> &                      phiTot,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiTot,
      const dealii::VectorizedArray<double> &                      rho);

    /// exchange-correlation part of the ELoc Eshelby tensor
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const dealii::VectorizedArray<double> &                      rho,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRho,
      const dealii::VectorizedArray<double> &                      exc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &derExcGradRho);


    /// psp part of the ELoc Eshelby tensor
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getELocPspEshelbyTensor(const dealii::VectorizedArray<double> &rho,
                            const dealii::VectorizedArray<double> &pseudoVLoc,
                            const dealii::VectorizedArray<double> &phiExt);

    /// Local pseudopotential force contribution
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFPSPLocal(
      const dealii::VectorizedArray<double> rho,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &gradPseudoVLoc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiExt);


    /// Nonlocal core correction pseudopotential force contribution
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const dealii::VectorizedArray<double> &                      vxc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRhoCore);

    /// Nonlocal core correction pseudopotential force contribution
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &derExcGradRho,
      const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
        &hessianRhoCore);

  }; // namespace eshelbyTensor

} // namespace dftfe
#endif
