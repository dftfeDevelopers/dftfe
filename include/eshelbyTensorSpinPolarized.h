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
  // using namespace dealii;
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
   * dealii::Tensor<1,2,dealii::VectorizedArray<double> is used in functions for periodic case)
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
    /// exchange-correlation and psp part of the ELoc Eshelby tensor
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const dealii::VectorizedArray<double> &              rho,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRhoSpin0,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRhoSpin1,
      const dealii::VectorizedArray<double> &              exc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &derExcGradRhoSpin0,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &derExcGradRhoSpin1);



    /// Nonlocal core correction pseudopotential force contribution
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const dealii::VectorizedArray<double> &              vxcSpin0,
      const dealii::VectorizedArray<double> &              vxcSpin1,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &derExcGradRhoSpin0,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &derExcGradRhoSpin1,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRhoCore,
      const dealii::Tensor<2, 3, dealii::VectorizedArray<double>> &hessianRhoCore,
      const bool                                   isXCGGA);
  }; // namespace eshelbyTensorSP

} // namespace dftfe
#endif
