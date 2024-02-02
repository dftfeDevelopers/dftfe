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
#include "../../../include/eshelbyTensor.h"

#include "../../../include/dftUtils.h"

namespace dftfe
{
  namespace eshelbyTensor
  {
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getPhiExtEshelbyTensor(
      const dealii::VectorizedArray<double> &                      phiExt,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiExt)
    {
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> identityTensor;
      identityTensor[0][0] = dealii::make_vectorized_array(1.0);
      identityTensor[1][1] = dealii::make_vectorized_array(1.0);
      identityTensor[2][2] = dealii::make_vectorized_array(1.0);



      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> eshelbyTensor =
        dealii::make_vectorized_array(1.0 / (4.0 * M_PI)) *
          outer_product(gradPhiExt, gradPhiExt) -
        dealii::make_vectorized_array(1.0 / (8.0 * M_PI)) *
          scalar_product(gradPhiExt, gradPhiExt) * identityTensor;

      return eshelbyTensor;
    }

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getVselfBallEshelbyTensor(
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradVself)
    {
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> identityTensor;
      identityTensor[0][0] = dealii::make_vectorized_array(1.0);
      identityTensor[1][1] = dealii::make_vectorized_array(1.0);
      identityTensor[2][2] = dealii::make_vectorized_array(1.0);



      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> eshelbyTensor =
        dealii::make_vectorized_array(1.0 / (8.0 * M_PI)) *
          scalar_product(gradVself, gradVself) * identityTensor -
        dealii::make_vectorized_array(1.0 / (4.0 * M_PI)) *
          outer_product(gradVself, gradVself);

      return eshelbyTensor;
    }


    dealii::Tensor<2, 3, double>
    getVselfBallEshelbyTensor(const dealii::Tensor<1, 3, double> &gradVself)
    {
      double identityTensorFactor =
        1.0 / (8.0 * M_PI) * scalar_product(gradVself, gradVself);
      dealii::Tensor<2, 3, double> eshelbyTensor =
        -1.0 / (4.0 * M_PI) * outer_product(gradVself, gradVself);

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;

      return eshelbyTensor;
    }


    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getEElectroEshelbyTensor(
      const dealii::VectorizedArray<double> &                      phiTot,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiTot,
      const dealii::VectorizedArray<double> &                      rho)
    {
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> eshelbyTensor =
        dealii::make_vectorized_array(1.0 / (4.0 * M_PI)) *
        outer_product(gradPhiTot, gradPhiTot);
      dealii::VectorizedArray<double> identityTensorFactor =
        dealii::make_vectorized_array(-1.0 / (8.0 * M_PI)) *
          scalar_product(gradPhiTot, gradPhiTot) +
        rho * phiTot;

      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getELocXcEshelbyTensor(
      const dealii::VectorizedArray<double> &                      rho,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRho,
      const dealii::VectorizedArray<double> &                      exc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &derExcGradRho)
    {
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> eshelbyTensor =
        -outer_product(gradRho, derExcGradRho);
      dealii::VectorizedArray<double> identityTensorFactor = exc * rho;


      eshelbyTensor[0][0] += identityTensorFactor;
      eshelbyTensor[1][1] += identityTensorFactor;
      eshelbyTensor[2][2] += identityTensorFactor;
      return eshelbyTensor;
    }

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
    getELocPspEshelbyTensor(const dealii::VectorizedArray<double> &rho,
                            const dealii::VectorizedArray<double> &pseudoVLoc,
                            const dealii::VectorizedArray<double> &phiExt)
    {
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>> eshelbyTensor;
      dealii::VectorizedArray<double> identityTensorFactor =
        (pseudoVLoc - phiExt) * rho;


      eshelbyTensor[0][0] = identityTensorFactor;
      eshelbyTensor[1][1] = identityTensorFactor;
      eshelbyTensor[2][2] = identityTensorFactor;
      return eshelbyTensor;
    }



    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFPSPLocal(
      const dealii::VectorizedArray<double> rho,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &gradPseudoVLoc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradPhiExt)

    {
      return rho * (gradPseudoVLoc - gradPhiExt);
    }


    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const dealii::VectorizedArray<double> &                      vxc,
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>> &gradRhoCore)

    {
      return vxc * gradRhoCore;
    }


    dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
    getFNonlinearCoreCorrection(
      const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        &derExcGradRho,
      const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
        &hessianRhoCore)

    {
      dealii::Tensor<1, 3, dealii::VectorizedArray<double>> temp;
      for (unsigned int i = 0; i < 3; i++)
        {
          temp[i] = dealii::make_vectorized_array(0.0);
          for (unsigned int j = 0; j < 3; j++)
            temp[i] += derExcGradRho[j] * hessianRhoCore[j][i];
        }

      return temp;
      // return hessianRhoCore*derExcGradRho;
    }
  } // namespace eshelbyTensor

} // namespace dftfe
