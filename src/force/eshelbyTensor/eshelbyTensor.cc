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
    getFPSPLocal(const VectorizedArray<double>                rho,
                 const Tensor<1, 3, VectorizedArray<double>> &gradPseudoVLoc,
                 const Tensor<1, 3, VectorizedArray<double>> &gradPhiExt)

    {
      return rho * (gradPseudoVLoc - gradPhiExt);
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
