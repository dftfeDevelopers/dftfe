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
