// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian
//

#include "expConfiningPotential.h"

namespace dftfe
{
  expConfiningPotential::expConfiningPotential()
  {
    d_confiningPotential.resize(0);
    d_confiningPotential.setValue(0);
  }

  void
  expConfiningPotential::init(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                                           &feBasisOp,
    const dftParameters                    &dftParams,
    const std::vector<std::vector<double>> &atomLocations)
  {
    auto quadPoints = feBasisOp->quadPoints();

    dftfe::uInt numQuadPoints = quadPoints.size() / 3;

    double periodicFactorX = dftParams.periodicX == true ? 0.0 : 1.0;
    double periodicFactorY = dftParams.periodicY == true ? 0.0 : 1.0;
    double periodicFactorZ = dftParams.periodicZ == true ? 0.0 : 1.0;
    double maxDist         = 0.0;


    AssertThrow(
      !(dftParams.periodicX && dftParams.periodicY && dftParams.periodicZ),
      dealii::ExcMessage(
        "DFT-FE Error: Confining potential can not be applied in an all-periodic setting"));


    for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        double atomDist = 0.0;
        atomDist =
          atomLocations[iAtom][2] * atomLocations[iAtom][2] * periodicFactorX;
        atomDist +=
          atomLocations[iAtom][3] * atomLocations[iAtom][3] * periodicFactorY;
        atomDist +=
          atomLocations[iAtom][4] * atomLocations[iAtom][4] * periodicFactorZ;
        if (atomDist < maxDist)
          {
            maxDist = atomDist;
          }
      }

    double tol  = 1e-3;
    double tol1 = 1e-8;
    double r1   = dftParams.confiningInnerPotRad;
    double r2   = dftParams.confiningOuterPotRad;

    d_confiningPotential.resize(numQuadPoints);
    d_confiningPotential.setValue(0.0);
    for (dftfe::uInt iQuad = 0; iQuad < numQuadPoints; iQuad++)
      {
        double quadDist = 0.0;
        quadDist = quadPoints[3 * iQuad + 0] * quadPoints[3 * iQuad + 0] *
                   periodicFactorX;
        quadDist += quadPoints[3 * iQuad + 1] * quadPoints[3 * iQuad + 1] *
                    periodicFactorY;
        quadDist += quadPoints[3 * iQuad + 2] * quadPoints[3 * iQuad + 2] *
                    periodicFactorZ;


        double dist1 = quadDist - (maxDist + r1);
        double dist2 = (maxDist + r2) - quadDist;
        double dist3 = r2 - r1;

        if (quadDist < maxDist + r1)
          {
            d_confiningPotential.data()[iQuad] = 0.0;
          }
        else if (quadDist < maxDist + r2)
          {
            double expFactor =
              std::exp(-dftParams.confiningWParam / (dist1 + tol1));
            d_confiningPotential.data()[iQuad] = dftParams.confiningCParam *
                                                 expFactor /
                                                 (dist2 * dist2 + tol * tol);
          }
        else
          {
            double expFactor = std::exp(-dftParams.confiningWParam / (dist3));
            d_confiningPotential.data()[iQuad] =
              dftParams.confiningCParam * expFactor / (tol * tol);
          }
      }
  }

  void
  expConfiningPotential::addConfiningPotential(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &externalPotential) const
  {
    for (dftfe::uInt iQuad = 0; iQuad < externalPotential.size(); iQuad++)
      {
        externalPotential.data()[iQuad] += d_confiningPotential.data()[iQuad];
      }
  }


} // namespace dftfe
