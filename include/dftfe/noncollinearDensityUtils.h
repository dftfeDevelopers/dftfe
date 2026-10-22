// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2026 The Regents of the University of Michigan and DFT-FE
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

#ifndef noncollinearDensityUtils_H_
#define noncollinearDensityUtils_H_

#include <dftfe/FEBasisOperations.h>
#include <dftfe/MemoryStorage.h>
#include <dftfe/TypeConfig.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dftfe
{
  namespace noncollinearDensityUtils
  {
    /**
     * @brief |m| limited to the physical region |m| <= n, and to zero where
     * n is negative.
     *
     * |m| <= n holds for any physical spinor density (Cauchy-Schwarz on the
     * two-component wavefunction), but density mixing treats n and the
     * components of m as independent channels with signed coefficients, so a
     * mixed density can violate it and drive rho^down = (n - |m|)/2 negative
     * before it reaches the XC functional. The bound is attained at full
     * polarization, which libxc handles.
     */
    inline double
    physicalMagnetizationNorm(const double magNorm, const double rho)
    {
      return magNorm > rho ? (rho > 0.0 ? rho : 0.0) : magNorm;
    }

    /**
     * @brief Locally collinear form of a noncollinear density: the charge
     * density and the spin density along a local axis, with the gradient of
     * the spin density taken along that same axis for GGA.
     *
     * @param[in] basisOperationsPtr basis of the quadrature the densities
     * live on
     * @param[in] quadratureId that quadrature
     * @param[in] isGGA whether the gradients are needed
     * @param[in] nonColinDensityValues n, m_z, m_y, m_x
     * @param[in] nonColinGradDensityValues their gradients, read for GGA
     * @param[out] spinPolarizedDensityValues n and the spin density
     * @param[out] spinPolarizedGradDensityValues their gradients, for GGA:
     * that of n, and the gradient of m projected on the unit axis
     * @param[in] projectionAxisDensityValues when null, the axis is m_hat
     * and the spin density is physicalMagnetizationNorm(|m|, n); the clamp
     * applies to the spin density only, not to the axis. Otherwise
     * the spin density is the projection of m onto the magnetization axis of
     * this density, negative where m is anti-aligned with the axis and zero
     * where the axis vanishes. The Kohn-Sham double counting uses the latter:
     * the Hamiltonian applied B_xc = b_in m_hat_in, so the quantity removed
     * from the band energy is b_in (m_hat_in . m_out).
     */
    template <typename T>
    void
    transformToSpinPolarizedDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const bool        isGGA,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &nonColinDensityValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &nonColinGradDensityValues,
      std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &spinPolarizedDensityValues,
      std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &spinPolarizedGradDensityValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        *projectionAxisDensityValues = nullptr)
    {
      spinPolarizedDensityValues.clear();
      spinPolarizedGradDensityValues.clear();
      spinPolarizedDensityValues.resize(2);
      spinPolarizedGradDensityValues.resize(isGGA ? 2 : 0);
      spinPolarizedDensityValues[0] = nonColinDensityValues[0];
      spinPolarizedDensityValues[1].resize(spinPolarizedDensityValues[0].size(),
                                           0.0);
      if (isGGA)
        {
          spinPolarizedGradDensityValues[0] = nonColinGradDensityValues[0];
          spinPolarizedGradDensityValues[1].resize(
            spinPolarizedGradDensityValues[0].size(), 0.0);
        }
      basisOperationsPtr->reinit(0, 0, quadratureId);
      const dftfe::uInt   nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      std::vector<double> cellMagAxisValues(isGGA ? nQuadsPerCell * 3 : 0, 0.0);
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          const double *cellRhoValues =
            nonColinDensityValues[0].data() + iCell * nQuadsPerCell;
          const double *cellMagZValues =
            nonColinDensityValues[1].data() + iCell * nQuadsPerCell;
          const double *cellMagYValues =
            nonColinDensityValues[2].data() + iCell * nQuadsPerCell;
          const double *cellMagXValues =
            nonColinDensityValues[3].data() + iCell * nQuadsPerCell;
          double *cellMagNormValues =
            spinPolarizedDensityValues[1].data() + iCell * nQuadsPerCell;
          // magnetization whose direction defines the local axis
          const double *axisMagZ =
            projectionAxisDensityValues == nullptr ?
              cellMagZValues :
              (*projectionAxisDensityValues)[1].data() + iCell * nQuadsPerCell;
          const double *axisMagY =
            projectionAxisDensityValues == nullptr ?
              cellMagYValues :
              (*projectionAxisDensityValues)[2].data() + iCell * nQuadsPerCell;
          const double *axisMagX =
            projectionAxisDensityValues == nullptr ?
              cellMagXValues :
              (*projectionAxisDensityValues)[3].data() + iCell * nQuadsPerCell;

          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            if (projectionAxisDensityValues == nullptr)
              cellMagNormValues[iQuad] = physicalMagnetizationNorm(
                std::sqrt(cellMagZValues[iQuad] * cellMagZValues[iQuad] +
                          cellMagYValues[iQuad] * cellMagYValues[iQuad] +
                          cellMagXValues[iQuad] * cellMagXValues[iQuad]),
                cellRhoValues[iQuad]);
            else
              {
                const double axisNorm =
                  std::sqrt(axisMagZ[iQuad] * axisMagZ[iQuad] +
                            axisMagY[iQuad] * axisMagY[iQuad] +
                            axisMagX[iQuad] * axisMagX[iQuad]);
                cellMagNormValues[iQuad] =
                  axisNorm > 1e-12 ? (axisMagX[iQuad] * cellMagXValues[iQuad] +
                                      axisMagY[iQuad] * cellMagYValues[iQuad] +
                                      axisMagZ[iQuad] * cellMagZValues[iQuad]) /
                                       axisNorm :
                                     0.0;
              }

          if (isGGA)
            {
              const double *cellGradMagZValues =
                nonColinGradDensityValues[1].data() + 3 * iCell * nQuadsPerCell;
              const double *cellGradMagYValues =
                nonColinGradDensityValues[2].data() + 3 * iCell * nQuadsPerCell;
              const double *cellGradMagXValues =
                nonColinGradDensityValues[3].data() + 3 * iCell * nQuadsPerCell;
              double *cellGradMagNormValues =
                spinPolarizedGradDensityValues[1].data() +
                3 * iCell * nQuadsPerCell;
              for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                {
                  // unit axis from the unclamped norm, as the Hamiltonian
                  // builds it
                  const double axisNorm =
                    std::sqrt(axisMagZ[iQuad] * axisMagZ[iQuad] +
                              axisMagY[iQuad] * axisMagY[iQuad] +
                              axisMagX[iQuad] * axisMagX[iQuad]);
                  if (axisNorm > 1e-12)
                    {
                      cellMagAxisValues[3 * iQuad + 0] =
                        axisMagX[iQuad] / axisNorm;
                      cellMagAxisValues[3 * iQuad + 1] =
                        axisMagY[iQuad] / axisNorm;
                      cellMagAxisValues[3 * iQuad + 2] =
                        axisMagZ[iQuad] / axisNorm;
                    }
                  else
                    {
                      cellMagAxisValues[3 * iQuad + 0] = 0.0;
                      cellMagAxisValues[3 * iQuad + 1] = 0.0;
                      cellMagAxisValues[3 * iQuad + 2] = 0.0;
                    }
                  for (dftfe::uInt idim = 0; idim < 3; ++idim)
                    cellGradMagNormValues[iQuad * 3 + idim] =
                      cellMagAxisValues[3 * iQuad + 2] *
                        cellGradMagZValues[3 * iQuad + idim] +
                      cellMagAxisValues[3 * iQuad + 1] *
                        cellGradMagYValues[3 * iQuad + idim] +
                      cellMagAxisValues[3 * iQuad + 0] *
                        cellGradMagXValues[3 * iQuad + idim];
                }
            }
        }
    }
  } // namespace noncollinearDensityUtils
} // namespace dftfe
#endif
