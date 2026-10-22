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
// Kernels of the initial guesses (atomic density, core density, local
// pseudopotential, atomic and random wavefunctions). Each launcher has a
// HOST specialization, an OpenMP loop, and a DEVICE one, a grid-stride
// kernel; both evaluate the same DFTFE_HOST_DEVICE_INLINE body per entry.
//

#ifndef atomicInitKernels_H_
#define atomicInitKernels_H_

#include <dftfe/MemorySpaceType.h>
#include <dftfe/TypeConfig.h>
#include <dftfe/splineDeviceEval.h>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>
#include <cmath>
#include <cstddef>

namespace dftfe
{
  namespace atomicInitKernels
  {
    /**
     * @brief Cap, in bytes, on the transient buffers staged for one block of
     * points or pairs passed to these kernels; callers size their blocks
     * from it so the device memory stays bounded.
     */
    constexpr std::size_t maxBlockBytes = 256000000;

    /**
     * @brief Electrostatic potential, divided by the charge, of the smeared
     * nuclear charge of radius rc at distance r: 1/r beyond rc.
     */
    DFTFE_HOST_DEVICE_INLINE double
    smearedPot(const double r, const double rc)
    {
      if (r > rc)
        return 1.0 / r;
      const double r2  = r * r;
      const double r4  = r2 * r2;
      const double r5  = r4 * r;
      const double r6  = r5 * r;
      const double r7  = r6 * r;
      const double rc2 = rc * rc;
      const double rc4 = rc2 * rc2;
      const double rc5 = rc4 * rc;
      const double rc7 = rc5 * rc2;
      const double rc8 = rc4 * rc4;
      return (9.0 * r7 - 30.0 * r6 * rc + 28.0 * r5 * rc2 - 14.0 * r2 * rc5 +
              12.0 * rc7) /
             (5.0 * rc8);
    }

    /**
     * @brief Seed of the random wavefunction initial guess.
     */
    constexpr unsigned long long randomWavefunctionSeed = 137137ULL;

    /**
     * @brief splitmix64 mixer. Pure integer arithmetic, so the stream is the
     * same on every host and device vendor; keyed on global indices it is
     * independent of the parallel decomposition. unsigned long long rather
     * than dftfe::uInt, which is 32 bit unless USE_64BIT_INT is set and would
     * change the stream between builds.
     */
    DFTFE_HOST_DEVICE_INLINE unsigned long long
    splitmix64(unsigned long long z)
    {
      z += 0x9E3779B97F4A7C15ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    }

    /**
     * @brief Uniform double in [-0.5, 0.5) from a seed and three integer
     * keys.
     */
    DFTFE_HOST_DEVICE_INLINE double
    uniformRandomCentered(const unsigned long long seed,
                          const unsigned long long key0,
                          const unsigned long long key1,
                          const unsigned long long key2)
    {
      unsigned long long h = splitmix64(seed ^ splitmix64(key0));
      h                    = splitmix64(h ^ splitmix64(key1));
      h                    = splitmix64(h ^ splitmix64(key2));
      // top 53 bits -> double in [0,1)
      const double u = static_cast<double>(h >> 11) * 1.1102230246251565e-16;
      return u - 0.5;
    }

    /**
     * @brief Random initial-guess coefficient of wavefunction iWave at the
     * node with global key nodeKey: the standard normal pdf at z = 3 u1 with
     * the sign of u2, u1 and u2 uniform in [-0.5, 0.5) drawn from the streams
     * 0 and 1.
     */
    DFTFE_HOST_DEVICE_INLINE double
    randomWavefunctionValue(const unsigned long long seed,
                            const unsigned long long nodeKey,
                            const unsigned long long iWave)
    {
      const double u1 = uniformRandomCentered(seed, nodeKey, iWave, 0ULL);
      const double u2 = uniformRandomCentered(seed, nodeKey, iWave, 1ULL);
      const double z  = 3.0 * u1;
      return std::exp(-0.5 * z * z) * 0.3989422804014327 *
             (u2 < 0.0 ? -1.0 : 1.0);
    }

    /**
     * @brief Radial atomic or core density, its gradient and its hessian at
     * one quadrature point for one center:
     * @code
     *   d    = x_q - R,    r = |d|
     *   if floatingNuclearCharges and r < 1e-4:
     *        r = 1e-4, d_i = 1e-4 / sqrt(3)
     *   (v, v', v'') = r < cutoff ? spline(max(r, rMin)) : (0, 0, 0)
     *   *valueOut     = v
     *   gradOut[i]    = v' d_i / r
     *   hessOut[3i+j] = (v'' - v'/r) (d_i/r)(d_j/r) + delta_ij v'/r
     * @endcode
     * which matches AtomCenteredSphericalFunctionSpline::getDerivativeValue.
     * Each output is written only when non-null, and each derivative is
     * evaluated only when an output needs it.
     *
     * @return whether r lies within the cutoff of the spline
     */
    DFTFE_HOST_DEVICE_INLINE bool
    radialDensityQuadPoint(const double     *quadPoint,
                           const double     *center,
                           const dftfe::uInt splineId,
                           const bool        floatingNuclearCharges,
                           const splineEval::SplineTableGroupView &tables,
                           double                                 *valueOut,
                           double                                 *gradOut,
                           double                                 *hessOut)
    {
      double diff[3];
      diff[0] = quadPoint[0] - center[0];
      diff[1] = quadPoint[1] - center[1];
      diff[2] = quadPoint[2] - center[2];
      double r =
        std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      if (floatingNuclearCharges && r < 1.0e-4)
        {
          r       = 1.0e-4;
          diff[0] = (1.0e-4) / std::sqrt(3.0);
          diff[1] = (1.0e-4) / std::sqrt(3.0);
          diff[2] = (1.0e-4) / std::sqrt(3.0);
        }

      const double cutoff = tables.cutoffs[splineId];
      double       v      = 0.0;
      double       d1     = 0.0;
      double       d2     = 0.0;
      if (r < cutoff)
        {
          const splineEval::SplineTableView t =
            splineEval::splineView(tables, splineId);
          const double rMin  = tables.rMins[splineId];
          const double rEval = r <= rMin ? rMin : r;
          v                  = splineEval::splineEvalValue(t, rEval);
          if (gradOut != nullptr || hessOut != nullptr)
            d1 = splineEval::splineEvalDerivative(t, rEval, 1);
          if (hessOut != nullptr)
            d2 = splineEval::splineEvalDerivative(t, rEval, 2);
        }

      if (valueOut != nullptr)
        *valueOut = v;

      if (gradOut != nullptr)
        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
          gradOut[iDim] = d1 * diff[iDim] / r;

      if (hessOut != nullptr)
        {
          const double radial = d2 - d1 / r;
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              {
                double temp = radial * (diff[iDim] / r) * (diff[jDim] / r);
                if (iDim == jDim)
                  temp += d1 / r;
                hessOut[3 * iDim + jDim] = temp;
              }
        }

      return r <= cutoff;
    }

    /**
     * @brief Free-atom density, magnetization and, when gradRhoOut is
     * non-null, their gradients at one point.
     *
     * The centers are looped in the order given. Each contributes, with v and
     * v' from radialDensityQuadPoint without the floating-charge clamp,
     * @code
     *   t = centerRhoFactor v,       rho += t,      m_k += centerMag_k t
     * @endcode
     * and, unless floatingNuclearCharges and r < 1e-3,
     * @code
     *   g = centerRhoFactor v' d/r,  gradRho += g,  gradM_k += centerMag_k g
     * @endcode
     * rhoOut receives |rho| and gradRhoOut gradRho where |rho| > 1e-8, zero
     * elsewhere. With applyConstraintMag, magZOut is totMagnetization |rho|
     * and gradMagZOut totMagnetization times the unmasked gradRho. The magZ
     * outputs are written for numMagComponents >= 1, magY and magX for
     * numMagComponents == 3. Outputs are indexed by pointIndex.
     */
    DFTFE_HOST_DEVICE_INLINE void
    densitySuperpositionQuadPoint(
      const dftfe::uInt                       pointIndex,
      const double                           *pointCoords,
      const dftfe::uInt                       numCenters,
      const dftfe::uInt                       numMagComponents,
      const bool                              floatingNuclearCharges,
      const bool                              applyConstraintMag,
      const double                            totMagnetization,
      const double                           *centerCoords,
      const dftfe::uInt                      *centerSplineId,
      const double                           *centerRhoFactor,
      const double                           *centerMagZFactor,
      const double                           *centerMagYFactor,
      const double                           *centerMagXFactor,
      const splineEval::SplineTableGroupView &tables,
      double                                 *rhoOut,
      double                                 *magZOut,
      double                                 *magYOut,
      double                                 *magXOut,
      double                                 *gradRhoOut,
      double                                 *gradMagZOut,
      double                                 *gradMagYOut,
      double                                 *gradMagXOut)
    {
      const double *point = pointCoords + pointIndex * 3;
      double        rho   = 0.0;
      double        magZ  = 0.0;
      double        magY  = 0.0;
      double        magX  = 0.0;
      double        gradRho[3];
      double        gradMagZ[3];
      double        gradMagY[3];
      double        gradMagX[3];
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        {
          gradRho[iDim]  = 0.0;
          gradMagZ[iDim] = 0.0;
          gradMagY[iDim] = 0.0;
          gradMagX[iDim] = 0.0;
        }

      for (dftfe::uInt iCenter = 0; iCenter < numCenters; ++iCenter)
        {
          const double *center   = centerCoords + iCenter * 3;
          bool          withGrad = gradRhoOut != nullptr;
          if (withGrad && floatingNuclearCharges)
            {
              const double dx = point[0] - center[0];
              const double dy = point[1] - center[1];
              const double dz = point[2] - center[2];
              withGrad = std::sqrt(dx * dx + dy * dy + dz * dz) >= 1.0e-3;
            }
          double v;
          double g[3];
          radialDensityQuadPoint(point,
                                 center,
                                 centerSplineId[iCenter],
                                 false,
                                 tables,
                                 &v,
                                 withGrad ? g : nullptr,
                                 nullptr);
          const double t = centerRhoFactor[iCenter] * v;
          rho += t;
          magZ += centerMagZFactor[iCenter] * t;
          magY += centerMagYFactor[iCenter] * t;
          magX += centerMagXFactor[iCenter] * t;
          if (withGrad)
            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
              {
                const double gi = centerRhoFactor[iCenter] * g[iDim];
                gradRho[iDim] += gi;
                gradMagZ[iDim] += centerMagZFactor[iCenter] * gi;
                gradMagY[iDim] += centerMagYFactor[iCenter] * gi;
                gradMagX[iDim] += centerMagXFactor[iCenter] * gi;
              }
        }

      const double rhoAbs = rho < 0.0 ? -rho : rho;
      rhoOut[pointIndex]  = rhoAbs;
      if (numMagComponents >= 1)
        magZOut[pointIndex] =
          applyConstraintMag ? totMagnetization * rhoAbs : magZ;
      if (numMagComponents == 3)
        {
          magYOut[pointIndex] = magY;
          magXOut[pointIndex] = magX;
        }

      if (gradRhoOut != nullptr)
        {
          const double rhoMask = rhoAbs > 1.0e-8 ? 1.0 : 0.0;
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            {
              gradRhoOut[pointIndex * 3 + iDim] = rhoMask * gradRho[iDim];
              if (numMagComponents >= 1)
                gradMagZOut[pointIndex * 3 + iDim] =
                  applyConstraintMag ? totMagnetization * gradRho[iDim] :
                                       gradMagZ[iDim];
              if (numMagComponents == 3)
                {
                  gradMagYOut[pointIndex * 3 + iDim] = gradMagY[iDim];
                  gradMagXOut[pointIndex * 3 + iDim] = gradMagX[iDim];
                }
            }
        }
    }

    /**
     * @brief Sum of radialDensityQuadPoint over every center, value and
     * gradient, at quadrature point index.
     */
    DFTFE_HOST_DEVICE_INLINE void
    radialDensityQuadTotalEntry(const dftfe::uInt  index,
                                const dftfe::uInt  numCenters,
                                const bool         floatingNuclearCharges,
                                const double      *quadCoords,
                                const double      *centerCoords,
                                const dftfe::uInt *centerSplineId,
                                const splineEval::SplineTableGroupView &tables,
                                double *valueOut,
                                double *gradOut)
    {
      const double *qp    = quadCoords + index * 3;
      double        value = 0.0;
      double        grad[3];
      grad[0] = 0.0;
      grad[1] = 0.0;
      grad[2] = 0.0;
      for (dftfe::uInt iCenter = 0; iCenter < numCenters; ++iCenter)
        {
          double v;
          double g[3];
          radialDensityQuadPoint(qp,
                                 centerCoords + iCenter * 3,
                                 centerSplineId[iCenter],
                                 floatingNuclearCharges,
                                 tables,
                                 &v,
                                 g,
                                 nullptr);
          value += v;
          grad[0] += g[0];
          grad[1] += g[1];
          grad[2] += g[2];
        }
      valueOut[index]        = value;
      gradOut[index * 3 + 0] = grad[0];
      gradOut[index * 3 + 1] = grad[1];
      gradOut[index * 3 + 2] = grad[2];
    }

    /**
     * @brief radialDensityQuadPoint gradient and, when hessOut is non-null,
     * hessian for entry index = iPair * numQuadPoints + iQuad.
     *
     * @return whether the point lies within the cutoff of the spline
     */
    DFTFE_HOST_DEVICE_INLINE bool
    radialDensityPairEntry(const dftfe::uInt  index,
                           const dftfe::uInt  numQuadPoints,
                           const bool         floatingNuclearCharges,
                           const double      *quadCoords,
                           const double      *pairCenters,
                           const dftfe::uInt *pairCellSlot,
                           const dftfe::uInt *pairSplineId,
                           const splineEval::SplineTableGroupView &tables,
                           double                                 *gradOut,
                           double                                 *hessOut)
    {
      const dftfe::uInt iPair    = index / numQuadPoints;
      const dftfe::uInt iQuad    = index % numQuadPoints;
      const dftfe::uInt cellSlot = pairCellSlot[iPair];
      return radialDensityQuadPoint(quadCoords +
                                      (cellSlot * numQuadPoints + iQuad) * 3,
                                    pairCenters + iPair * 3,
                                    pairSplineId[iPair],
                                    floatingNuclearCharges,
                                    tables,
                                    nullptr,
                                    gradOut + index * 3,
                                    hessOut == nullptr ? nullptr :
                                                         hessOut + index * 9);
    }

    /**
     * @brief Atomic-orbital initial guess for entry
     * index = iPoint * numOrbitals + iOrbital: the raw radial spline times the
     * real spherical harmonic, summed over the orbital's centers within
     * truncationRadius and scaled by 1/sqrt(2) when isNoncolin.
     *
     * The spline is evaluated without the zero-beyond-cutoff and rMin clamp
     * of AtomCenteredSphericalFunctionSpline, as alglib::spline1dcalc is on
     * the atomic orbital data.
     */
    DFTFE_HOST_DEVICE_INLINE double
    atomicWavefunctionEntry(const dftfe::uInt  index,
                            const dftfe::uInt  numOrbitals,
                            const bool         isNoncolin,
                            const double       truncationRadius,
                            const double      *pointCoords,
                            const dftfe::uInt *orbitalCenterStart,
                            const double      *centerCoords,
                            const dftfe::Int  *orbitalL,
                            const dftfe::Int  *orbitalM,
                            const dftfe::uInt *orbitalSplineId,
                            const splineEval::SplineTableGroupView &tables)
    {
      const dftfe::uInt iPoint = index / numOrbitals;
      const dftfe::uInt iOrb   = index - iPoint * numOrbitals;

      const double *pp = pointCoords + iPoint * 3;
      const double  px = pp[0];
      const double  py = pp[1];
      const double  pz = pp[2];

      const int                         l = orbitalL[iOrb];
      const int                         m = orbitalM[iOrb];
      const splineEval::SplineTableView t =
        splineEval::splineView(tables, orbitalSplineId[iOrb]);

      double acc = 0.0;
      for (dftfe::uInt iAtom = orbitalCenterStart[iOrb];
           iAtom < orbitalCenterStart[iOrb + 1];
           ++iAtom)
        {
          const double x = px - centerCoords[iAtom * 3 + 0];
          const double y = py - centerCoords[iAtom * 3 + 1];
          const double z = pz - centerCoords[iAtom * 3 + 2];
          const double r = std::sqrt(x * x + y * y + z * z);

          if (r > truncationRadius)
            continue;

          double theta = std::acos(z / r);
          double phi   = std::atan2(y, x);
          if (r == 0.0)
            {
              theta = 0.0;
              phi   = 0.0;
            }

          const double R = splineEval::splineEvalValue(t, r);
          const double Y =
            atomCenteredSphericalFunctionDeviceUtils::realSphericalHarmonic(
              l, m, theta, phi);

          acc += isNoncolin ? R * Y / 1.4142135623730950488016887242097 : R * Y;
        }
      return acc;
    }

    /**
     * @brief Local pseudopotential of one center for entry
     * index = iPair * numQuadPoints + iQuad:
     * @code
     *   r = |x_q - R_pair|
     *   v = r > cutoff ? -Z_pair / r : radialSplineValue(r)
     * @endcode
     * so that r == cutoff gives the zero of the spline path, as the host
     * getRadialValue does.
     *
     * @param[in] index iPair * numQuadPoints + iQuad
     * @param[in] numQuadPoints quadrature points per cell
     * @param[in] quadCoords x, y, z of every quadrature point of the staged
     * cells, cell slot major
     * @param[in] pairCenters x, y, z of the center of each pair
     * @param[in] pairCharges valence charge Z_pair of each pair
     * @param[in] pairCellSlot staged cell of each pair in quadCoords
     * @param[in] pairSplineId function id of each pair in tables
     * @param[in] tables the local pseudopotential splines
     * @param[out] distance r, for the caller's support test
     * @return v at entry index
     */
    DFTFE_HOST_DEVICE_INLINE double
    localPseudoPotentialPairEntry(
      const dftfe::uInt                       index,
      const dftfe::uInt                       numQuadPoints,
      const double                           *quadCoords,
      const double                           *pairCenters,
      const double                           *pairCharges,
      const dftfe::uInt                      *pairCellSlot,
      const dftfe::uInt                      *pairSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 &distance)
    {
      const dftfe::uInt iPair    = index / numQuadPoints;
      const dftfe::uInt iQuad    = index % numQuadPoints;
      const dftfe::uInt cellSlot = pairCellSlot[iPair];
      const double *qp = quadCoords + (cellSlot * numQuadPoints + iQuad) * 3;
      const double  dx = qp[0] - pairCenters[iPair * 3 + 0];
      const double  dy = qp[1] - pairCenters[iPair * 3 + 1];
      const double  dz = qp[2] - pairCenters[iPair * 3 + 2];
      const double  r  = std::sqrt(dx * dx + dy * dy + dz * dz);
      distance         = r;

      const dftfe::uInt splineId = pairSplineId[iPair];
      return r > tables.cutoffs[splineId] ?
               -pairCharges[iPair] / r :
               splineEval::radialSplineValue(tables, splineId, r);
    }

    /**
     * @brief Local pseudopotential summed over every center at quadrature
     * point index. For each center:
     * @code
     *   r = |x_q - R_i|
     *   v = (r <= maxTail && r <= tailCutoff_i) ? radialSplineValue_i(r)
     *                                           : -Z_i / r
     *   if analyticSmearedCharge:
     *      v -= (r > w_i) ? -Z_i/r : -Z_i * smearedPot(r, w_i)
     * @endcode
     */
    DFTFE_HOST_DEVICE_INLINE double
    localPseudoPotentialTotalEntry(
      const dftfe::uInt                       index,
      const dftfe::uInt                       numCenters,
      const bool                              analyticSmearedCharge,
      const double                            maxTail,
      const double                           *quadCoords,
      const double                           *centerCoords,
      const double                           *centerCharges,
      const double                           *centerTailCutoff,
      const double                           *centerSmearedWidth,
      const dftfe::uInt                      *centerSplineId,
      const splineEval::SplineTableGroupView &tables)
    {
      const double *qp  = quadCoords + index * 3;
      const double  qx  = qp[0];
      const double  qy  = qp[1];
      const double  qz  = qp[2];
      double        acc = 0.0;
      for (dftfe::uInt iAtom = 0; iAtom < numCenters; ++iAtom)
        {
          const double dx     = qx - centerCoords[iAtom * 3 + 0];
          const double dy     = qy - centerCoords[iAtom * 3 + 1];
          const double dz     = qz - centerCoords[iAtom * 3 + 2];
          const double r      = std::sqrt(dx * dx + dy * dy + dz * dz);
          const double rInv   = 1.0 / r;
          const double charge = centerCharges[iAtom];

          double v =
            (r <= maxTail && r <= centerTailCutoff[iAtom]) ?
              splineEval::radialSplineValue(tables, centerSplineId[iAtom], r) :
              -charge * rInv;

          if (analyticSmearedCharge)
            {
              const double w = centerSmearedWidth[iAtom];
              v -= r > w ? -charge / r : -charge * smearedPot(r, w);
            }
          acc += v;
        }
      return acc;
    }

    /**
     * @brief Random initial-guess entry of a flattened block indexed
     * [(iNode * numWaveFunctions + iWave) * numRealsPerCoeff + comp], for
     * the entry index over iNode, iWave >= firstWave and comp. The real part
     * of an unconstrained node is randomWavefunctionValue keyed on
     * globalNodeStartIndex + iNode; imaginary parts and the nodes of
     * constrained dofs (iNode / numSpinorComponents) are zero.
     */
    DFTFE_HOST_DEVICE_INLINE void
    randomWavefunctionEntry(const dftfe::uInt        index,
                            const dftfe::uInt        numWaveFunctions,
                            const dftfe::uInt        firstWave,
                            const dftfe::uInt        numRealsPerCoeff,
                            const dftfe::uInt        numSpinorComponents,
                            const unsigned long long globalNodeStartIndex,
                            const unsigned long long seed,
                            const dftfe::uInt       *constrainedMask,
                            double                  *wfcData)
    {
      const dftfe::uInt numRandomWaves = numWaveFunctions - firstWave;
      const dftfe::uInt coeffIndex     = index / numRealsPerCoeff;
      const dftfe::uInt comp           = index % numRealsPerCoeff;
      const dftfe::uInt iNode          = coeffIndex / numRandomWaves;
      const dftfe::uInt iWave = firstWave + coeffIndex % numRandomWaves;
      double            value = 0.0;
      if (comp == 0 && constrainedMask[iNode / numSpinorComponents] == 0)
        value =
          randomWavefunctionValue(seed,
                                  globalNodeStartIndex +
                                    static_cast<unsigned long long>(iNode),
                                  static_cast<unsigned long long>(iWave));
      wfcData[(iNode * numWaveFunctions + iWave) * numRealsPerCoeff + comp] =
        value;
    }

    // Launchers. numHostThreads is the OpenMP thread count of the HOST
    // specialization; the DEVICE specialization ignores it.

    /**
     * @brief atomicWavefunctionEntry at every (point, orbital) entry,
     * valueOut indexed [iPoint * numOrbitals + iOrbital].
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicWavefunctionSuperposition(
      const dftfe::uInt                       numPoints,
      const dftfe::uInt                       numOrbitals,
      const bool                              isNoncolin,
      const double                            truncationRadius,
      const double                           *pointCoords,
      const dftfe::uInt                      *orbitalCenterStart,
      const double                           *centerCoords,
      const dftfe::Int                       *orbitalL,
      const dftfe::Int                       *orbitalM,
      const dftfe::uInt                      *orbitalSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *valueOut,
      const dftfe::uInt                       numHostThreads);

    /**
     * @brief localPseudoPotentialPairEntry at every quadrature point of every
     * (atom-or-image, cell) pair. pairHasSupport[iPair] is set when any
     * quadrature point of the pair lies within cutOffForPsp, which decides
     * whether the caller stores the pair at all.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    localPseudoPotentialValues(const dftfe::uInt  numPairs,
                               const dftfe::uInt  numQuadPoints,
                               const double       cutOffForPsp,
                               const double      *quadCoords,
                               const double      *pairCenters,
                               const double      *pairCharges,
                               const dftfe::uInt *pairCellSlot,
                               const dftfe::uInt *pairSplineId,
                               const splineEval::SplineTableGroupView &tables,
                               double                                 *valueOut,
                               dftfe::uInt      *pairHasSupport,
                               const dftfe::uInt numHostThreads);

    /**
     * @brief localPseudoPotentialTotalEntry at every quadrature point of every
     * cell. The loop over the centers is inside the entry, so nothing
     * per-(center, point) is stored.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    localPseudoPotentialTotal(const dftfe::uInt  numCells,
                              const dftfe::uInt  numQuadPoints,
                              const dftfe::uInt  numCenters,
                              const bool         analyticSmearedCharge,
                              const double       maxTail,
                              const double      *quadCoords,
                              const double      *centerCoords,
                              const double      *centerCharges,
                              const double      *centerTailCutoff,
                              const double      *centerSmearedWidth,
                              const dftfe::uInt *centerSplineId,
                              const splineEval::SplineTableGroupView &tables,
                              double                                 *valueOut,
                              const dftfe::uInt numHostThreads);

    /**
     * @brief radialDensityQuadTotalEntry at every quadrature point of every
     * cell -- the totals of initAtomicRho and initCoreRho. Centers beyond the
     * cutoff contribute exactly zero, so the sum equals the host's over the
     * centers it does not reject.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicRadialDensityQuadTotal(const dftfe::uInt  numCells,
                                 const dftfe::uInt  numQuadPoints,
                                 const dftfe::uInt  numCenters,
                                 const bool         floatingNuclearCharges,
                                 const double      *quadCoords,
                                 const double      *centerCoords,
                                 const dftfe::uInt *centerSplineId,
                                 const splineEval::SplineTableGroupView &tables,
                                 double           *valueOut,
                                 double           *gradOut,
                                 const dftfe::uInt numHostThreads);

    /**
     * @brief radialDensityPairEntry at every quadrature point of every
     * (atom-or-image, cell) pair -- the per-atom core-density maps of
     * initCoreRho read by the force and stress. pairHasSupport[iPair] is set
     * when any quadrature point of the pair lies within the cutoff, which
     * decides whether the caller stores the pair at all.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicRadialDensityPairDerivatives(
      const dftfe::uInt                       numPairs,
      const dftfe::uInt                       numQuadPoints,
      const bool                              floatingNuclearCharges,
      const double                           *quadCoords,
      const double                           *pairCenters,
      const dftfe::uInt                      *pairCellSlot,
      const dftfe::uInt                      *pairSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *gradOut,
      double                                 *hessOut,
      dftfe::uInt                            *pairHasSupport,
      const dftfe::uInt                       numHostThreads);

    /**
     * @brief densitySuperpositionQuadPoint at every point -- the initial
     * density of initRho, at quadrature points or, with gradRhoOut null, at
     * nodal points.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicDensityQuadSuperposition(
      const dftfe::uInt                       numPoints,
      const dftfe::uInt                       numCenters,
      const dftfe::uInt                       numMagComponents,
      const bool                              floatingNuclearCharges,
      const bool                              applyConstraintMag,
      const double                            totMagnetization,
      const double                           *pointCoords,
      const double                           *centerCoords,
      const dftfe::uInt                      *centerSplineId,
      const double                           *centerRhoFactor,
      const double                           *centerMagZFactor,
      const double                           *centerMagYFactor,
      const double                           *centerMagXFactor,
      const splineEval::SplineTableGroupView &tables,
      double                                 *rhoOut,
      double                                 *magZOut,
      double                                 *magYOut,
      double                                 *magXOut,
      double                                 *gradRhoOut,
      double                                 *gradMagZOut,
      double                                 *gradMagYOut,
      double                                 *gradMagXOut,
      const dftfe::uInt                       numHostThreads);

    /**
     * @brief randomWavefunctionEntry over numLocalNodes nodes and the
     * wavefunctions [firstWave, numWaveFunctions); entries of the other
     * wavefunctions are left untouched. numLocalNodes counts spinor
     * components, a node being dof * numSpinorComponents + spinor, and
     * constrainedMask holds one flag per dof. A complex array is passed as
     * its interleaved real and imaginary parts with numRealsPerCoeff = 2.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    fillRandomWavefunctions(const dftfe::uInt        numLocalNodes,
                            const dftfe::uInt        numWaveFunctions,
                            const dftfe::uInt        firstWave,
                            const dftfe::uInt        numRealsPerCoeff,
                            const dftfe::uInt        numSpinorComponents,
                            const unsigned long long globalNodeStartIndex,
                            const unsigned long long seed,
                            const dftfe::uInt       *constrainedMask,
                            double                  *wfcData,
                            const dftfe::uInt        numHostThreads);

    // The HOST and DEVICE specializations are defined in
    // src/dft/atomicInitKernels.cc and src/dft/atomicInitDeviceKernels.cc.
    // They must be declared before any use, otherwise a caller would
    // trigger implicit instantiation of the primary and the program would
    // be ill-formed.
    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double,
      const double *,
      const dftfe::uInt *,
      const double *,
      const dftfe::Int *,
      const dftfe::Int *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      const dftfe::uInt);
    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const double,
      const double *,
      const double *,
      const double *,
      const dftfe::uInt *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      dftfe::uInt *,
      const dftfe::uInt);
    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double,
      const double *,
      const double *,
      const double *,
      const double *,
      const double *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      const dftfe::uInt);
    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double *,
      const double *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      const dftfe::uInt);
    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double *,
      const double *,
      const dftfe::uInt *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      dftfe::uInt *,
      const dftfe::uInt);
    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const bool,
      const double,
      const double *,
      const double *,
      const dftfe::uInt *,
      const double *,
      const double *,
      const double *,
      const double *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      const dftfe::uInt);
    template <>
    void
    fillRandomWavefunctions<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const unsigned long long,
      const unsigned long long,
      const dftfe::uInt *,
      double *,
      const dftfe::uInt);

    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double,
      const double *,
      const dftfe::uInt *,
      const double *,
      const dftfe::Int *,
      const dftfe::Int *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      const dftfe::uInt);
    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const double,
      const double *,
      const double *,
      const double *,
      const dftfe::uInt *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      dftfe::uInt *,
      const dftfe::uInt);
    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double,
      const double *,
      const double *,
      const double *,
      const double *,
      const double *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      const dftfe::uInt);
    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double *,
      const double *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      const dftfe::uInt);
    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const double *,
      const double *,
      const dftfe::uInt *,
      const dftfe::uInt *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      dftfe::uInt *,
      const dftfe::uInt);
    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const bool,
      const bool,
      const double,
      const double *,
      const double *,
      const dftfe::uInt *,
      const double *,
      const double *,
      const double *,
      const double *,
      const splineEval::SplineTableGroupView &,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      double *,
      const dftfe::uInt);
    template <>
    void
    fillRandomWavefunctions<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const dftfe::uInt,
      const unsigned long long,
      const unsigned long long,
      const dftfe::uInt *,
      double *,
      const dftfe::uInt);

  } // namespace atomicInitKernels
} // namespace dftfe
#endif
