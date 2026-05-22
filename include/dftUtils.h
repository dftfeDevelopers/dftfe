// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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



#ifndef dftUtils_H_
#define dftUtils_H_

#include <headers.h>
#include <mpi.h>

namespace dftfe
{
  /**
   *  @brief Contains repeatedly used functions in the KSDFT calculations
   *
   *  @author Sambit Das, Krishnendu Ghosh, Phani Motamarri
   */

  namespace dftUtils
  {
    extern "C"
    {
      //
      // lapack Ax=b
      //
      void
      dgesv_(int    *N,
             int    *NRHS,
             double *A,
             int    *LDA,
             int    *IPIV,
             double *B,
             int    *LDB,
             int    *INFO);
    }

    inline double
    smearedCharge(double r, double rc)
    {
      if (r > rc)
        return 0.0;

      const double rmrc = r - rc;
      const double r2   = r * r;
      const double rc2  = rc * rc;
      const double rc4  = rc2 * rc2;
      const double rc8  = rc4 * rc4;

      return -21.0 * rmrc * rmrc * rmrc * (6.0 * r2 + 3.0 * r * rc + rc2) /
             (5.0 * M_PI * rc8);
    }

    inline double
    smearedChargeDr(double r, double rc)
    {
      if (r > rc)
        return 0.0;

      const double rmrc  = r - rc;
      const double rmrc2 = rmrc * rmrc;
      const double r2    = r * r;
      const double rc2   = rc * rc;
      const double rc4   = rc2 * rc2;
      const double rc8   = rc4 * rc4;

      return (-63.0 * rmrc2 * (6.0 * r2 + 3.0 * r * rc + rc2) -
              63.0 * rmrc2 * rmrc * (4.0 * r + rc)) /
             (5.0 * M_PI * rc8);
    }

    inline double
    smearedPot(double r, double rc)
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

    // derivative w.r.t r
    inline double
    smearedPotDr(double r, double rc)
    {
      if (r > rc)
        return -1.0 / (r * r);

      const double r2  = r * r;
      const double r4  = r2 * r2;
      const double r5  = r4 * r;
      const double r6  = r5 * r;
      const double rc2 = rc * rc;
      const double rc4 = rc2 * rc2;
      const double rc5 = rc4 * rc;
      const double rc8 = rc4 * rc4;

      return (63.0 * r6 - 180.0 * r5 * rc + 140.0 * r4 * rc2 - 28.0 * r * rc5) /
             (5.0 * rc8);
    }


    inline double
    smearedPairInteractionEqualRadius(const double radius,
                                      const double separation)
    {
      if (radius <= 0.0)
        return 0.0;

      if (separation <= 1.0e-14)
        return (31924.0 / 17875.0) / radius;

      if (separation >= 2.0 * radius)
        return 1.0 / separation;

      const double s = separation / radius;
      double       p;
      if (s < 1.0)
        {
          p = 81.0;
          p = p * s - 360.0;
          p = p * s + 1035.0;
          p = p * s - 4200.0;
          p = p * s + 10920.0;
          p = p * s - 6552.0;
          p = p * s - 30030.0;
          p = p * s + 62920.0;
          p = p * s;
          p = p * s - 102960.0;
          p = p * s;
          p = p * s + 196560.0;
          p = p * s;
          p = p * s - 299600.0;
          p = p * s;
          p = p * s + 383088.0;
          return p / 214500.0 / radius;
        }

      p = 27.0;
      p = p * s - 360.0;
      p = p * s + 1845.0;
      p = p * s - 4200.0;
      p = p * s + 3640.0;
      p = p * s - 6552.0;
      p = p * s + 30030.0;
      p = p * s + 62920.0;
      p = p * s - 772200.0;
      p = p * s + 2471040.0;
      p = p * s - 4420416.0;
      p = p * s + 5241600.0;
      p = p * s - 4542720.0;
      p = p * s + 2867200.0;
      p = p * s - 944640.0;
      p = p * s - 178176.0;
      p = p * s - 19940.0;
      return -p / (214500.0 * s * radius);
    }

    inline double
    smearedPairInteractionDerEqualRadius(const double radius,
                                         const double separation)
    {
      if (radius <= 0.0)
        return 0.0;

      if (separation <= 1.0e-12)
        return 0.0;

      if (separation >= 2.0 * radius)
        return -1.0 / (separation * separation);

      const double s = separation / radius;
      double       p;
      if (s < 1.0)
        {
          p = 243.0;
          p = p * s - 1008.0;
          p = p * s + 2691.0;
          p = p * s - 10080.0;
          p = p * s + 24024.0;
          p = p * s - 13104.0;
          p = p * s - 54054.0;
          p = p * s + 100672.0;
          p = p * s;
          p = p * s - 123552.0;
          p = p * s;
          p = p * s + 157248.0;
          p = p * s;
          p = p * s - 119840.0;
          return s * p / 42900.0 / (radius * radius);
        }

      p = 81.0;
      p = p * s - 1008.0;
      p = p * s + 4797.0;
      p = p * s - 10080.0;
      p = p * s + 8008.0;
      p = p * s - 13104.0;
      p = p * s + 54054.0;
      p = p * s + 100672.0;
      p = p * s - 1081080.0;
      p = p * s + 2965248.0;
      p = p * s - 4420416.0;
      p = p * s + 4193280.0;
      p = p * s - 2725632.0;
      p = p * s + 1146880.0;
      p = p * s - 188928.0;
      p = p * s;
      p = p * s + 3988.0;
      return -p / (42900.0 * s * s * radius * radius);
    }


    template <typename Function>
    inline double
    gaussLegendre32Integrate(const Function &function,
                             const double    lower,
                             const double    upper)
    {
      if (upper <= lower)
        return 0.0;

      static constexpr double points[16]  = {0.0483076656877383162348126,
                                             0.1444719615827964934851864,
                                             0.2392873622521370745446032,
                                             0.3318686022821276497799168,
                                             0.4213512761306353453641194,
                                             0.5068999089322293900237475,
                                             0.5877157572407623290407455,
                                             0.6630442669302152009751152,
                                             0.7321821187402896803874267,
                                             0.7944837959679424069630973,
                                             0.8493676137325699701336930,
                                             0.8963211557660521239653072,
                                             0.9349060759377396891709191,
                                             0.9647622555875064307738119,
                                             0.9856115115452683354001750,
                                             0.9972638618494815635449811};
      static constexpr double weights[16] = {0.0965400885147278005667648,
                                             0.0956387200792748594190820,
                                             0.0938443990808045656391802,
                                             0.0911738786957638847128686,
                                             0.0876520930044038111427715,
                                             0.0833119242269467552221991,
                                             0.0781938957870703064717409,
                                             0.0723457941088485062253994,
                                             0.0658222227763618468376501,
                                             0.0586840934785355471452836,
                                             0.0509980592623761761961632,
                                             0.0428358980222266806568786,
                                             0.0342738629130214331026877,
                                             0.0253920653092620594557526,
                                             0.0162743947309056706051706,
                                             0.0070186100094700966004071};

      const double center = 0.5 * (lower + upper);
      const double half   = 0.5 * (upper - lower);
      double       result = 0.0;
      for (dftfe::uInt i = 0; i < 16; ++i)
        result += weights[i] * (function(center - half * points[i]) +
                                function(center + half * points[i]));
      return half * result;
    }


    inline void
    addUniqueBreakpoint(std::vector<double> &breakpoints,
                        const double         value,
                        const double         lower,
                        const double         upper)
    {
      if (value <= lower || value >= upper)
        return;

      const double tolerance =
        1.0e-12 * std::max(1.0, std::max(std::abs(lower), std::abs(upper)));
      for (const double breakpoint : breakpoints)
        if (std::abs(value - breakpoint) <= tolerance)
          return;

      breakpoints.push_back(value);
    }


    inline double
    smearedPotentialShellAveragePrimitive(const double r, const double radius)
    {
      if (r <= 0.0 || radius <= 0.0)
        return 0.0;

      if (r >= radius)
        {
          const double radiusPrimitive =
            (1.0 - 15.0 / 4.0 + 4.0 - 7.0 / 2.0 + 6.0) * radius / 5.0;
          return radiusPrimitive + (r - radius);
        }

      const double r2  = r * r;
      const double r4  = r2 * r2;
      const double r7  = r4 * r2 * r;
      const double r8  = r4 * r4;
      const double r9  = r8 * r;
      const double rc2 = radius * radius;
      const double rc4 = rc2 * rc2;
      const double rc5 = rc4 * radius;
      const double rc7 = rc5 * rc2;
      const double rc8 = rc4 * rc4;

      return (r9 - 15.0 / 4.0 * radius * r8 + 4.0 * rc2 * r7 -
              7.0 / 2.0 * rc5 * r4 + 6.0 * rc7 * r2) /
             (5.0 * rc8);
    }

    inline double
    smearedPotentialShellAverage(const double shellRadius,
                                 const double centerSeparation,
                                 const double potentialRadius)
    {
      if (shellRadius <= 1.0e-14)
        return smearedPot(centerSeparation, potentialRadius);
      if (centerSeparation <= 1.0e-14)
        return smearedPot(shellRadius, potentialRadius);

      const double upper = centerSeparation + shellRadius;
      const double lower = std::abs(centerSeparation - shellRadius);
      return (smearedPotentialShellAveragePrimitive(upper, potentialRadius) -
              smearedPotentialShellAveragePrimitive(lower, potentialRadius)) /
             (2.0 * centerSeparation * shellRadius);
    }

    inline double
    smearedPairInteraction(const double radiusA,
                           const double radiusB,
                           const double separation)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteraction(radiusB, radiusA, separation);

      const double radiusScale = std::max(radiusA, radiusB);
      if (std::abs(radiusA - radiusB) <= 1.0e-13 * std::max(1.0, radiusScale))
        return smearedPairInteractionEqualRadius(0.5 * (radiusA + radiusB),
                                                 separation);

      if (separation >= radiusA + radiusB)
        return 1.0 / separation;

      auto integrand = [&](const double r) {
        return 4.0 * M_PI * r * r * smearedCharge(r, radiusA) *
               smearedPotentialShellAverage(r, separation, radiusB);
      };

      std::vector<double> breakpoints;
      breakpoints.reserve(5);
      breakpoints.push_back(0.0);
      addUniqueBreakpoint(breakpoints,
                          std::abs(separation - radiusB),
                          0.0,
                          radiusA);
      addUniqueBreakpoint(breakpoints, separation, 0.0, radiusA);
      addUniqueBreakpoint(breakpoints, separation + radiusB, 0.0, radiusA);
      breakpoints.push_back(radiusA);
      std::sort(breakpoints.begin(), breakpoints.end());

      double integral = 0.0;
      for (dftfe::uInt iInterval = 0; iInterval + 1 < breakpoints.size();
           ++iInterval)
        integral += gaussLegendre32Integrate(integrand,
                                             breakpoints[iInterval],
                                             breakpoints[iInterval + 1]);
      return integral;
    }

    inline double
    smearedPairInteractionDer(const double radiusA,
                              const double radiusB,
                              const double separation)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteractionDer(radiusB, radiusA, separation);

      if (separation <= 1.0e-12)
        return 0.0;

      const double radiusScale = std::max(radiusA, radiusB);
      if (std::abs(radiusA - radiusB) <= 1.0e-13 * std::max(1.0, radiusScale))
        return smearedPairInteractionDerEqualRadius(0.5 * (radiusA + radiusB),
                                                    separation);

      if (separation >= radiusA + radiusB)
        return -1.0 / (separation * separation);

      double h = 1.0e-5 * std::max(1.0, std::max(separation, radiusScale));
      if (h >= 0.5 * separation)
        h = 0.5 * separation;

      if (separation > 2.0 * h)
        {
          const double fPlusTwo =
            smearedPairInteraction(radiusA, radiusB, separation + 2.0 * h);
          const double fPlus =
            smearedPairInteraction(radiusA, radiusB, separation + h);
          const double fMinus =
            smearedPairInteraction(radiusA, radiusB, separation - h);
          const double fMinusTwo =
            smearedPairInteraction(radiusA, radiusB, separation - 2.0 * h);
          return (-fPlusTwo + 8.0 * fPlus - 8.0 * fMinus + fMinusTwo) /
                 (12.0 * h);
        }

      const double fPlus =
        smearedPairInteraction(radiusA, radiusB, separation + h);
      const double fMinus =
        smearedPairInteraction(radiusA, radiusB, separation - h);
      return (fPlus - fMinus) / (2.0 * h);
    }

    inline double
    smearedPairInteractionDifference(const double broadRadiusA,
                                     const double broadRadiusB,
                                     const double referenceRadiusA,
                                     const double referenceRadiusB,
                                     const double separation)
    {
      if (broadRadiusA <= 0.0 || broadRadiusB <= 0.0 ||
          referenceRadiusA <= 0.0 || referenceRadiusB <= 0.0)
        return 0.0;

      const double broad =
        smearedPairInteraction(broadRadiusA, broadRadiusB, separation);
      const double reference =
        smearedPairInteraction(referenceRadiusA, referenceRadiusB, separation);
      return broad - reference;
    }

    inline double
    smearedPairInteractionDerDifference(const double broadRadiusA,
                                        const double broadRadiusB,
                                        const double referenceRadiusA,
                                        const double referenceRadiusB,
                                        const double separation)
    {
      if (broadRadiusA <= 0.0 || broadRadiusB <= 0.0 ||
          referenceRadiusA <= 0.0 || referenceRadiusB <= 0.0)
        return 0.0;

      if (separation <= 1.0e-12)
        return 0.0;

      const double maxBroad = std::max(broadRadiusA, broadRadiusB);
      const double maxRef   = std::max(referenceRadiusA, referenceRadiusB);
      if (std::abs(broadRadiusA - referenceRadiusA) <=
            1.0e-13 * std::max(1.0, std::max(maxBroad, maxRef)) &&
          std::abs(broadRadiusB - referenceRadiusB) <=
            1.0e-13 * std::max(1.0, std::max(maxBroad, maxRef)))
        return 0.0;

      const double broad =
        smearedPairInteractionDer(broadRadiusA, broadRadiusB, separation);
      const double reference = smearedPairInteractionDer(referenceRadiusA,
                                                         referenceRadiusB,
                                                         separation);
      return broad - reference;
    }


    inline double
    smearedPairInteractionDifferenceEqualRadius(const double broadRadius,
                                                const double referenceRadius,
                                                const double separation)
    {
      if (broadRadius <= 0.0 || referenceRadius <= 0.0)
        return 0.0;

      if (std::abs(broadRadius - referenceRadius) <= 1.0e-14)
        return 0.0;

      const double maxRadius =
        broadRadius > referenceRadius ? broadRadius : referenceRadius;
      if (separation >= 2.0 * maxRadius)
        return 0.0;

      return smearedPairInteractionDifference(
        broadRadius, broadRadius, referenceRadius, referenceRadius, separation);
    }

    inline double
    smearedPairInteractionDerDifferenceEqualRadius(const double broadRadius,
                                                   const double referenceRadius,
                                                   const double separation)
    {
      if (broadRadius <= 0.0 || referenceRadius <= 0.0)
        return 0.0;

      if (separation <= 1.0e-12 ||
          std::abs(broadRadius - referenceRadius) <= 1.0e-14)
        return 0.0;

      const double maxRadius =
        broadRadius > referenceRadius ? broadRadius : referenceRadius;
      if (separation >= 2.0 * maxRadius)
        return 0.0;

      return smearedPairInteractionDerDifference(
        broadRadius, broadRadius, referenceRadius, referenceRadius, separation);
    }


    inline std::vector<double>
    getFractionalCoordinates(
      const std::vector<double> &latticeVectorsFlattened,
      const std::vector<double> &coordWithRespectToCellCorner)
    {
      std::vector<double> latticeVectorsDup = latticeVectorsFlattened;
      std::vector<double> coordDup          = coordWithRespectToCellCorner;
      //
      // to get the fractionalCoords, solve a linear
      // system of equations
      //
      int N    = 3;
      int NRHS = 1;
      int LDA  = 3;
      int IPIV[3];
      int info;

      dgesv_(&N,
             &NRHS,
             &latticeVectorsDup[0],
             &LDA,
             &IPIV[0],
             &coordDup[0],
             &LDA,
             &info);
      AssertThrow(info == 0,
                  dealii::ExcMessage(
                    "LU solve in finding fractional coordinates failed."));
      return coordDup;
    }

    /** @brief Calculates value of composite generator
     *
     */
    double
    getCompositeGeneratorVal(const double rc,
                             const double r,
                             const double a0,
                             const double power);

    /** @brief Create bounding box around a sphere.
     *
     *  @param  sphere center
     *  @param  sphere radius
     *  @return bounding box
     */
    dealii::BoundingBox<3>
    createBoundingBoxForSphere(const dealii::Point<3> &center,
                               const double            sphereRadius);

    /** @brief Calculates partial occupancy of the atomic orbital using
     *  Fermi-Dirac smearing.
     *
     *  @param  eigenValue
     *  @param  fermiEnergy
     *  @param  kb Boltzmann constant
     *  @param  T smearing temperature
     *  @return double The partial occupancy of the orbital
     */
    double
    getPartialOccupancy(const double eigenValue,
                        const double fermiEnergy,
                        const double kb,
                        const double T);

    /** @brief Calculates the derivative of the partial occupancy of the atomic orbital
     * with respect to (x=eigenvalue-fermiEnergy) using Fermi-Dirac smearing.
     *
     *  @param  eigenValue
     *  @param  fermiEnergy
     *  @param  kb Boltzmann constant
     *  @param  T smearing temperature
     *  @return double The partial occupancy derivative of the orbital
     */
    double
    getPartialOccupancyDer(const double eigenValue,
                           const double fermiEnergy,
                           const double kb,
                           const double T);

    /** @brief Calculates cross product of two vectors
     *
     *  @param  a first vector
     *  @param  b second vector
     *  @param  crossProductVector cross product of a and b
     *  @return void
     */
    void
    cross_product(const std::vector<double> &a,
                  const std::vector<double> &b,
                  std::vector<double>       &crossProductVector);


    /** @brief Applies an affine transformation to the domain bounding vectors
     *
     *  @param  d_domainBoundingVectors the bounding vectors of the domain given as a 2d array
     *  @param  deformationGradient
     *  @return void.
     */
    void
    transformDomainBoundingVectors(
      std::vector<std::vector<double>>   &domainBoundingVectors,
      const dealii::Tensor<2, 3, double> &deformationGradient);

    /** @brief Writes to vtu file only from the lowest pool id
     *
     *  @param  dataOut  DataOut class object
     *  @param  mpiCommParent parent mpi communicator
     *  @param  mpiCommDomain mpi communicator of domain decomposition inside each pool
     *  @param  interpoolcomm  mpi communicator across k point pools
     *  @param  interBandGroupComm  mpi communicator across band groups
     *  @param  fileName
     */
    void
    writeDataVTUParallelLowestPoolId(const dealii::DoFHandler<3> &dofHandler,
                                     const dealii::DataOut<3>    &dataOut,
                                     const MPI_Comm              &mpiCommParent,
                                     const MPI_Comm              &mpiCommDomain,
                                     const MPI_Comm              &interpoolcomm,
                                     const MPI_Comm    &interBandGroupComm,
                                     const std::string &folderName,
                                     const std::string &fileName);

    /** @brief Create index vector which is used for band parallelization
     *
     *  @[in]param  interBandGroupComm  mpi communicator across band groups
     *  @[in]param  numBands
     *  @[out]param bandGroupLowHighPlusOneIndices
     */
    void
    createBandParallelizationIndices(
      const MPI_Comm           &interBandGroupComm,
      const dftfe::uInt         numBands,
      std::vector<dftfe::uInt> &bandGroupLowHighPlusOneIndices);

    void
    createKpointParallelizationIndices(
      const MPI_Comm          &interKptPoolComm,
      const dftfe::Int         numberIndices,
      std::vector<dftfe::Int> &kptGroupLowHighPlusOneIndices);


    /** @brief Wrapper to print current memory usage (prints only the maximum across mpiComm)
     * using PetscMemoryGetCurrentUsage
     *
     *  @[in]param mpiComm  mpi communicator across which the memory printing
     * will be synchronized
     *  @[in]param message message to be printed alongwith the memory usage
     */
    void
    printCurrentMemoryUsage(const MPI_Comm &mpiComm, const std::string message);

    /**
     * A class to split the given communicator into a number of pools
     */
    class Pool
    {
    public:
      Pool(const MPI_Comm   &mpi_communicator,
           const dftfe::uInt n_pools,
           const dftfe::Int  verbosity);

      /**
       * @brief get the communicator across the processor groups
       */
      MPI_Comm &
      get_interpool_comm();

      /**
       * @brief get the communicator associated with processor group
       */
      MPI_Comm &
      get_intrapool_comm();

    private:
      MPI_Comm interpoolcomm;
      MPI_Comm intrapoolcomm;
    };

    /// Exception handler for not implemented functionality
    DeclExceptionMsg(
      ExcNotImplementedYet,
      "This functionality is not implemented yet or not needed to be implemented.");

    /// Exception handler for DFT-FE internal error
    DeclExceptionMsg(ExcInternalError, "DFT-FE internal error.");
  } // namespace dftUtils

} // namespace dftfe
#endif
