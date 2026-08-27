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

#include <dftfe/headers.h>
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
