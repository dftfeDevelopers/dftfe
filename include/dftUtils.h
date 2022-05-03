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
    inline double
    smearedCharge(double r, double rc)
    {
      double val;
      if (r > rc)
        {
          val = 0.0;
        }
      else
        {
          val = -21.0 * pow(r - rc, 3.0) *
                (6.0 * r * r + 3.0 * r * rc + rc * rc) /
                (5.0 * M_PI * pow(rc, 8.0));
        }
      return val;
    }

    inline double
    smearedChargeDr(double r, double rc)
    {
      double val;
      if (r > rc)
        {
          val = 0.0;
        }
      else
        {
          val =
            (-63.0 * pow(r - rc, 2.0) * (6.0 * r * r + 3.0 * r * rc + rc * rc) -
             63.0 * pow(r - rc, 3.0) * (4.0 * r + rc)) /
            (5.0 * M_PI * pow(rc, 8.0));
        }
      return val;
    }

    inline double
    smearedPot(double r, double rc)
    {
      double val;
      if (r > rc)
        {
          val = 1.0 / r;
        }
      else
        {
          val = (9.0 * pow(r, 7.0) - 30.0 * pow(r, 6.0) * rc +
                 28.0 * pow(r, 5.0) * pow(rc, 2.0) -
                 14.0 * pow(r, 2.0) * pow(rc, 5) + 12.0 * pow(rc, 7)) /
                (5.0 * pow(rc, 8.0));
        }
      return val;
    }

    // derivative w.r.t r
    inline double
    smearedPotDr(double r, double rc)
    {
      double val;
      if (r > rc)
        {
          val = -1.0 / pow(r, 2.0);
        }
      else
        {
          val = (63.0 * pow(r, 6.0) - 180.0 * pow(r, 5.0) * rc +
                 140.0 * pow(r, 4.0) * pow(rc, 2.0) -
                 28.0 * pow(r, 1.0) * pow(rc, 5)) /
                (5.0 * pow(rc, 8.0));
        }
      return val;
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
                  std::vector<double> &      crossProductVector);


    /** @brief Applies an affine transformation to the domain bounding vectors
     *
     *  @param  d_domainBoundingVectors the bounding vectors of the domain given as a 2d array
     *  @param  deformationGradient
     *  @return void.
     */
    void
    transformDomainBoundingVectors(
      std::vector<std::vector<double>> &  domainBoundingVectors,
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
                                     const dealii::DataOut<3> &   dataOut,
                                     const MPI_Comm &             mpiCommParent,
                                     const MPI_Comm &             mpiCommDomain,
                                     const MPI_Comm &             interpoolcomm,
                                     const MPI_Comm &   interBandGroupComm,
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
      const MPI_Comm &           interBandGroupComm,
      const unsigned int         numBands,
      std::vector<unsigned int> &bandGroupLowHighPlusOneIndices);

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
      Pool(const MPI_Comm &   mpi_communicator,
           const unsigned int n_pools,
           const int          verbosity);

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
