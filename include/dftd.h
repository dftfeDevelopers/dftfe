// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
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

#include <headers.h>
#ifdef DFTFE_WITH_DFTD3
#  include <dftd3.h>
#endif
#ifdef DFTFE_WITH_DFTD4
#  include <dftd4.h>
#endif

#ifndef dispersionCorrection_H_
#  define dispersionCorrection_H_
namespace dftfe
{
  /**
   * @brief Calculates dispersion correction to energy, force and stress
   *
   * @author Nikhil Kodali
   */
  class dispersionCorrection
  {
  public:
    /**
     * @brief Constructor
     *
     */
    dispersionCorrection(const MPI_Comm &mpi_comm_parent,
                         const MPI_Comm &mpi_comm_domain,
                         const MPI_Comm &interpool_comm,
                         const MPI_Comm &interBandGroupComm);

    /**
     * Wrapper function for various dispersion corrections to energy, force and
     * stress.
     *
     * @param atomLocations
     * @param d_domainBoundingVectors
     */
    void
    computeDispresionCorrection(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &d_domainBoundingVectors);

    double
    getEnergyCorrection() const;

    double
    getForceCorrection(int atomNo, int dim) const;

    double
    getStressCorrection(int dim1, int dim2) const;

  private:
    int                   d_natoms;
    double                d_energyDispersion;
    std::vector<double>   d_forceDispersion;
    std::array<double, 9> d_stressDispersion;
    std::vector<double>   d_atomCoordinates;
    std::vector<int>      d_atomicNumbers;
    std::array<double, 9> d_latticeVectors;


    const MPI_Comm mpi_communicator_global;
    const MPI_Comm mpi_communicator_domain;
    const MPI_Comm interpoolcomm;
    const MPI_Comm interBandGroupComm;


    // initialize the variables needed for dispersion correction calculations
    void
    initDispersionCorrection(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &d_domainBoundingVectors);


    // Compute D3/4 correction
    void
    computeDFTDCorrection();


    // Compute TS correction, placeholder
  };
} // namespace dftfe
#endif // dispersionCorrection_H_
