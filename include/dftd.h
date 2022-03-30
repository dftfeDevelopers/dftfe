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
#include <dftd3.h>
#include <dftd4.h>

#ifndef dispersionCorrection_H_
#  define dispersionCorrection_H_
namespace dftfe
{
  /**
   * @brief Calculates dispersion correction to energy, force and stress
   *
   */
  class dispersionCorrection
  {

  public:
    /**
     * @brief Constructor
     *
     */
    dispersionCorrection(const MPI_Comm &mpi_comm,
                     const MPI_Comm &interpool_comm,
                     const MPI_Comm &interBandGroupComm,
                     const int n_atoms);

    /**
     * Wrapper function for various dispersion corrections to energy, force and stress.
     *
     * @param atomLocations 
     * @param d_domainBoundingVectors 
     */
    void
    computeDispresionCorrection(
      const std::vector<std::vector<double>>  &atomLocations,
      const std::vector<std::vector<double>>  &d_domainBoundingVectors
    ) ;

    double
    getEnergyCorrection() const;

    double
    getForceCorrection(int atomNo, int dim) const;

    double
    getStressCorrection(int dim1, int dim2) const;

    private:
    int natoms;
    double d_energyDispersion;
    std::vector<double> d_forceDispersion;
    std::array<double,9> d_stressDispersion;
    std::vector<double> atomCoordinates;
    std::vector<int> atomicNumbers;
    std::array<double,9> latticeVectors;


    const MPI_Comm mpi_communicator;
    const MPI_Comm interpoolcomm;
    const MPI_Comm interBandGroupComm;


    //initialize the variables needed for dispersion correction calculations
    void
    initDispersionCorrection(
      const std::vector<std::vector<double>>  &atomLocations,
      const std::vector<std::vector<double>>  &d_domainBoundingVectors
    );


    // Compute D3/4 correction
    void
    computeDFTDCorrection();


    //Compute TS correction, placeholder




  };
}
  #endif // dispersionCorrection_H_
