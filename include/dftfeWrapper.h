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

#ifndef dftfeWrapper_H_
#define dftfeWrapper_H_

#include <mpi.h>
#include <string>
#include <vector>

namespace dftfe
{
  class dftBase;
  class dftParameters;
  /**
   * @brief wrapper class for dftfe
   *
   * @author Sambit Das
   */
  class dftfeWrapper
  {
  public:
    static void
    globalHandlesInitialize();

    static void
    globalHandlesFinalize();

    dftfeWrapper(const std::string parameter_file,
                 const MPI_Comm &  mpi_comm_parent,
                 const bool        printParams                      = false,
                 const bool        setGPUToMPITaskBindingInternally = false);

    ~dftfeWrapper();

    void
    reinit(const std::string parameter_file,
           const MPI_Comm &  mpi_comm_parent,
           const bool        printParams                      = false,
           const bool        setGPUToMPITaskBindingInternally = false);

    void
    clear();

    /**
     * @brief Legacy function (to be deprecated)
     */
    void
    run();

    /**
     * @brief solve ground-state and return DFT free energy which is sum of internal
     * energy and negative of electronic entropic energy
     */
    double
    computeDFTFreeEnergy();


    /**
     * @brief Get physical forces on atoms
     *  @return std::vector<double> flattened array of forces with dimension index being
     *  the fastest index i.e atom1forcex atom1forcey atom1forcz atom2forcex ...
     */
    std::vector<double>
    getForcesAtoms();

    /**
     * @brief update atom positions and reinitialize all related  datastructures
     *  @param[in] std::vector<double> flattened array of displacements
     *  with dimension index being the fastest index
     *  i.e atom1dispx atom1dispy atom1dispz atom2dispx ...
     */
    void
    updateAtomPositions(const std::vector<double> atomsDisplacements);

    /**
     * @brief Gets the current atom Locations in cartesian form
     * (origin at center of domain)
     *
     *  @return std::vector<double> flattened array of coords with dimension index being
     *  the fastest index i.e atom1coordx atom1coordy atom1coordz atom2coordx
     * ...
     */
    std::vector<double>
    getAtomLocationsCart();

    /**
     * @brief Gets the current atom Locations in fractional form
     * (only applicable for periodic and semi-periodic BCs).
     * CAUTION: during relaxation and MD fractional coordinates may have negaive
     * values
     *
     *  @return std::vector<double> flattened array of coords with dimension index being
     *  the fastest index i.e atom1coordx atom1coordy atom1coordz atom2coordx
     * ...
     */
    std::vector<double>
    getAtomLocationsFrac();


    dftBase *
    getDftfeBasePtr();

  private:
    dftBase *      d_dftfeBasePtr;
    dftParameters *d_dftfeParamsPtr;
  };
} // namespace dftfe
#endif
