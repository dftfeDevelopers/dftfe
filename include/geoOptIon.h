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

#ifndef geoOptIon_H_
#define geoOptIon_H_
#include "constants.h"
#include "nonlinearSolverProblem.h"

namespace dftfe
{
  using namespace dealii;
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class dftClass;

  /**
   * @brief problem class for atomic force relaxation solver.
   *
   * @author Sambit Das
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class geoOptIon : public nonlinearSolverProblem
  {
  public:
    /** @brief Constructor.
     *
     *  @param _dftPtr pointer to dftClass
     *  @param mpi_comm_parent parent mpi_communicator
     */
    geoOptIon(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
              const MPI_Comm &                   mpi_comm_parent);

    /**
     * @brief initializes the data member d_relaxationFlags.
     *
     */
    void
    init();

    /**
     * @brief calls the atomic force relaxation solver.
     *
     * Currently we have option of one solver: Polak–Ribière nonlinear CG solver
     * with secant based line search. In future releases, we will have more
     * options like BFGS solver.
     *
     * @return int total geometry update calls
     */
    int
    run();

    /**
     * @brief Obtain number of unknowns (total number of force components to be relaxed).
     *
     * @return int Number of unknowns.
     */
    unsigned int
    getNumberUnknowns() const;

    /**
     * @brief Compute function gradient (aka forces).
     *
     * @param gradient STL vector for gradient values.
     */
    void
    gradient(std::vector<double> &gradient);

    /**
     * @brief Update atomic positions.
     *
     * @param solution displacement of the atoms with respect to their current position.
     * The size of the solution vector is equal to the number of unknowns.
     */
    void
    update(const std::vector<double> &solution,
           const bool                 computeForces      = true,
           const bool useSingleAtomSolutionsInitialGuess = false);

    /**
     * @brief create checkpoint file for current domain bounding vectors and atomic coordinates.
     *
     */
    void
    save();

    const MPI_Comm &
    getMPICommunicator();

    /// not implemented
    void
    value(std::vector<double> &functionValue);

    /// not implemented
    void
    precondition(std::vector<double> &      s,
                 const std::vector<double> &gradient) const;

    /// not implemented
    void
    solution(std::vector<double> &solution);

    /// not implemented
    std::vector<unsigned int>
    getUnknownCountFlag() const;

  private:
    /// storage for relaxation flags and external force components for each
    /// global atom. each atom has three flags corresponding to three components
    /// (0- no relax, 1- relax) and three external force components
    std::vector<unsigned int> d_relaxationFlags;
    std::vector<double>       d_externalForceOnAtom;

    /// maximum force component to be relaxed
    double d_maximumAtomForceToBeRelaxed;

    /// total number of calls to update()
    unsigned int d_totalUpdateCalls;

    /// pointer to dft class
    dftClass<FEOrder, FEOrderElectro> *dftPtr;

    /// parallel communication objects
    const MPI_Comm     mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    /// conditional stream object
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif
