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

#ifndef geoOptCell_H_
#define geoOptCell_H_
#include "constants.h"
#include "nonlinearSolverProblem.h"

namespace dftfe
{
  using namespace dealii;
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class dftClass;

  /**
   * @brief problem class for cell stress relaxation solver.
   *
   * @author Sambit Das
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class geoOptCell : public nonlinearSolverProblem
  {
  public:
    /** @brief Constructor.
     *
     *  @param _dftPtr pointer to dftClass
     *  @param mpi_comm_parent parent mpi_communicator
     */
    geoOptCell(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
               const MPI_Comm &                   mpi_comm_replica);

    /**
     * @brief initializes the data member d_relaxationFlags.
     *
     */
    void
    init();

    /**
     * @brief calls the cell stress relaxation solver.
     *
     * The Polak–Ribière nonlinear CG solver with secant based line search
     * is used for the stress relaxation.
     *
     * @return int total geometry update calls
     */
    int
    run();

    /**
     * @brief writes the current fem mesh.
     *
     */
    void
    writeMesh(std::string meshFileName);

    /**
     * @brief Obtain number of unknowns (depends on the stress relaxation constraint type).
     *
     * @return int Number of unknowns.
     */
    unsigned int
    getNumberUnknowns() const;

    /**
     * @brief Compute function gradient (stress).
     *
     * @param gradient STL vector for gradient values.
     */
    void
    gradient(std::vector<double> &gradient);

    /**
     * @brief Update the strain tensor epsilon.
     *
     * The new strain tensor is epsilonNew= epsilon+ delta(epsilon). Since
     * epsilon strain is already applied to the domain. The new strain to be
     * applied to the domain is epsilonNew*inv(epsilon)
     *
     * @param solution delta(epsilon).
     */
    void
    update(const std::vector<double> &solution,
           const bool                 computeStress      = true,
           const bool useSingleAtomSolutionsInitialGuess = false);

    /**
     * @brief create checkpoint file for current domain bounding vectors and atomic coordinates.
     *
     */
    void
    save();

    const MPI_Comm &
    getMPICommunicator();

    /// Not implemented
    void
    value(std::vector<double> &functionValue);

    /// Not implemented
    void
    precondition(std::vector<double> &      s,
                 const std::vector<double> &gradient) const;

    /// Not implemented
    void
    solution(std::vector<double> &solution);

    /// Not implemented
    std::vector<unsigned int>
    getUnknownCountFlag() const;

  private:
    /// Relaxation flags which determine whether a particular stress component
    /// is to be relaxed or not.
    //  The relaxation flags are determined based on the stress relaxation
    //  constraint type.
    std::vector<unsigned int> d_relaxationFlags;

    /// total number of calls to update()
    unsigned int d_totalUpdateCalls;

    /// current strain tensor applied on the domain
    Tensor<2, 3, double> d_strainEpsilon;

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
