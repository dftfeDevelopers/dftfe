// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#include "nonlinearSolverProblem.h"
#include "constants.h"

namespace dftfe {

    using namespace dealii;
    template <unsigned int FEOrder> class dftClass;

    /** @file geoOptIon.h
     *
     *  @brief problem class for atomic force relaxation solver.
     *
     *  @author Sambit Das
     */
    template <unsigned int FEOrder>
    class geoOptIon : public nonlinearSolverProblem
    {
    public:
    /** @brief Constructor.
     *
     *  @param _dftPtr pointer to dftClass
     *  @param mpi_comm_replica mpi_communicator of the current pool
     */
      geoOptIon(dftClass<FEOrder>* _dftPtr,const  MPI_Comm &mpi_comm_replica);

    /**
     * @brief initializes the data member d_relaxationFlags.
     *
     */
      void init();

    /**
     * @brief calls the atomic force relaxation solver.
     *
     * Currently we have option of one solver: Polak–Ribière nonlinear CG solver
     * with secant based line search. In future releases, we will have more options like BFGS solver.
     *
     */
      void run();

    /**
     * @brief Obtain number of unknowns (total number of force components to be relaxed).
     *
     * @return int Number of unknowns.
     */
     unsigned int getNumberUnknowns() const ;

    /**
     * @brief Compute function gradient (aka forces).
     *
     * @param gradient STL vector for gradient values.
     */
      void gradient(std::vector<double> & gradient);

    /**
     * @brief Update atomic positions.
     *
     * @param solution displacement of the atoms with respect to their current position.
     * The size of the solution vector is equal to the number of unknowns.
     */
      void update(const std::vector<double> & solution);

      /// not implemented
      void value(std::vector<double> & functionValue);

      /// not implemented
      void precondition(std::vector<double>       & s,
			const std::vector<double> & gradient) const;

      /// not implemented
      void solution(std::vector<double> & solution);

      /// not implemented
      std::vector<unsigned int> getUnknownCountFlag() const;

    private:

      /// storage for relaxation flags for each global atom.
      /// each atom has three flags corresponding to three components (0- no relax, 1- relax)
      std::vector<unsigned int> d_relaxationFlags;

      /// maximum force component to be relaxed
      double d_maximumAtomForceToBeRelaxed;

      /// total number of calls to update()
      unsigned int d_totalUpdateCalls;

      /// pointer to dft class
      dftClass<FEOrder>* dftPtr;

      /// parallel communication objects
      const MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;

      /// conditional stream object
      dealii::ConditionalOStream   pcout;
    };

}
#endif
