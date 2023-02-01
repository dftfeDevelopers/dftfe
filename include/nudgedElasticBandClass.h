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



#ifndef nudgedElasticBandClass_H_
#define nudgedElasticBandClass_H_
#include <vector>
#include "nonlinearSolverProblem.h"
#include "nonLinearSolver.h"
#include "dftBase.h"
#include "dftfeWrapper.h"
#include "headers.h"
#include "constants.h"

namespace dftfe
{
  class nudgedElasticBandClass : public nonlinearSolverProblem
  {
  public:
    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */

    nudgedElasticBandClass(const std::string  parameter_file,
                           const std::string  restartFilesPath,
                           const MPI_Comm &   mpi_comm_parent,
                           const bool         restart,
                           const int          verbosity,
                           int                numberOfImages,
                           bool               imageFreeze,
                           double             Kmax,
                           double             Kmin,
                           double             pathThreshold,
                           int                maximumNEBIteration,
                           const std::string &coordinatesFileNEB,
                           const std::string &domainVectorsFileNEB);

    //~nudgedElasticBandClass();

    double d_kmax = 0.1; // 0.1 Ha/bohr
    double d_kmin = 0.1; // 0.1Ha/bohr
    int    d_NEBImageno;
    /**
     * @brief Calls optimizer(nonLinearClass) solve. Prints the Final NEB energies and forces.
     */
    int
    run();
    /**
     * @brief Returns the Normed vetor satistfying ||v||_2 = 1
     */
    void
    ReturnNormedVector(std::vector<double> &, int);
    /**
     * @brief Calculates the L-norm of a vector
     */
    void
    LNorm(double &, std::vector<double>, int, int);
    /**
     * @brief Identifies the images to freeze, calculates gradient.
     */
    void
    gradient(std::vector<double> &gradient);
    /**
     * @brief Returns the total DoFs of the optimizer problem.
     */
    unsigned int
    getNumberUnknowns() const;
    /**
     * @brief Updates the positions of atoms and the total step count.
     */
    void
    update(const std::vector<double> &solution,
           const bool                 computeForces      = true,
           const bool useSingleAtomSolutionsInitialGuess = false);
    /**
     * @brief Saves the output files for restart.
     */
    void
    save();
    /**
     * @brief initializes the data member d_relaxationFlags, nonlinearSolver,
     *
     */
    void
    init();
    /**
     * @brief Not working. Finds the saddle point energy.
     */
    void
    value(std::vector<double> &functionValue);

    /// not implemented
    void
    precondition(std::vector<double> &s, const std::vector<double> &gradient);
    /// not implemented
    void
    solution(std::vector<double> &solution);
    /// not implemented
    std::vector<unsigned int>
    getUnknownCountFlag() const;

  private:
    std::vector<std::unique_ptr<dftfeWrapper>> d_dftfeWrapper;
    dftBase *                                  d_dftPtr;
    std::unique_ptr<nonLinearSolver>           d_nonLinearSolverPtr;
    // parallel communication objects
    const MPI_Comm d_mpiCommParent;
    // const unsigned int n_mpi_processes;
    const unsigned int d_this_mpi_process;

    // conditional stream object
    dealii::ConditionalOStream pcout;

    int         d_verbosity;
    std::string d_restartFilesPath, d_solverRestartPath;
    bool        d_imageFreeze;

    /// total number of calls to update()
    int                 d_totalUpdateCalls;
    int                 d_startStep;
    int                 d_solver;
    bool                d_isRestart;
    bool                d_solverRestart;
    unsigned int        d_restartFlag;
    unsigned int        d_numberGlobalCharges;
    double              d_maximumAtomForceToBeRelaxed;
    unsigned int        d_numberOfImages;
    unsigned int        d_maximumNEBIteration;
    double              d_optimizertolerance;
    unsigned int        optimizermatItr;
    double              Forcecutoff;
    unsigned int        d_countrelaxationFlags;
    std::vector<double> d_forceOnImages;
    std::map<int, std::vector<std::vector<double>>> d_atomLocationsInitial;
    std::vector<unsigned int>                       d_relaxationFlags;
    std::vector<double>                             d_externalForceOnAtom;
    std::vector<double>                             d_ImageError;
    std::vector<double>                             d_Length;
    std::string d_coordinatesFileNEB, d_domainVectorsFileNEB;
    const MPI_Comm &
    getMPICommunicator();

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculatePathTangent(int, std::vector<double> &);

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculateForceparallel(int,
                           std::vector<double> &,
                           const std::vector<double> &);
    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculateForceperpendicular(int,
                                std::vector<double> &,
                                const std::vector<double> &,
                                const std::vector<double> &);


    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculateSpringForce(int, std::vector<double> &, std::vector<double>);

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculateForceonImage(const std::vector<double> &,
                          const std::vector<double> &,
                          std::vector<double> &);

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculatePathLength(double &) const;

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    WriteRestartFiles(int step);


    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    CalculateSpringConstant(int, double &);

    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    void
    ImageError(int image, double &Force);

    /**
     * @brief  set() initalises all the private datamembers of nudgedElasticBandClass object from the parameters declared by user.
     */
    void
    set();

    /**
     * @brief check for convergence.
     *
     */
    bool
    isConverged() const;
    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     */
    int
    checkRestart(bool &periodic);
  };


} // namespace dftfe
#endif
