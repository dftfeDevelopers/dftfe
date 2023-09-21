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
#include <cgPRPNonLinearSolver.h>
#include <BFGSNonLinearSolver.h>
#include <LBFGSNonLinearSolver.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <iomanip>
#include <sys/stat.h>
namespace dftfe
{
  class nudgedElasticBandClass : public nonlinearSolverProblem
  {
  public:
    /**
     * @brief First, sets the nebRestart path. Second, creates Step0 folder with coordinaes and domainVectors file.
     * Third, creates the array of pointers of dftClass for each image.
     * If in restart mode, calls function to read coordinates and initialise
     * parameters Sets solvermode: CGPT, LBFGS, BFGS
     */

    nudgedElasticBandClass(const std::string  parameter_file,
                           const std::string  restartFilesPath,
                           const MPI_Comm &   mpi_comm_parent,
                           const bool         restart,
                           const int          verbosity,
                           const int          d_numberOfImages,
                           const bool         imageFreeze,
                           double             Kmax,
                           double             Kmin,
                           const double       pathThreshold,
                           const int          maximumNEBIteration,
                           const unsigned int _maxLineSearchIterCGPRP,
                           const unsigned int _lbfgsNumPastSteps,
                           const std::string &_bfgsStepMethod,
                           const double       optimizermaxIonUpdateStep,
                           const std::string &optimizationSolver,
                           const std::string &coordinatesFileNEB,
                           const std::string &domainVectorsFileNEB,
                           const std::string &ionRelaxFlagsFile);

    //~nudgedElasticBandClass();

    double d_kmax = 0.1; // 0.1 Ha/bohr
    double d_kmin = 0.1; // 0.1Ha/bohr
    int    d_NEBImageno;
    /**
     * @brief Calls optimizer(nonLinearClass) solve. Prints the Final NEB energies and forces.
     * References:
     * 1.
     * https://pubs.aip.org/aip/jcp/article/113/22/9978/184858/Improved-tangent-estimate-in-the-nudged-elastic
     * 2.
     * https://pubs.aip.org/aip/jcp/article/128/13/134106/977389/Optimization-methods-for-finding-minimum-energy
     */
    int
    findMEP();
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
     * First prints the Image No., free energy and force error of each image
     * Prints activation energy of current step
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
     * Calls dftPtr colve to compute eenergy and force for current step.
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
    int          d_totalUpdateCalls;
    int          d_startStep;
    int          d_solver;
    bool         d_isRestart;
    bool         d_solverRestart;
    unsigned int d_restartFlag;
    unsigned int d_numberGlobalCharges;
    double       d_maximumAtomForceToBeRelaxed;
    unsigned int d_numberOfImages;
    unsigned int d_countrelaxationFlags;
    // Solver Details
    unsigned int d_maximumNEBIteration;
    double       d_optimizertolerance;
    unsigned int maxLineSearchIterCGPRP;
    std::string  bfgsStepMethod;
    double       d_optimizermaxIonUpdateStep;
    unsigned int lbfgsNumPastSteps;
    std::string  d_optimizationSolver;
    std::string  d_ionRelaxFlagsFile;


    std::map<int, std::vector<std::vector<double>>> d_atomLocationsInitial;
    std::vector<unsigned int>                       d_relaxationFlags;
    std::vector<double>                             d_externalForceOnAtom;
    std::vector<double>                             d_ImageError;
    std::vector<double>                             d_Length;
    std::string d_coordinatesFileNEB, d_domainVectorsFileNEB;
    const MPI_Comm &
    getMPICommunicator();

    /**
     * @brief Calculate the tangent between each image
     */
    void
    CalculatePathTangent(int, std::vector<double> &);

    /**
     * @brief Calculates the force on atom along the tangent between images
     */
    void
    CalculateForceparallel(int,
                           std::vector<double> &,
                           const std::vector<double> &);
    /**
     * @brief Calculates force perpendicular to the tangent
     */
    void
    CalculateForceperpendicular(int,
                                std::vector<double> &,
                                const std::vector<double> &,
                                const std::vector<double> &);


    /**
     * @brief Calculates the force due to the spring.
     */
    void
    CalculateSpringForce(int, std::vector<double> &, std::vector<double>);

    /**
     * @brief Calculates F_NEB = G_per+ F_spring
     */
    void
    CalculateForceonImage(const std::vector<double> &,
                          const std::vector<double> &,
                          std::vector<double> &);

    /**
     * @brief Calculate path length: max diaplacement of atoms
     */
    double
    CalculatePathLength(bool flag) const;

    /**
     * @brief Write Restart files
     */
    void
    WriteRestartFiles(int step);


    /**
     * @brief Find spring constant based on k_max and k_min.
     */
    void
    CalculateSpringConstant(int, double &);

    /**
     * @brief Calculate F_per norm
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
     * @brief Check the restart files.
     */
    int
    checkRestart(bool &periodic);
  };


} // namespace dftfe
#endif
