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

namespace dftfe
{

    class nudgedElasticBandClass : public nonlinearSolverProblem
    {
        public:



    nudgedElasticBandClass(
    const std::string parameter_file,
    const std::string restartFilesPath,
    const MPI_Comm &  mpi_comm_parent,
    const bool        restart,
    const int         verbosity,
    int numberOfImages,
    bool imageFreeze,
    double Kmax,
    double Kmin,
    double pathThreshold,
    int maximumNEBIteration,
    const std::string &coordinatesFileNEB,
    const std::string &domainVectorsFile );
    //~nudgedElasticBandClass();  
            const double haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
            const double haToeV              = 27.211386245988;
            const double bohrToAng           = 0.529177210903;
            const double pi                            = 3.14159265359;
            const double AngTobohr           = 1.0 / bohrToAng;
            double d_kmax                        = 0.1; //0.1 Ha/bohr
            double d_kmin                        = 0.1; //0.1Ha/bohr
            unsigned int NEBImageno                  ;

            int runNEB();         
            void ReturnNormedVector(std::vector<double> & , int);
            void LNorm(double &, std::vector<double> , int, int); 
            void gradient(std::vector<double> &gradient); 
 

            unsigned int
            getNumberUnknowns() const;

            void
            update(const std::vector<double> &solution,
            const bool                 computeForces      = true,
            const bool useSingleAtomSolutionsInitialGuess = false);

            void 
            save();

            void value(std::vector<double> &functionValue);

    /// not implemented
    void
    precondition(std::vector<double> &s, const std::vector<double> &gradient);

            void
            solution(std::vector<double> &solution); 

            
            
            std::vector<unsigned int>
            getUnknownCountFlag() const;      

        private:
            std::vector<std::unique_ptr<dftfeWrapper>> d_dftfeWrapper;
            //std::vector<dftBase *>                     d_dftPtr;
        // parallel communication objects
            const MPI_Comm     d_mpiCommParent;
            //const unsigned int n_mpi_processes;
            const unsigned int d_this_mpi_process;

        // conditional stream object
            dealii::ConditionalOStream pcout;
    
        int d_verbosity;
        std::string d_restartFilesPath;
        bool d_imageFreeze;
        
        /// total number of calls to update()
            unsigned int d_totalUpdateCalls;    
            int           d_startStep;  
            
            unsigned int d_restartFlag;
            unsigned int d_numberGlobalCharges; 
            double d_maximumAtomForceToBeRelaxed; 
            unsigned int d_numberOfImages;
            unsigned int d_maximumNEBIteration;
            double d_optimizertolerance;
            unsigned int optimizermatItr;
            double       Forcecutoff;
            unsigned int d_countrelaxationFlags;
            std::vector<unsigned int> d_relaxationFlags;
             std::vector<double>       d_externalForceOnAtom;
            std::vector<double> d_ForceonImages;
            std::vector<double> d_ImageError;
            std::vector<double> d_Length;
            std::string d_coordinatesFileNEB, d_domainVectorsFileNEB;
            const MPI_Comm &
            getMPICommunicator();
            void CalculatePathTangent(int , std::vector<double> &);
            void CalculateForceparallel(int , std::vector<double> & , std::vector<double> );
            void CalculateForceperpendicular(int , std::vector<double> & , std::vector<double> , std::vector<double>);
            void CalculateSpringForce(int , std::vector<double> &, std::vector<double> );
            void CalculateForceonImage(std::vector<double>, std::vector<double>, std::vector<double> &);
            void CalculatePathLength(double &);
            void WriteRestartFiles(int step);
            void CalculateSpringConstant(int, double & );
            void ImageError(int image, double &Force);

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

    
    };


}// namespace dftfe
#endif