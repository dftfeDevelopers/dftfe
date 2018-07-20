// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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



#ifndef dftParameters_H_
#define dftParameters_H_

#include <string>
#include <deal.II/base/parameter_handler.h>

namespace dftfe {
    // FIXME: document Parameters
    // FIXME: this should really be an object, not global values
    /**
     * @brief Namespace which declares the input parameters and the functions to parse them
     *  from the input parameter file
     *
     *  @author Phani Motamarri, Sambit Das
     */
    namespace dftParameters
    {

      extern unsigned int finiteElementPolynomialOrder,n_refinement_steps,numberEigenValues,xc_id, spinPolarized, nkx,nky,nkz , offsetFlagX,offsetFlagY,offsetFlagZ;
      extern unsigned int chebyshevOrder,numPass,numSCFIterations,maxLinearSolverIterations, mixingHistory, npool;

      extern double radiusAtomBall, mixingParameter;
      extern double lowerEndWantedSpectrum,relLinearSolverTolerance,selfConsistentSolverTolerance,TVal, start_magnetization;

      extern bool isPseudopotential, periodicX, periodicY, periodicZ, useSymm, timeReversal,pseudoTestsFlag;
      extern std::string meshFileName,coordinatesFile,domainBoundingVectorsFile,kPointDataFile, ionRelaxFlagsFile, orthogType,pseudoPotentialFile;

      extern double outerAtomBallRadius, meshSizeOuterDomain;
      extern double meshSizeInnerBall, meshSizeOuterBall;
      extern double chebyshevTolerance;


      extern bool isIonOpt, isCellOpt, isIonForce, isCellStress;
      extern bool nonSelfConsistentForce;
      extern double forceRelaxTol, stressRelaxTol;
      extern unsigned int cellConstraintType;

      extern unsigned int verbosity, chkType;
      extern bool restartFromChk;

      extern bool reproducible_output;

      extern bool writeWfcSolutionFields;

      extern bool writeDensitySolutionFields;

      extern std::string startingWFCType;
      extern unsigned int numCoreWfcRR;
      extern unsigned int chebyshevBlockSize;
      extern bool useBatchGEMM;
      extern unsigned int orthoRRWaveFuncBlockSize;
      extern unsigned int subspaceRotDofsBlockSize;
      extern bool enableSwitchToGS;
      extern unsigned int nbandGrps;
      extern bool computeEnergyEverySCF;
      extern unsigned int scalapackParalProcs;
      extern unsigned int natoms;
      extern unsigned int natomTypes;
      extern double lowerBoundUnwantedFracUpper;

      /**
       * Declare parameters.
       */
      void declare_parameters(dealii::ParameterHandler &prm);

      /**
       * Parse parameters.
       */
      void parse_parameters(dealii::ParameterHandler &prm);

      /**
       * Check and print parameters
       */
      void check_print_parameters(const dealii::ParameterHandler &prm);

      /**
       * Check and print parameters
       */
      void setHeuristicParameters();

    };

}
#endif
