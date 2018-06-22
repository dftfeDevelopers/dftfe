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
//


/** @file dftParameters.h
 *
 *  @author Phani Motamarri, Sambit Das
 */

#ifndef dftParameters_H_
#define dftParameters_H_

#include <string>
#include <deal.II/base/parameter_handler.h>

namespace dftfe {
    // FIXME: document Parameters
    // FIXME: this should really be an object, not global values
    //
    //Declare dftUtils functions
    //
    namespace dftParameters
    {

      extern unsigned int finiteElementPolynomialOrder,n_refinement_steps,numberEigenValues,xc_id, spinPolarized, nkx,nky,nkz, pseudoProjector;
      extern unsigned int chebyshevOrder,numPass,numSCFIterations,maxLinearSolverIterations, mixingHistory, npool;

      extern double radiusAtomBall, mixingParameter, dkx, dky, dkz;
      extern double lowerEndWantedSpectrum,relLinearSolverTolerance,selfConsistentSolverTolerance,TVal, start_magnetization;

      extern bool isPseudopotential,periodicX,periodicY,periodicZ, useSymm, timeReversal;
      extern std::string meshFileName,coordinatesFile,domainBoundingVectorsFile,kPointDataFile, ionRelaxFlagsFile, orthogType;

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

      extern bool electrostaticsPRefinement;

      extern bool writeSolutionFields;

      extern std::string startingWFCType;
      extern unsigned int chebyshevBlockSize;
      extern bool useBatchGEMM;
      extern unsigned int chebyshevOMPThreads;
      extern unsigned int orthoRROMPThreads;
      extern bool cacheShapeGradData;
      extern unsigned int orthoRRWaveFuncBlockSize;
      extern unsigned int subspaceRotDofsBlockSize;
      extern bool enableSwitchToGS;
      extern unsigned int nbandGrps;
      extern bool computeEnergyEverySCF;

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

    };

}
#endif
