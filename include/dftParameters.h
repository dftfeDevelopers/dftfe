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
//



#ifndef dftParameters_H_
#define dftParameters_H_

#include <deal.II/base/parameter_handler.h>

#include <string>

namespace dftfe
{
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
    extern unsigned int finiteElementPolynomialOrder,
      finiteElementPolynomialOrderElectrostatics, n_refinement_steps,
      numberEigenValues, xc_id, spinPolarized, nkx, nky, nkz, offsetFlagX,
      offsetFlagY, offsetFlagZ;
    extern unsigned int chebyshevOrder, numPass, numSCFIterations,
      maxLinearSolverIterations, mixingHistory, npool,
      numberWaveFunctionsForEstimate, numLevels,
      maxLinearSolverIterationsHelmholtz;

    extern double radiusAtomBall, mixingParameter;
    extern double absLinearSolverTolerance, selfConsistentSolverTolerance, TVal,
      start_magnetization, absLinearSolverToleranceHelmholtz;

    extern bool isPseudopotential, periodicX, periodicY, periodicZ, useSymm,
      timeReversal, pseudoTestsFlag, constraintMagnetization, writeDosFile,
      writeLdosFile, writeLocalizationLengths, pinnedNodeForPBC, writePdosFile;
    extern std::string coordinatesFile, domainBoundingVectorsFile,
      kPointDataFile, ionRelaxFlagsFile, orthogType, algoType,
      pseudoPotentialFile;

    extern std::string coordinatesGaussianDispFile;

    extern double outerAtomBallRadius, innerAtomBallRadius, meshSizeOuterDomain;
    extern bool   autoAdaptBaseMeshSize;
    extern double meshSizeInnerBall, meshSizeOuterBall;
    extern double chebyshevTolerance, topfrac, kerkerParameter;
    extern std::string mixingMethod, ionOptSolver;


    extern bool isIonOpt, isCellOpt, isIonForce, isCellStress, isBOMD, isXLBOMD;
    extern bool nonSelfConsistentForce, meshAdaption;
    extern double       forceRelaxTol, stressRelaxTol, toleranceKinetic;
    extern unsigned int cellConstraintType;

    extern unsigned int verbosity, chkType;
    extern bool         restartSpinFromNoSpin;
    extern bool         restartFromChk;
    extern bool         restartMdFromChk;
    extern bool         electrostaticsHRefinement;

    extern bool reproducible_output;

    extern bool writeWfcSolutionFields;

    extern bool writeDensitySolutionFields;

    extern std::string  startingWFCType;
    extern unsigned int numCoreWfcRR;
    extern unsigned int wfcBlockSize;
    extern unsigned int chebyWfcBlockSize;
    extern unsigned int subspaceRotDofsBlockSize;
    extern unsigned int nbandGrps;
    extern bool         computeEnergyEverySCF;
    extern unsigned int scalapackParalProcs;
    extern unsigned int scalapackBlockSize;
    extern unsigned int natoms;
    extern unsigned int natomTypes;
    extern bool         reuseWfcGeoOpt;
    extern unsigned int reuseDensityGeoOpt;
    extern double       mpiAllReduceMessageBlockSizeMB;
    extern bool         useMixedPrecCGS_SR;
    extern bool         useMixedPrecCGS_O;
    extern bool         useMixedPrecXTHXSpectrumSplit;
    extern bool         useMixedPrecSubspaceRotRR;
    extern unsigned int spectrumSplitStartingScfIter;
    extern bool         useELPA;
    extern bool         constraintsParallelCheck;
    extern bool         createConstraintsFromSerialDofhandler;
    extern bool         bandParalOpt;
    extern bool         useGPU;
    extern bool         gpuFineGrainedTimings;
    extern bool         allowFullCPUMemSubspaceRot;
    extern bool         useMixedPrecCheby;
    extern bool         overlapComputeCommunCheby;
    extern bool         overlapComputeCommunOrthoRR;
    extern bool         autoGPUBlockSizes;
    extern bool         readWfcForPdosPspFile;
    extern double       maxJacobianRatioFactorForMD;
    extern double       chebyshevFilterTolXLBOMD;
    extern double       chebyshevFilterTolXLBOMDRankUpdates;
    extern double       chebyshevFilterPolyDegreeFirstScfScalingFactor;
    extern double       timeStepBOMD;
    extern unsigned int numberStepsBOMD;
    extern unsigned int TotalImages;
    extern std::string  solvermode;
    // extern double       startingTempBOMDNVE;
    extern double       gaussianConstantForce;
    extern double       gaussianOrderForce;
    extern double       gaussianOrderMoveMeshToAtoms;
    extern bool         useFlatTopGenerator;
    extern double       diracDeltaKernelScalingConstant;
    extern unsigned int kernelUpdateRankXLBOMD;
    extern unsigned int kmaxXLBOMD;
    extern bool         useAtomicRhoXLBOMD;
    extern bool         useMeshSizesFromAtomsFile;
    extern unsigned int numberPassesRRSkippedXLBOMD;
    extern double       xlbomdRestartChebyTol;
    extern bool         useDensityMatrixPerturbationRankUpdates;
    extern double       xlbomdKernelRankUpdateFDParameter;
    extern bool         smearedNuclearCharges;
    extern bool         HXOptimFlag;
    extern bool         floatingNuclearCharges;
    extern bool         nonLinearCoreCorrection;
    extern unsigned int maxLineSearchIterCGPRP;
    extern std::string  atomicMassesFile;
    extern bool         useGPUDirectAllReduce;
    extern double       pspCutoffImageCharges;
    extern bool         reuseLanczosUpperBoundFromFirstCall;
    extern bool         allowMultipleFilteringPassesAfterFirstScf;
    extern bool         useELPAGPUKernel;
    extern std::string  xcFamilyType;
    extern bool         gpuMemOptMode;
    // New Paramters for moleculardyynamics class
    extern double      startingTempBOMD;
    extern double      MaxWallTime;
    extern double      thermostatTimeConstantBOMD;
    extern std::string tempControllerTypeBOMD;
    extern int         MDTrack;
    /**
     * Declare parameters.
     */
    void
    declare_parameters(dealii::ParameterHandler &prm);

    /**
     * Parse parameters.
     */
    void
    parse_parameters(dealii::ParameterHandler &prm,const MPI_Comm & mpi_comm_parent);

    /**
     * Check and print parameters
     */
    void
    check_print_parameters(const dealii::ParameterHandler &prm,const MPI_Comm & mpi_comm_parent);

    /**
     * Set automated choices for parameters
     */
    void
    setAutoParameters(const MPI_Comm & mpi_comm_parent);

    /**
     * set family type exchange correlation functional
     *
     */
    void
    setXCFamilyType();
  }; // namespace dftParameters

} // namespace dftfe
#endif
