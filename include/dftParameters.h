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

#include <string>
#include <mpi.h>

namespace dftfe
{
  /**
   * @brief Namespace which declares the input parameters and the functions to parse them
   *  from the input parameter file
   *
   *  @author Phani Motamarri, Sambit Das
   */
  class dftParameters
  {
  public:
    unsigned int finiteElementPolynomialOrder,
      finiteElementPolynomialOrderElectrostatics, n_refinement_steps,
      numberEigenValues, xc_id, spinPolarized, nkx, nky, nkz, offsetFlagX,
      offsetFlagY, offsetFlagZ;
    unsigned int chebyshevOrder, numPass, numSCFIterations,
      maxLinearSolverIterations, mixingHistory, npool,
      numberWaveFunctionsForEstimate, numLevels,
      maxLinearSolverIterationsHelmholtz;

    bool        poissonGPU;
    std::string modelXCInputFile;

    double radiusAtomBall, mixingParameter;
    bool   adaptAndersonMixingParameter;
    double absLinearSolverTolerance, selfConsistentSolverTolerance, TVal,
      start_magnetization, absLinearSolverToleranceHelmholtz;

    bool isPseudopotential, periodicX, periodicY, periodicZ, useSymm,
      timeReversal, pseudoTestsFlag, constraintMagnetization, writeDosFile,
      writeLdosFile, writeBandsFile, writeLocalizationLengths, pinnedNodeForPBC,
      writePdosFile;


    /** parameters for LRD preconditioner **/

    double      startingNormLRDLargeDamping;
    std::string methodSubTypeLRD;
    double      adaptiveRankRelTolLRD;
    double      betaTol;
    double      absPoissonSolverToleranceLRD;
    bool        singlePrecLRD;
    bool        estimateJacCondNoFinalSCFIter;

    /**********************************************/

    std::string coordinatesFile, domainBoundingVectorsFile, kPointDataFile,
      ionRelaxFlagsFile, orthogType, algoType, pseudoPotentialFile,
      restartFolder;

    std::string coordinatesGaussianDispFile;

    double      outerAtomBallRadius, innerAtomBallRadius, meshSizeOuterDomain;
    bool        autoAdaptBaseMeshSize;
    double      meshSizeInnerBall, meshSizeOuterBall;
    double      chebyshevTolerance, topfrac, kerkerParameter;
    std::string optimizationMode, mixingMethod, ionOptSolver, cellOptSolver;


    bool         isIonForce, isCellStress, isBOMD;
    bool         nonSelfConsistentForce, meshAdaption;
    double       forceRelaxTol, stressRelaxTol, toleranceKinetic;
    unsigned int cellConstraintType;

    int         verbosity;
    std::string solverMode;
    bool        keepScratchFolder;
    bool        saveRhoData;
    bool        loadRhoData;
    bool        restartSpinFromNoSpin;

    bool reproducible_output;

    bool writeWfcSolutionFields;

    bool writeDensitySolutionFields;

    bool writeDensityQuadData;

    std::string  startingWFCType;
    bool         restrictToOnePass;
    unsigned int numCoreWfcRR;
    unsigned int numCoreWfcXtHX;
    unsigned int wfcBlockSize;
    unsigned int chebyWfcBlockSize;
    unsigned int subspaceRotDofsBlockSize;
    unsigned int nbandGrps;
    bool         computeEnergyEverySCF;
    unsigned int scalapackParalProcs;
    unsigned int scalapackBlockSize;
    unsigned int natoms;
    unsigned int natomTypes;
    bool         reuseWfcGeoOpt;
    unsigned int reuseDensityGeoOpt;
    double       mpiAllReduceMessageBlockSizeMB;
    bool         useSubspaceProjectedSHEPGPU;
    bool         useMixedPrecCGS_SR;
    bool         useMixedPrecCGS_O;
    bool         useMixedPrecXTHXSpectrumSplit;
    bool         useMixedPrecSubspaceRotRR;
    bool         useMixedPrecCommunOnlyXTHXCGSO;
    unsigned int spectrumSplitStartingScfIter;
    bool         useELPA;
    bool         constraintsParallelCheck;
    bool         createConstraintsFromSerialDofhandler;
    bool         bandParalOpt;
    bool         useDevice;
    bool         useTF32Device;
    bool         deviceFineGrainedTimings;
    bool         allowFullCPUMemSubspaceRot;
    bool         useMixedPrecCheby;
    bool         overlapComputeCommunCheby;
    bool         overlapComputeCommunOrthoRR;
    bool         autoDeviceBlockSizes;
    bool         readWfcForPdosPspFile;
    double       maxJacobianRatioFactorForMD;
    double       chebyshevFilterPolyDegreeFirstScfScalingFactor;
    int          extrapolateDensity;
    double       timeStepBOMD;
    unsigned int numberStepsBOMD;
    unsigned int TotalImages;
    double       gaussianConstantForce;
    double       gaussianOrderForce;
    double       gaussianOrderMoveMeshToAtoms;
    bool         useFlatTopGenerator;
    double       diracDeltaKernelScalingConstant;
    bool         useMeshSizesFromAtomsFile;
    double       xlbomdRestartChebyTol;
    bool         useDensityMatrixPerturbationRankUpdates;
    double       xlbomdKernelRankUpdateFDParameter;
    bool         smearedNuclearCharges;
    bool         HXOptimFlag;
    bool         floatingNuclearCharges;
    bool         nonLinearCoreCorrection;
    unsigned int maxLineSearchIterCGPRP;
    std::string  atomicMassesFile;
    bool         useDeviceDirectAllReduce;
    double       pspCutoffImageCharges;
    bool         reuseLanczosUpperBoundFromFirstCall;
    bool         allowMultipleFilteringPassesAfterFirstScf;
    unsigned int highestStateOfInterestForChebFiltering;
    bool         useELPADeviceKernel;
    bool         deviceMemOptMode;


    unsigned int dc_dispersioncorrectiontype;
    unsigned int dc_d3dampingtype;
    bool         dc_d3ATM;
    bool         dc_d4MBD;
    std::string  dc_dampingParameterFilename;
    double       dc_d3cutoff2;
    double       dc_d3cutoff3;
    double       dc_d3cutoffCN;


    std::string  bfgsStepMethod;
    bool         usePreconditioner;
    unsigned int lbfgsNumPastSteps;
    unsigned int maxOptIter;
    unsigned int maxStaggeredCycles;
    double       maxIonUpdateStep, maxCellUpdateStep;

    // New Paramters for moleculardyynamics class
    double      startingTempBOMD;
    double      MaxWallTime;
    double      thermostatTimeConstantBOMD;
    std::string tempControllerTypeBOMD;
    int         MDTrack;

    bool writeStructreEnergyForcesFileForPostProcess;

    dftParameters();

    /**
     * Parse parameters.
     */
    void
    parse_parameters(const std::string &parameter_file,
                     const MPI_Comm &   mpi_comm_parent,
                     const bool         printParams      = false,
                     const std::string  mode             = "GS",
                     const std::string  restartFilesPath = ".",
                     const int          _verbosity       = 1);

    /**
     * Check parameters
     */
    void
    check_parameters(const MPI_Comm &mpi_comm_parent) const;

    /**
     * Set automated choices for parameters
     */
    void
    setAutoParameters(const MPI_Comm &mpi_comm_parent);

  }; // class dftParameters

} // namespace dftfe
#endif
