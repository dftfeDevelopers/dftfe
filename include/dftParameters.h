// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
#include <TypeConfig.h>

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
    dftfe::uInt finiteElementPolynomialOrder,
      finiteElementPolynomialOrderRhoNodal,
      finiteElementPolynomialOrderElectrostatics, n_refinement_steps,
      numberEigenValues, spinPolarized, nkx, nky, nkz, offsetFlagX, offsetFlagY,
      offsetFlagZ;
    dftfe::uInt densityQuadratureRule;
    dftfe::uInt chebyshevOrder, numPass, numSCFIterations,
      maxLinearSolverIterations, mixingHistory, npool, numLevels,
      maxLinearSolverIterationsHelmholtz;

    bool        poissonGPU;
    bool        vselfGPU;
    std::string XCType;
    std::string modelXCInputFile;
    std::string auxBasisTypeXC;
    std::string auxBasisDataXC;
    bool        useLibXCForXCEvaluation;

    double radiusAtomBall, mixingParameter, spinMixingEnhancementFactor;
    bool   adaptAndersonMixingParameter;
    double absLinearSolverTolerance, selfConsistentSolverTolerance, TVal,
      selfConsistentSolverEnergyTolerance, tot_magnetization,
      absLinearSolverToleranceHelmholtz, smearTval, intervalSize;

    bool useAtomicMagnetizationGuessConstraintMag;

    bool isPseudopotential, periodicX, periodicY, periodicZ, useSymm,
      timeReversal, pseudoTestsFlag, constraintMagnetization, writeDosFile,
      writeLdosFile, writeLocalizationLengths, pinnedNodeForPBC, writePdosFile;

    bool pureState;

    double netCharge;

    /** parameters for functional tests **/
    std::string functionalTestName;
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
      tensorOpType, restartFolder, meshSizesFile;

    std::string coordinatesGaussianDispFile;

    double outerAtomBallRadius, innerAtomBallRadius, meshSizeOuterDomain;
    bool   autoAdaptBaseMeshSize;
    double meshSizeInnerBall, meshSizeOuterBall;
    double chebyshevTolerance, topfrac, kerkerParameter, restaScreeningLength,
      restaFermiWavevector;
    std::string optimizationMode, mixingMethod, ionOptSolver, cellOptSolver;

    std::string hubbardFileName;
    bool        isIonForce, isCellStress, isBOMD;
    bool        nonSelfConsistentForce;
    double      forceRelaxTol, stressRelaxTol;
    dftfe::uInt cellConstraintType;

    dftfe::Int  verbosity;
    std::string solverMode;
    bool        keepScratchFolder;
    bool        saveQuadData;
    bool        loadQuadData;
    bool        restartSpinFromNoSpin;

    bool reproducible_output;

    bool writeWfcSolutionFields;
    bool printKE;

    bool writeDensitySolutionFields;

    bool writeDensityQuadData;

    std::string startingWFCType;
    bool        restrictToOnePass;
    dftfe::uInt numCoreWfcForMixedPrecRR;
    dftfe::uInt wfcBlockSize;
    dftfe::uInt chebyWfcBlockSize;
    dftfe::uInt subspaceRotDofsBlockSize;
    dftfe::uInt nbandGrps;
    bool        computeEnergyEverySCF;
    bool        useEnergyResidualTolerance;
    dftfe::uInt scalapackParalProcs;
    dftfe::uInt scalapackBlockSize;
    dftfe::uInt natoms;
    dftfe::uInt natomTypes;
    bool        reuseWfcGeoOpt;
    dftfe::uInt reuseDensityGeoOpt;
    double      mpiAllReduceMessageBlockSizeMB;
    bool        useSubspaceProjectedSHEPGPU;
    bool        useMixedPrecCGS_SR;
    bool        useMixedPrecXtOX;
    bool        useMixedPrecXtHX;
    bool        useMixedPrecSubspaceRotRR;
    bool        useMixedPrecCommunOnlyXtHXXtOX;
    bool        useELPA;
    bool        constraintsParallelCheck;
    bool        createConstraintsFromSerialDofhandler;
    bool        bandParalOpt;
    bool        useDevice;
    bool        deviceFineGrainedTimings;
    bool        allowFullCPUMemSubspaceRot;
    std::string     communPrecCheby;
    bool            useSinglePrecCheby;
    mutable dftfe::uInt scfIterCount = 0;
    dftfe::uInt     compressLateStartSCF = 6;
    dftfe::uInt     compressBitsPerValueEarly = 16;
    dftfe::uInt     compressBitsPerValueLate  = 8;
    std::string     compressAlgoEarly = "BFP";
    std::string     compressAlgoLate  = "BFP";
    bool        overlapComputeCommunCheby;
    bool        overlapComputeCommunOrthoRR;
    bool        autoDeviceBlockSizes;
    bool        readWfcForPdosPspFile;
    double      maxJacobianRatioFactorForMD;
    double      chebyshevFilterPolyDegreeFirstScfScalingFactor;
    dftfe::Int  extrapolateDensity;
    double      timeStepBOMD;
    dftfe::uInt numberStepsBOMD;
    dftfe::uInt TotalImages;
    double      gaussianConstantForce;
    double      gaussianOrderForce;
    double      gaussianOrderMoveMeshToAtoms;
    bool        useFlatTopGenerator;
    double      diracDeltaKernelScalingConstant;
    double      xlbomdRestartChebyTol;
    bool        useDensityMatrixPerturbationRankUpdates;
    double      xlbomdKernelRankUpdateFDParameter;
    bool        smearedNuclearCharges;
    bool        floatingNuclearCharges;
    bool        multipoleBoundaryConditions;
    bool        nonLinearCoreCorrection;
    dftfe::uInt maxLineSearchIterCGPRP;
    std::string atomicMassesFile;
    bool        useDeviceDirectAllReduce;
    bool        useDCCL;
    double      pspCutoffImageCharges;
    bool        reuseLanczosUpperBoundFromFirstCall;
    bool        allowMultipleFilteringPassesAfterFirstScf;
    dftfe::uInt highestStateOfInterestForChebFiltering;
    bool        useELPADeviceKernel;
    bool        memOptMode;
    bool        approxOverlapMatrix;
    bool        useReformulatedChFSI;

    dftfe::uInt dc_dispersioncorrectiontype;
    dftfe::uInt dc_d3dampingtype;
    bool        dc_d3ATM;
    bool        dc_d4MBD;
    std::string dc_dampingParameterFilename;
    double      dc_d3cutoff2;
    double      dc_d3cutoff3;
    double      dc_d3cutoffCN;


    std::string bfgsStepMethod;
    bool        usePreconditioner;
    dftfe::uInt lbfgsNumPastSteps;
    dftfe::uInt maxOptIter;
    dftfe::uInt maxStaggeredCycles;
    double      maxIonUpdateStep, maxCellUpdateStep;

    // New Paramters for moleculardyynamics class
    double      startingTempBOMD;
    double      MaxWallTime;
    double      thermostatTimeConstantBOMD;
    std::string tempControllerTypeBOMD;
    dftfe::Int  MDTrack;

    bool writeStructreEnergyForcesFileForPostProcess;

    // Parameters for confining potential
    bool   confiningPotential;
    double confiningInnerPotRad;
    double confiningOuterPotRad;
    double confiningWParam;
    double confiningCParam;

    dftParameters();

    /**
     * Parse parameters.
     */
    void
    parse_parameters(const std::string &parameter_file,
                     const MPI_Comm    &mpi_comm_parent,
                     const bool         printParams      = false,
                     const std::string  mode             = "GS",
                     const std::string  restartFilesPath = ".",
                     const dftfe::Int   _verbosity       = 1,
                     const bool         _useDevice       = false);

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
