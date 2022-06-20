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

    double radiusAtomBall, mixingParameter;
    double absLinearSolverTolerance, selfConsistentSolverTolerance, TVal,
      start_magnetization, absLinearSolverToleranceHelmholtz;

    bool isPseudopotential, periodicX, periodicY, periodicZ, useSymm,
      timeReversal, pseudoTestsFlag, constraintMagnetization, writeDosFile,
      writeLdosFile, writeLocalizationLengths, pinnedNodeForPBC, writePdosFile;


    /** parameters for LRJI preconditioner **/

    double      startingNormLRJILargeDamping;
    double      mixingParameterLRJI;
    std::string methodSubTypeLRJI;
    double      adaptiveRankRelTolLRJI;
    double      factorAdapAccumClearLRJI;
    double      absPoissonSolverToleranceLRJI;
    bool        singlePrecLRJI;
    bool        estimateJacCondNoFinalSCFIter;

    /**********************************************/

    std::string coordinatesFile, domainBoundingVectorsFile, kPointDataFile,
      ionRelaxFlagsFile, orthogType, algoType, pseudoPotentialFile;

    std::string coordinatesGaussianDispFile;

    double      outerAtomBallRadius, innerAtomBallRadius, meshSizeOuterDomain;
    bool        autoAdaptBaseMeshSize;
    double      meshSizeInnerBall, meshSizeOuterBall;
    double      chebyshevTolerance, topfrac, kerkerParameter;
    std::string mixingMethod, ionOptSolver;


    bool         isIonOpt, isCellOpt, isIonForce, isCellStress, isBOMD;
    bool         nonSelfConsistentForce, meshAdaption;
    double       forceRelaxTol, stressRelaxTol, toleranceKinetic;
    unsigned int cellConstraintType;

    int          verbosity;
    bool         keepScratchFolder;
    unsigned int chkType;
    bool         restartSpinFromNoSpin;
    bool         restartFromChk;
    bool         electrostaticsHRefinement;

    bool reproducible_output;

    bool writeWfcSolutionFields;

    bool writeDensitySolutionFields;

    std::string  startingWFCType;
    unsigned int numCoreWfcRR;
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
    bool         useMixedPrecCGS_SR;
    bool         useMixedPrecCGS_O;
    bool         useMixedPrecXTHXSpectrumSplit;
    bool         useMixedPrecSubspaceRotRR;
    unsigned int spectrumSplitStartingScfIter;
    bool         useELPA;
    bool         constraintsParallelCheck;
    bool         createConstraintsFromSerialDofhandler;
    bool         bandParalOpt;
    bool         useGPU;
    bool         gpuFineGrainedTimings;
    bool         allowFullCPUMemSubspaceRot;
    bool         useMixedPrecCheby;
    bool         overlapComputeCommunCheby;
    bool         overlapComputeCommunOrthoRR;
    bool         autoGPUBlockSizes;
    bool         readWfcForPdosPspFile;
    double       maxJacobianRatioFactorForMD;
    double       chebyshevFilterPolyDegreeFirstScfScalingFactor;
    int          reuseDensityMD;
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
    bool         useGPUDirectAllReduce;
    double       pspCutoffImageCharges;
    bool         reuseLanczosUpperBoundFromFirstCall;
    bool         allowMultipleFilteringPassesAfterFirstScf;
    bool         useELPAGPUKernel;
    std::string  xcFamilyType;
    bool         gpuMemOptMode;

    unsigned int dc_dispersioncorrectiontype;
    unsigned int dc_d3dampingtype;
    bool         dc_d3ATM;
    bool         dc_d4MBD;
    std::string  dc_dampingParameterFilename;
    double       dc_d3cutoff2;
    double       dc_d3cutoff3;
    double       dc_d3cutoffCN;


    // New Paramters for moleculardyynamics class
    double      startingTempBOMD;
    double      MaxWallTime;
    double      thermostatTimeConstantBOMD;
    std::string tempControllerTypeBOMD;
    int         MDTrack;

    dftParameters();

    /**
     * Parse parameters.
     */
    void
    parse_parameters(const std::string &parameter_file,
                     const MPI_Comm &   mpi_comm_parent,
                     const bool         printParams = false);

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

    /**
     * set family type exchange correlation functional
     *
     */
    void
    setXCFamilyType();
  }; // class dftParameters

} // namespace dftfe
#endif
