#!/usr/bin/env python3
"""
Step 2: Update #include directives to use <dftfe/foo.h> form.
Processes all files in include/dftfe/, src/, utils/.
"""

import os
import re

# Complete set of dftfe-owned basenames (no path, just filename)
DFTFE_HEADERS = {
    "atomCenteredPostProcessing.h",
    "AtomCenteredPseudoWavefunctionSpline.h",
    "AtomCenteredSphericalFunctionBase.h",
    "AtomCenteredSphericalFunctionBessel.h",
    "AtomCenteredSphericalFunctionContainer.h",
    "AtomCenteredSphericalFunctionCoreDensitySpline.h",
    "AtomCenteredSphericalFunctionGaussian.h",
    "AtomCenteredSphericalFunctionLocalPotentialSpline.h",
    "AtomCenteredSphericalFunctionProjectorSpline.h",
    "AtomCenteredSphericalFunctionSinc.h",
    "AtomCenteredSphericalFunctionSpline.h",
    "AtomCenteredSphericalFunctionValenceDensitySpline.h",
    "AtomCenteredSphericalFunctionZOverR.h",
    "AtomicBasisData.h",
    "AtomicBasis.h",
    "AtomicCenteredNonLocalOperator.h",
    "AtomicCenteredNonLocalOperatorKernelsDevice.h",
    "AtomPseudoWavefunctions.h",
    "AuxDensityMatrixFE.h",
    "AuxDensityMatrix.h",
    "BFGSNonLinearSolver.h",
    "BLASWrapper.h",
    "Cell.h",
    "cgPRPNonLinearSolver.h",
    "chebyshevOrthogonalizedSubspaceIterationSolverDevice.h",
    "chebyshevOrthogonalizedSubspaceIterationSolver.h",
    "CompositeData.h",
    "computeAuxProjectedDensityMatrixFromPSI.h",
    "configurationalForce.h",
    "configurationalForceKernels.h",
    "constants.h",
    "constraintMatrixInfoDeviceKernels.h",
    "constraintMatrixInfo.h",
    "DataTypeOverloads.h",
    "dealiiLinearSolver.h",
    "dealiiLinearSolverProblem.h",
    "densityCalculatorDeviceKernels.h",
    "densityCalculator.h",
    "densityFirstOrderResponseCalculator.h",
    "DeviceAPICalls.h",
    "DeviceDataTypeOverloads.cu.h",
    "DeviceDataTypeOverloads.h",
    "DeviceDataTypeOverloads.hip.h",
    "DeviceDataTypeOverloads.sycl.h",
    "deviceDirectCCLWrapper.h",
    "DeviceExceptions.cu.h",
    "DeviceExceptions.h",
    "DeviceExceptions.hip.h",
    "DeviceExceptions.sycl.h",
    "DeviceKernelLauncherHelpers.h",
    "deviceKernelsGeneric.h",
    "DeviceTypeConfig.cu.h",
    "DeviceTypeConfig.h",
    "DeviceTypeConfigHalfPrec.cu.h",
    "DeviceTypeConfigHalfPrec.h",
    "DeviceTypeConfigHalfPrec.hip.h",
    "DeviceTypeConfigHalfPrec.sycl.h",
    "DeviceTypeConfig.hip.h",
    "DeviceTypeConfig.sycl.h",
    "dftBase.h",
    "dftd.h",
    "dftfeDataTypes.h",
    "dftfeWrapper.h",
    "dft.h",
    "dftParameters.h",
    "dftUtils.h",
    "eigenSolver.h",
    "elpaScalaManager.h",
    "energyCalculator.h",
    "eshelbyTensor.h",
    "eshelbyTensorSpinPolarized.h",
    "excDensityGGAClass.h",
    "excDensityLDAClass.h",
    "excDensityLLMGGAClass.h",
    "excDensityPositivityCheckTypes.h",
    "ExcDFTPlusU.h",
    "Exceptions.h",
    "Exceptions.t.cc",
    "exchangeCorrelationFunctionalEvaluation.def",
    "exchangeCorrelationFunctionalEvaluator.h",
    "excManager.h",
    "excManagerKernels.h",
    "ExcSSDFunctionalBaseClass.h",
    "ExcSSDFunctionalBaseClass.t.cc",
    "excTauMGGAClass.h",
    "expConfiningPotential.h",
    "FEBasisOperations.h",
    "FEBasisOperationsKernelsInternal.h",
    "FECell.h",
    "feevaluationWrapper3Comp.def",
    "feevaluationWrapper.def",
    "feevaluationWrapper.h",
    "fileReaders.h",
    "FiniteDifference.h",
    "forceWfcContractionsDeviceKernels.h",
    "forceWfcContractions.h",
    "functionalTest.h",
    "GaussianBasis.h",
    "geometryOptimizationClass.h",
    "geoOptCell.h",
    "geoOptIon.h",
    "git_info.h",
    "git_info.h.in",
    "groupSymmetry.h",
    "headers.h",
    "hubbardClass.h",
    "InterpolateCellWiseDataToPoints.h",
    "InterpolateFromCellToLocalPoints.h",
    "kerkerSolverProblemDevice.h",
    "kerkerSolverProblem.h",
    "kerkerSolverProblemWrapper.def",
    "kerkerSolverProblemWrapper.h",
    "KohnShamDFTBaseOperator.h",
    "KohnShamDFTOperatorKernels.h",
    "KohnShamDFTStandardEigenOperator.h",
    "lapack_support.h",
    "LBFGSNonLinearSolver.h",
    "libraryMDI.h",
    "linearAlgebraOperationsCPU.h",
    "linearAlgebraOperationsDevice.h",
    "linearAlgebraOperationsDeviceKernels.h",
    "linearAlgebraOperations.h",
    "linearAlgebraOperationsInternal.h",
    "linearSolverCGDevice.h",
    "linearSolverCGDeviceKernels.h",
    "linearSolverDevice.h",
    "linearSolver.h",
    "linearSolverProblemDevice.h",
    "MapPointsToCells.h",
    "MatrixFreeDevice.h",
    "MatrixFree.h",
    "MatrixFreeWrapper.def",
    "MatrixFreeWrapper.h",
    "MDIEngine.h",
    "MemoryManager.h",
    "MemorySpaceType.h",
    "MemoryStorage.h",
    "MemoryStorage.t.cc",
    "MemoryTransfer.h",
    "MemoryTransferKernelsDevice.h",
    "MemoryTransfer.t.cc",
    "meshGenUtils.h",
    "meshMovementAffineTransform.h",
    "meshMovementGaussian.h",
    "meshMovement.h",
    "mixingClass.h",
    "molecularDynamicsClass.h",
    "MPICommunicatorP2P.h",
    "MPICommunicatorP2PKernels.h",
    "MPIPatternP2P.h",
    "MPIPatternP2P.t.cc",
    "MPIRequestersBase.h",
    "MPIRequestersNBX.h",
    "MPITags.h",
    "MPIWriteOnFile.h",
    "MultiVectorCGSolver.h",
    "MultiVector.h",
    "MultiVectorLinearSolverProblem.h",
    "MultiVectorMinResSolver.h",
    "MultiVectorPoissonLinearSolverProblem.h",
    "MultiVector.t.cc",
    "NNGGA.h",
    "NNLDA.h",
    "NNLLMGGA.h",
    "NodalData.h",
    "nonlinearSolverFunction.h",
    "nonLinearSolver.h",
    "nonlinearSolverProblem.h",
    "nudgedElasticBandClass.h",
    "oncvClass.h",
    "operator.h",
    "OptimizedIndexSet.h",
    "OptimizedIndexSet.t.cc",
    "PeriodicTable.h",
    "poissonSolverProblemDevice.h",
    "poissonSolverProblem.h",
    "poissonSolverProblemWrapper.def",
    "poissonSolverProblemWrapper.h",
    "process_grid.h",
    "pseudoConverter.h",
    "pseudopotentialBaseClass.h",
    "pseudoUtils.h",
    "QuadDataCompositeWrite.h",
    "RTreeBox.h",
    "RTreePoint.h",
    "runParameters.h",
    "scalapack.templates.h",
    "scalapackWrapper.h",
    "SlaterBasis.h",
    "solveVselfInBinsDevice.h",
    "solveVselfInBinsDeviceKernels.h",
    "SphericalFunctionUtil.h",
    "sphericalHarmonicUtils.h",
    "StringOperations.h",
    "TransferBetweenMeshesIncompatiblePartitioning.h",
    "triangulationManager.h",
    "TypeConfig.h",
    "vectorUtilities.h",
    "vselfBinsManager.h",
    "config.h",  # new generated file
}

# Directories to process
BASE = "/home/bikash/Documents/dftfe/.claude/worktrees/agent-ac6cae319ca552232"
DIRS = [
    os.path.join(BASE, "include/dftfe"),
    os.path.join(BASE, "src"),
    os.path.join(BASE, "utils"),
]

# File extensions to process
EXTS = {".h", ".cc", ".cpp", ".t.cc", ".def", ".cu.h", ".hip.h", ".sycl.h"}

def has_matching_ext(fname):
    """Check if filename matches one of our target extensions."""
    for ext in EXTS:
        if fname.endswith(ext):
            return True
    return False

def transform_line(line):
    """Apply Rules A and B to a single line."""
    # Rule A: angle brackets - bare name (no path separators)
    m = re.match(r'^(\s*#\s*include\s*)<([^/>"]+)>(.*\n?)$', line)
    if m:
        pre, name, post = m.groups()
        if name in DFTFE_HEADERS:
            return f'{pre}<dftfe/{name}>{post}'

    # Rule A: double quotes - bare name (no path separators)
    m = re.match(r'^(\s*#\s*include\s*)"([^/>"]+)"(.*\n?)$', line)
    if m:
        pre, name, post = m.groups()
        if name in DFTFE_HEADERS:
            return f'{pre}<dftfe/{name}>{post}'

    # Rule B: XCfunctionalDefs with angle brackets
    m = re.match(r'^(\s*#\s*include\s*)<(XCfunctionalDefs/[^>]+)>(.*\n?)$', line)
    if m:
        pre, name, post = m.groups()
        return f'{pre}<dftfe/{name}>{post}'

    # Rule B: XCfunctionalDefs with double quotes
    m = re.match(r'^(\s*#\s*include\s*)"(XCfunctionalDefs/[^"]+)"(.*\n?)$', line)
    if m:
        pre, name, post = m.groups()
        return f'{pre}<dftfe/{name}>{post}'

    return line

def process_file(filepath):
    """Transform includes in a file. Returns True if file was changed."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False

    new_lines = [transform_line(line) for line in lines]

    if new_lines != lines:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

changed = []
skipped = []

for d in DIRS:
    for root, dirs, files in os.walk(d):
        # Sort for deterministic ordering
        dirs.sort()
        files.sort()
        for fname in files:
            if not has_matching_ext(fname):
                skipped.append(os.path.join(root, fname))
                continue
            fpath = os.path.join(root, fname)
            if process_file(fpath):
                changed.append(fpath)

print(f"Changed: {len(changed)} files")
print(f"Unchanged/skipped: {len(skipped)} files")
if changed:
    print("\nChanged files:")
    for f in changed:
        rel = f.replace(BASE + "/", "")
        print(f"  {rel}")
