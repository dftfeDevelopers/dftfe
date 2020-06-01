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
// @author Phani Motamarri, Sambit Das
//
#include <dftParameters.h>
#include <iostream>
#include <fstream>
#include <deal.II/base/data_out_base.h>

using namespace dealii;

namespace dftfe {

	namespace dftParameters
	{

		unsigned int finiteElementPolynomialOrder=1,n_refinement_steps=1,numberEigenValues=1,xc_id=1, spinPolarized=0, nkx=1,nky=1,nkz=1, offsetFlagX=0,offsetFlagY=0,offsetFlagZ=0;
		unsigned int chebyshevOrder=1,numPass=1, numSCFIterations=1,maxLinearSolverIterations=1, mixingHistory=1, npool=1,maxLinearSolverIterationsHelmholtz=1;

		double radiusAtomBall=0.0, mixingParameter=0.5;
		double lowerEndWantedSpectrum=0.0,absLinearSolverTolerance=1e-10,selfConsistentSolverTolerance=1e-10,TVal=500, start_magnetization=0.0,absLinearSolverToleranceHelmholtz=1e-10;
		double chebyshevTolerance = 1e-02;
		double chebyshevFilterTolXLBOMDRankUpdates = 1e-07;
		std::string mixingMethod = "";
		std::string ionOptSolver = "";

		bool isPseudopotential=false,periodicX=false,periodicY=false,periodicZ=false, useSymm=false, timeReversal=false,pseudoTestsFlag=false, constraintMagnetization=false, writeDosFile=false, writeLdosFile=false, writePdosFile=false, writeLocalizationLengths=false;
		std::string meshFileName="",coordinatesFile="",domainBoundingVectorsFile="",kPointDataFile="", ionRelaxFlagsFile="",orthogType="", algoType="", pseudoPotentialFile="";

		std::string coordinatesGaussianDispFile="";

		double outerAtomBallRadius=2.0, innerAtomBallRadius=0.0, meshSizeOuterDomain=10.0;
		double meshSizeInnerBall=1.0, meshSizeOuterBall=1.0;
		unsigned int numLevels = 1,numberWaveFunctionsForEstimate = 5;
		double topfrac = 0.1;
		double kerkerParameter = 0.05;

		bool isIonOpt=false, isCellOpt=false, isIonForce=false, isCellStress=false, isBOMD=false, isXLBOMD=false;
		bool nonSelfConsistentForce=false;
		double forceRelaxTol  = 1e-4;//Hartree/Bohr
		double stressRelaxTol = 1e-6;//Hartree/Bohr^3
		double toleranceKinetic = 1e-03;
		unsigned int cellConstraintType=12;// all cell components to be relaxed

		unsigned int verbosity=0; unsigned int chkType=0;
		bool restartFromChk=false;
		bool restartMdFromChk=false;
		bool reproducible_output=false;
		bool electrostaticsHRefinement = false;
		bool electrostaticsPRefinement = false;
		bool meshAdaption = false;
		bool pinnedNodeForPBC = true;

		std::string startingWFCType="";
		bool useBatchGEMM=false;
		bool writeWfcSolutionFields=false;
		bool writeDensitySolutionFields=false;
		unsigned int wfcBlockSize=400;
		unsigned int chebyWfcBlockSize=400;
		unsigned int subspaceRotDofsBlockSize=2000;
		bool enableSwitchToGS=true;
		unsigned int nbandGrps=1;
		bool computeEnergyEverySCF=true;
		unsigned int scalapackParalProcs=0;
		unsigned int scalapackBlockSize=50;
		unsigned int natoms=0;
		unsigned int natomTypes=0;
		double lowerBoundUnwantedFracUpper=0;
		unsigned int numCoreWfcRR=0;
		bool triMatPGSOpt=true;
		bool reuseWfcGeoOpt=false;
		bool reuseDensityGeoOpt=false;
		double mpiAllReduceMessageBlockSizeMB=2.0;
		bool useHigherQuadNLP=true;
		bool useMixedPrecPGS_SR=false;
		bool useMixedPrecPGS_O=false;
		bool useAsyncChebPGS_SR = false;
		bool useMixedPrecXTHXSpectrumSplit=false;
		bool useMixedPrecSubspaceRotSpectrumSplit=false;
		bool useMixedPrecSubspaceRotRR=false;
		bool useSinglePrecXtHXOffDiag=false;
		unsigned int numAdaptiveFilterStates=0;
		unsigned int spectrumSplitStartingScfIter=1;
		bool useELPA=false;
		bool constraintsParallelCheck=true;
		bool createConstraintsFromSerialDofhandler=true;
		bool bandParalOpt=true;
		bool rrGEP=false;
		bool rrGEPFullMassMatrix=false;
		bool autoUserMeshParams=false;
		bool readWfcForPdosPspFile=false;
		bool useGPU=false;
		bool gpuFineGrainedTimings=false;
		bool allowFullCPUMemSubspaceRot=true;
		bool useMixedPrecCheby=false;
		bool useMixedPrecChebyNonLocal=false;
		unsigned int mixedPrecXtHXFracStates=0;
		bool overlapComputeCommunCheby=false;
		bool overlapComputeCommunOrthoRR=false;
		bool autoGPUBlockSizes=true;
		double maxJacobianRatioFactorForMD=1.5;
		double chebyshevFilterTolXLBOMD=1e-8;
		double timeStepBOMD=0.5;
		unsigned int numberStepsBOMD=1000;
		double startingTempBOMDNVE=300.0;
		double gaussianConstantForce=0.75;
		double gaussianOrderForce=4.0;
		double gaussianOrderMoveMeshToAtoms=4.0;
		bool useFlatTopGenerator=false;
		double diracDeltaKernelScalingConstant=0.1;
		unsigned int kernelUpdateRankXLBOMD=0;
		unsigned int kmaxXLBOMD=8;
		bool autoMeshStepInterpolateBOMD=false;
		double ratioOfMeshMovementToForceGaussianBOMD=1.0;
		bool useAtomicRhoXLBOMD=true;
		bool useMeshSizesFromAtomsFile=false;
		bool chebyCommunAvoidanceAlgo=false;
		double chebyshevFilterPolyDegreeFirstScfScalingFactor=1.34;
		unsigned int numberPassesRRSkippedXLBOMD=0;
		bool useSingleFullScfXLBOMD=false;
		bool skipHarmonicOscillatorTermInitialStepsXLBOMD=false;
		double xlbomdRestartChebyTol=1e-9;
		bool xlbomdRRPassMixedPrec=false;
		bool useDensityMatrixPerturbationRankUpdates=false;
		double xlbomdKernelRankUpdateFDParameter=1e-2;
		bool xlbomdStepTimingRun=false;
		bool smearedNuclearCharges=false;

		void declare_parameters(ParameterHandler &prm)
		{
			prm.declare_entry("REPRODUCIBLE OUTPUT", "false",
					Patterns::Bool(),
					"[Developer] Limit output to what is reproducible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.");


			prm.declare_entry("H REFINED ELECTROSTATICS", "false",
					Patterns::Bool(),
					"[Advanced] Compute electrostatic energy and forces on a h refined mesh after each ground-state solve. Default: false.");

			prm.declare_entry("P REFINED ELECTROSTATICS", "false",
					Patterns::Bool(),
					"[Advanced] Compute electrostatic energy on a p refined mesh after each ground-state solve. Default: false.");

			prm.declare_entry("VERBOSITY", "1",
					Patterns::Integer(0,5),
					"[Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users.");

			prm.enter_subsection ("GPU");
			{
				prm.declare_entry("USE GPU", "false",
						Patterns::Bool(),
						"[Standard] Use GPU for compute.");

				prm.declare_entry("AUTO GPU BLOCK SIZES", "true",
						Patterns::Bool(),
						"[Advanced] Automatically sets total number of kohn-sham wave functions and eigensolver optimal block sizes for running on GPUs. If manual tuning is desired set this parameter to false and set the block sizes using the input parameters for the block sizes. Default: true.");

				prm.declare_entry("FINE GRAINED GPU TIMINGS", "false",
						Patterns::Bool(),
						"[Developer] Print more fine grained GPU timing results. Default: false.");


				prm.declare_entry("SUBSPACE ROT FULL CPU MEM", "true",
						Patterns::Bool(),
						"[Developer] Option to use full NxN memory on CPU in subspace rotation and when mixed precision optimization is not being used. This reduces the number of MPI_Allreduce communication calls. Default: true.");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Postprocessing");
			{
				prm.declare_entry("WRITE WFC", "false",
						Patterns::Bool(),
						"[Standard] Writes DFT ground state wavefunction solution fields (FEM mesh nodal values) to wfcOutput.vtu file for visualization purposes. The wavefunction solution fields in wfcOutput.vtu are named wfc_s_k_i in case of spin-polarized calculations and wfc_k_i otherwise, where s denotes the spin index (0 or 1), k denotes the k point index starting from 0, and i denotes the Kohn-Sham wavefunction index starting from 0. In the case of geometry optimization, the wavefunctions corresponding to the last ground-state solve are written.  Default: false.");

				prm.declare_entry("WRITE DENSITY", "false",
						Patterns::Bool(),
						"[Standard] Writes DFT ground state electron-density solution fields (FEM mesh nodal values) to densityOutput.vtu file for visualization purposes. The electron-density solution field in densityOutput.vtu is named density. In case of spin-polarized calculation, two additional solution fields- density_0 and density_1 are also written where 0 and 1 denote the spin indices. In the case of geometry optimization, the electron-density corresponding to the last ground-state solve is written. Default: false.");

				prm.declare_entry("WRITE DENSITY OF STATES", "false",
						Patterns::Bool(),
						"[Standard] Computes density of states using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'dosData.out' containing two columns with first column indicating the energy in eV and second column indicating the density of states");

				prm.declare_entry("WRITE LOCAL DENSITY OF STATES", "false",
						Patterns::Bool(),
						"[Standard] Computes local density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'ldosData.out' containing NUMATOM+1 columns with first column indicating the energy in eV and all other NUMATOM columns indicating local density of states for each of the NUMATOM atoms.");

				prm.declare_entry("WRITE PROJECTED DENSITY OF STATES", "false",
						Patterns::Bool(),
						"[Standard] Computes projected density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'pdosData_x' with x denoting atomID. This file contains columns with first column indicating the energy in eV and all other columns indicating projected density of states corresponding to single atom wavefunctions.");

				prm.declare_entry("READ ATOMIC WFC PDOS FROM PSP FILE", "false",
						Patterns::Bool(),
						"[Standard] Read atomic wavefunctons from the pseudopotential file for computing projected density of states. When set to false atomic wavefunctions from the internal database are read, which correspond to sg15 ONCV pseudopotentials.");

				prm.declare_entry("WRITE LOCALIZATION LENGTHS", "false",
						Patterns::Bool(),
						"[Standard] Computes localization lengths of all wavefunctions which is defined as the deviation around the mean position of a given wavefunction. Outputs a file name 'localizationLengths.out' containing 2 columns with first column indicating the wavefunction index and second column indicating localization length of the corresponding wavefunction.");

			}
			prm.leave_subsection ();

			prm.enter_subsection ("Parallelization");
			{
				prm.declare_entry("NPKPT", "1",
						Patterns::Integer(1),
						"[Standard] Number of groups of MPI tasks across which the work load of the irreducible k-points is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPKPT must be less than or equal to the number of irreducible k-points.");

				prm.declare_entry("NPBAND", "1",
						Patterns::Integer(1),
						"[Standard] Number of groups of MPI tasks across which the work load of the bands is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPBAND must be less than or equal to NUMBER OF KOHN-SHAM WAVEFUNCTIONS.");

				prm.declare_entry("MPI ALLREDUCE BLOCK SIZE", "100.0",
						Patterns::Double(0),
						"[Advanced] Block message size in MB used to break a single MPI_Allreduce call on wavefunction vectors data into multiple MPI_Allreduce calls. This is useful on certain architectures which take advantage of High Bandwidth Memory to improve efficiency of MPI operations. This variable is relevant only if NPBAND>1. Default value is 100.0 MB.");

				prm.declare_entry("BAND PARAL OPT", "true",
						Patterns::Bool(),
						"[Standard] Uses a more optimal route for band parallelization but at the cost of extra wavefunctions memory.");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Checkpointing and Restart");
			{
				prm.declare_entry("CHK TYPE", "0",
						Patterns::Integer(0,3),
						"[Standard] Checkpoint type, 0 (do not create any checkpoint), 1 (create checkpoint for geometry optimization restart if either ION OPT or CELL OPT or BOMD is set to true. Currently, checkpointing and restart framework does not work if both ION OPT and CELL OPT are set to true simultaneously- the code will throw an error if attempted.), 2 (create checkpoint for scf restart. Currently, this option cannot be used if geometry optimization is being performed. The code will throw an error if this option is used in conjunction with geometry optimization.)");

				prm.declare_entry("RESTART FROM CHK", "false",
						Patterns::Bool(),
						"[Standard] Boolean parameter specifying if the current job reads from a checkpoint. The nature of the restart corresponds to the CHK TYPE parameter. Hence, the checkpoint being read must have been created using the CHK TYPE parameter before using this option. RESTART FROM CHK is always false for CHK TYPE 0.");

				prm.declare_entry("RESTART MD FROM CHK", "false",
						Patterns::Bool(),
						"[Standard] Boolean parameter specifying if the current job reads from a checkpoint. The nature of the restart corresponds to the CHK TYPE parameter. Hence, the checkpoint being read must have been created using the CHK TYPE parameter before using this option. RESTART FROM CHK is always false for CHK TYPE 0.");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Geometry");
			{
				prm.declare_entry("ATOMIC COORDINATES FILE", "",
						Patterns::Anything(),
						"[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");

				prm.declare_entry("ATOMIC DISP COORDINATES FILE", "",
						Patterns::Anything(),
						"[Standard] Atomic displacement coordinates input file name. The FEM mesh is deformed using Gaussian functions attached to the atoms. File format (example for two atoms): delx1 dely1 delz1 (row1), delx2 dely2 delz2 (row2). The number of rows must be equal to NATOMS. Units in a.u.");


				prm.declare_entry("NATOMS", "0",
						Patterns::Integer(0),
						"[Standard] Total number of atoms. This parameter requires a mandatory non-zero input which is equal to the number of rows in the file passed to ATOMIC COORDINATES FILE.");

				prm.declare_entry("NATOM TYPES", "0",
						Patterns::Integer(0),
						"[Standard] Total number of atom types. This parameter requires a mandatory non-zero input which is equal to the number of unique atom types in the file passed to ATOMIC COORDINATES FILE.");

				prm.declare_entry("DOMAIN VECTORS FILE", "",
						Patterns::Anything(),
						"[Standard] Domain vectors input file name. Domain vectors are the vectors bounding the three edges of the 3D parallelepiped computational domain. File format: v1x v1y v1z (row1), v2x v2y v2z (row2), v3x v3y v3z (row3). Units: a.u. CAUTION: please ensure that the domain vectors form a right-handed coordinate system i.e. dotProduct(crossProduct(v1,v2),v3)>0. Domain vectors are the typical lattice vectors in a fully periodic calculation.");

				prm.enter_subsection ("Optimization");
				{

					prm.declare_entry("ION FORCE", "false",
							Patterns::Bool(),
							"[Standard] Boolean parameter specifying if atomic forces are to be computed. Automatically set to true if ION OPT is true.");

					prm.declare_entry("NON SELF CONSISTENT FORCE", "false",
							Patterns::Bool(),
							"[Developer] Boolean parameter specifying whether to include the force contributions arising out of non self-consistency in the Kohn-Sham ground-state calculation. Currently non self-consistent force computation is still in experimental phase. The default option is false.");

					prm.declare_entry("ION OPT", "false",
							Patterns::Bool(),
							"[Standard] Boolean parameter specifying if atomic forces are to be relaxed.");

					prm.declare_entry("ION OPT SOLVER", "CGPRP",
							Patterns::Selection("CGDESCENT|LBFGS|CGPRP"),
							"[Standard] Method for Ion relaxation solver. CGPRP (Nonlinear conjugate gradient with Secant and Polak-Ribiere approach) is the default");


					prm.declare_entry("FORCE TOL", "1e-4",
							Patterns::Double(0,1.0),
							"[Standard] Sets the tolerance on the maximum force (in a.u.) on an atom during atomic relaxation, when the atoms are considered to be relaxed.");

					prm.declare_entry("ION RELAX FLAGS FILE", "",
							Patterns::Anything(),
							"[Standard] File specifying the permission flags (1-free to move, 0-fixed) and external forces for the 3-coordinate directions and for all atoms. File format (example for two atoms with atom 1 fixed and atom 2 free and 0.01 Ha/Bohr force acting on atom 2): 0 0 0 0.0 0.0 0.0(row1), 1 1 1 0.0 0.0 0.01(row2). External forces are optional.");

					prm.declare_entry("CELL STRESS", "false",
							Patterns::Bool(),
							"[Standard] Boolean parameter specifying if cell stress needs to be computed. Automatically set to true if CELL OPT is true.");

					prm.declare_entry("CELL OPT", "false",
							Patterns::Bool(),
							"[Standard] Boolean parameter specifying if cell needs to be relaxed to achieve zero stress");

					prm.declare_entry("STRESS TOL", "1e-6",
							Patterns::Double(0,1.0),
							"[Standard] Sets the tolerance of the cell stress (in a.u.) during cell-relaxation.");

					prm.declare_entry("CELL CONSTRAINT TYPE", "12",
							Patterns::Integer(1,13),
							"[Standard] Cell relaxation constraint type, 1 (isotropic shape-fixed volume optimization), 2 (volume-fixed shape optimization), 3 (relax along domain vector component v1x), 4 (relax along domain vector component v2x), 5 (relax along domain vector component v3x), 6 (relax along domain vector components v2x and v3x), 7 (relax along domain vector components v1x and v3x), 8 (relax along domain vector components v1x and v2x), 9 (volume optimization- relax along domain vector components v1x, v2x and v3x), 10 (2D - relax along x and y components), 11(2D- relax only x and y components with inplane area fixed), 12(relax all domain vector components), 13 automatically decides the constraints based on boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.");

					prm.declare_entry("REUSE WFC", "false",
							Patterns::Bool(),
							"[Standard] Reuse previous ground-state wavefunctions during geometry optimization. Default setting is false.");

					prm.declare_entry("REUSE DENSITY", "false",
							Patterns::Bool(),
							"[Standard] Reuse previous ground-state density during geometry optimization. Default setting is false.");

				}
				prm.leave_subsection ();

			}
			prm.leave_subsection ();

			prm.enter_subsection ("Boundary conditions");
			{
				prm.declare_entry("SELF POTENTIAL RADIUS", "0.0",
						Patterns::Double(0.0,50),
						"[Advanced] The radius (in a.u) of the ball around an atom in which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accommodate the largest radius possible for the given finite element mesh. The default approach works for most problems.");

				prm.declare_entry("PERIODIC1", "false",
						Patterns::Bool(),
						"[Standard] Periodicity along the first domain bounding vector.");

				prm.declare_entry("PERIODIC2", "false",
						Patterns::Bool(),
						"[Standard] Periodicity along the second domain bounding vector.");

				prm.declare_entry("PERIODIC3", "false",
						Patterns::Bool(),
						"[Standard] Periodicity along the third domain bounding vector.");

				prm.declare_entry("POINT WISE DIRICHLET CONSTRAINT", "false",
						Patterns::Bool(),
						"[Developer] Flag to set point wise dirichlet constraints to eliminate null-space associated with the discretized Poisson operator subject to periodic BCs.");

				prm.declare_entry("CONSTRAINTS PARALLEL CHECK", "true",
						Patterns::Bool(),
						"[Developer] Check for consistency of constraints in parallel.");

				prm.declare_entry("CONSTRAINTS FROM SERIAL DOFHANDLER", "false",
						Patterns::Bool(),
						"[Developer] Check constraints from serial dofHandler.");

				prm.declare_entry("SMEARED NUCLEAR CHARGES", "false",
						Patterns::Bool(),
						"[Developer] Nuclear charges are smeared for solving electrostatic fields.");        

			}
			prm.leave_subsection ();


			prm.enter_subsection ("Finite element mesh parameters");
			{

				prm.declare_entry("POLYNOMIAL ORDER", "4",
						Patterns::Integer(1,12),
						"[Standard] The degree of the finite-element interpolating polynomial. Default value is 4. POLYNOMIAL ORDER= 4 or 5 is usually a good choice for most pseudopotential as well as all-electron problems.");

				prm.declare_entry("MESH FILE", "",
						Patterns::Anything(),
						"[Developer] External mesh file path. If nothing is given auto mesh generation is performed. The option is only for testing purposes.");

				prm.enter_subsection ("Auto mesh generation parameters");
				{

					prm.declare_entry("BASE MESH SIZE", "0.0",
							Patterns::Double(0,20),
							"[Advanced] Mesh size of the base mesh on which refinement is performed. For the default value of 0.0, a heuristically determined base mesh size is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

					prm.declare_entry("ATOM BALL RADIUS","2.0",
							Patterns::Double(0,20),
							"[Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AROUND ATOM. The default value of 2.0 is good enough for most cases. On rare cases, where the nonlocal pseudopotential projectors have a compact support beyond 2.0, a slightly larger ATOM BALL RADIUS between 2.0 to 2.5 may be required. Standard users do not need to tune this parameter. Units: a.u.");

					prm.declare_entry("INNER ATOM BALL RADIUS","0.0",
							Patterns::Double(0,20),
							"[Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AT ATOM. Standard users do not need to tune this parameter. Units: a.u.");


					prm.declare_entry("MESH SIZE AROUND ATOM", "0.8",
							Patterns::Double(0.0001,10),
							"[Standard] Mesh size in a ball of radius ATOM BALL RADIUS around every atom. For pseudopotential calculations, a value between 0.5 to 1.0 is usually a good choice. For all-electron calculations, a value between 0.1 to 0.3 would be a good starting choice. In most cases, MESH SIZE AROUND ATOM is the only parameter to be tuned to achieve the desired accuracy in energy and forces with respect to the mesh refinement. Units: a.u.");

					prm.declare_entry("MESH SIZE AT ATOM", "0.0",
							Patterns::Double(0.0,10),
							"[Advanced] Mesh size of the finite elements in the immediate vicinity of the atom. For the default value of 0.0, a heuristically determined MESH SIZE AT ATOM is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

					prm.declare_entry("MESH ADAPTION","false",
							Patterns::Bool(),
							"[Standard] Generates adaptive mesh based on a-posteriori mesh adaption strategy using single atom wavefunctions before computing the ground-state. Default: false.");

					prm.declare_entry("AUTO USER MESH PARAMS","false",
							Patterns::Bool(),
							"[Standard] Except MESH SIZE AROUND ATOM, all other user defined mesh parameters are heuristically set. Default: false.");


					prm.declare_entry("TOP FRAC", "0.1",
							Patterns::Double(0.0,1),
							"[Developer] Top fraction of elements to be refined.");

					prm.declare_entry("NUM LEVELS", "10",
							Patterns::Integer(0,30),
							"[Developer] Number of times to be refined.");

					prm.declare_entry("TOLERANCE FOR MESH ADAPTION", "1",
							Patterns::Double(0.0,1),
							"[Developer] Tolerance criteria used for stopping the multi-level mesh adaption done apriori using single atom wavefunctions. This is used as Kinetic energy change between two successive iterations");

					prm.declare_entry("ERROR ESTIMATE WAVEFUNCTIONS", "5",
							Patterns::Integer(0),
							"[Developer] Number of wavefunctions to be used for error estimation.");

					prm.declare_entry("GAUSSIAN CONSTANT FORCE GENERATOR", "0.75",
							Patterns::Double(0.0),
							"[Developer] Force computation generator gaussian constant. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant)^(gaussianOrder)).");

					prm.declare_entry("GAUSSIAN ORDER FORCE GENERATOR", "4.0",
							Patterns::Double(0.0),
							"[Developer] Force computation generator gaussian order. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant)^(gaussianOrder)).");

					prm.declare_entry("GAUSSIAN ORDER MOVE MESH TO ATOMS", "4.0",
							Patterns::Double(0.0),
							"[Developer] Move mesh to atoms gaussian order. Gamma(r)= exp(-(r/gaussianConstant)^(gaussianOrder)).");

					prm.declare_entry("USE FLAT TOP GENERATOR", "false",
							Patterns::Bool(),
							"[Developer] Use a composite generator flat top and Gaussian generator for mesh movement and configurational force computation.");

					prm.declare_entry("USE MESH SIZES FROM ATOM LOCATIONS FILE", "false",
							Patterns::Bool(),
							"[Developer] Use mesh sizes from atom locations file.");

				}
				prm.leave_subsection ();
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Brillouin zone k point sampling options");
			{
				prm.enter_subsection ("Monkhorst-Pack (MP) grid generation");
				{
					prm.declare_entry("SAMPLING POINTS 1", "1",
							Patterns::Integer(1,1000),
							"[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 1.");

					prm.declare_entry("SAMPLING POINTS 2", "1",
							Patterns::Integer(1,1000),
							"[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 2.");

					prm.declare_entry("SAMPLING POINTS 3", "1",
							Patterns::Integer(1,1000),
							"[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 3.");

					prm.declare_entry("SAMPLING SHIFT 1", "0",
							Patterns::Integer(0,1),
							"[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 1.");

					prm.declare_entry("SAMPLING SHIFT 2", "0",
							Patterns::Integer(0,1),
							"[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 2.");

					prm.declare_entry("SAMPLING SHIFT 3", "0",
							Patterns::Integer(0,1),
							"[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 3.");

				}
				prm.leave_subsection ();

				prm.declare_entry("kPOINT RULE FILE", "",
						Patterns::Anything(),
						"[Developer] File providing list of k points on which eigen values are to be computed from converged KS Hamiltonian. The first three columns specify the crystal coordinates of the k points. The fourth column provides weights of the corresponding points, which is currently not used. The eigen values are written on an output file bands.out");

				prm.declare_entry("USE GROUP SYMMETRY", "false",
						Patterns::Bool(),
						"[Standard] Flag to control the use of point group symmetries. Currently this feature cannot be used if ION FORCE or CELL STRESS input parameters are set to true.");

				prm.declare_entry("USE TIME REVERSAL SYMMETRY", "false",
						Patterns::Bool(),
						"[Standard] Flag to control the use of time reversal symmetry.");

			}
			prm.leave_subsection ();

			prm.enter_subsection ("DFT functional parameters");
			{

				prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "true",
						Patterns::Bool(),
						"[Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed. For all-electron DFT calculation set to false.");

				prm.declare_entry("PSEUDO TESTS FLAG", "false",
						Patterns::Bool(),
						"[Developer] Boolean parameter specifying the explicit path of pseudopotential upf format files used for ctests");

				prm.declare_entry("PSEUDOPOTENTIAL FILE NAMES LIST", "",
						Patterns::Anything(),
						"[Standard] Pseudopotential file. This file contains the list of pseudopotential file names in UPF format corresponding to the atoms involved in the calculations. UPF version 2.0 or greater and norm-conserving pseudopotentials(ONCV and Troullier Martins) in UPF format are only accepted. File format (example for two atoms Mg(z=12), Al(z=13)): 12 filename1.upf(row1), 13 filename2.upf (row2). Important Note: ONCV pseudopotentials data base in UPF format can be downloaded from http://www.quantum-simulation.org/potentials/sg15_oncv.  Troullier-Martins pseudopotentials in UPF format can be downloaded from http://www.quantum-espresso.org/pseudopotentials/fhi-pp-from-abinit-web-site.");

				prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
						Patterns::Integer(1,4),
						"[Standard] Parameter specifying the type of exchange-correlation to be used: 1(LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), 2(LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), 3(LDA: Vosko, Wilk \\& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), 4(GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]).");

				prm.declare_entry("SPIN POLARIZATION", "0",
						Patterns::Integer(0,1),
						"[Standard] Spin polarization: 0 for no spin polarization and 1 for collinear spin polarization calculation. Default option is 0.");

				prm.declare_entry("START MAGNETIZATION", "0.0",
						Patterns::Double(-0.5,0.5),
						"[Standard] Starting magnetization to be used for spin-polarized DFT calculations (must be between -0.5 and +0.5). Corresponding magnetization per simulation domain will be (2 x START MAGNETIZATION x Number of electrons) a.u. ");
			}
			prm.leave_subsection ();


			prm.enter_subsection ("SCF parameters");
			{
				prm.declare_entry("TEMPERATURE", "500.0",
						Patterns::Double(1e-5),
						"[Standard] Fermi-Dirac smearing temperature (in Kelvin).");

				prm.declare_entry("MAXIMUM ITERATIONS", "200",
						Patterns::Integer(1,1000),
						"[Standard] Maximum number of iterations to be allowed for SCF convergence");

				prm.declare_entry("TOLERANCE", "5e-05",
						Patterns::Double(1e-12,1.0),
						"[Standard] SCF iterations stopping tolerance in terms of $L_2$ norm of the electron-density difference between two successive iterations. CAUTION: A tolerance close to 1e-7 or lower can deteriorate the SCF convergence due to the round-off error accumulation.");

				prm.declare_entry("MIXING HISTORY", "50",
						Patterns::Integer(1,1000),
						"[Standard] Number of SCF iteration history to be considered for density mixing schemes. For metallic systems, a mixing history larger than the default value provides better scf convergence.");

				prm.declare_entry("MIXING PARAMETER", "0.2",
						Patterns::Double(0.0,1.0),
						"[Standard] Mixing parameter to be used in density mixing schemes. Default: 0.2.");

				prm.declare_entry("KERKER MIXING PARAMETER", "0.05",
						Patterns::Double(0.0,1000.0),
						"[Standard] Mixing parameter to be used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k_{TF}**2).");

				prm.declare_entry("MIXING METHOD","ANDERSON",
						Patterns::Selection("BROYDEN|ANDERSON|ANDERSON_WITH_KERKER"),
						"[Standard] Method for density mixing. ANDERSON is the default option.");


				prm.declare_entry("CONSTRAINT MAGNETIZATION", "false",
						Patterns::Bool(),
						"[Standard] Boolean parameter specifying whether to keep the starting magnetization fixed through the SCF iterations. Default is FALSE");

				prm.declare_entry("STARTING WFC","RANDOM",
						Patterns::Selection("ATOMIC|RANDOM"),
						"[Standard] Sets the type of the starting Kohn-Sham wavefunctions guess: Atomic(Superposition of single atom atomic orbitals. Atom types for which atomic orbitals are not available, random wavefunctions are taken. Currently, atomic orbitals data is not available for all atoms.), Random(The starting guess for all wavefunctions are taken to be random). Default: RANDOM.");

				prm.declare_entry("COMPUTE ENERGY EACH ITER", "true",
						Patterns::Bool(),
						"[Advanced] Boolean parameter specifying whether to compute the total energy at the end of every SCF. Setting it to false can lead to some computational time savings.");

				prm.declare_entry("HIGHER QUAD NLP", "true",
						Patterns::Bool(),
						"[Advanced] Boolean parameter specifying whether to use a higher order quadrature rule for the calculations involving the non-local part of the pseudopotential. Default setting is true. Could be safely set to false if you are using a very refined mesh.");

				prm.enter_subsection ("Eigen-solver parameters");
				{
					prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "1",
							Patterns::Integer(0),
							"[Standard] Number of Kohn-Sham wavefunctions to be computed. For spin-polarized calculations, this parameter denotes the number of Kohn-Sham wavefunctions to be computed for each spin. A recommended value for this parameter is to set it to N/2+Nb where N is the number of electrons. Use Nb to be 5-10 percent of N/2 for insulators and for metals use Nb to be 10-15 percent of N/2. If 5-15 percent of N/2 is less than 10 wavefunctions, set Nb to be atleast 10.");

					prm.declare_entry("SPECTRUM SPLIT CORE EIGENSTATES", "0",
							Patterns::Integer(0),
							"[Advanced] Number of lowest Kohn-Sham eigenstates which should not be included in the Rayleigh-Ritz diagonalization.  In other words, only the eigenvalues and eigenvectors corresponding to the higher eigenstates (Number of Kohn-Sham wavefunctions minus the specified core eigenstates) are computed in the diagonalization of the projected Hamiltonian. This value is usually chosen to be the sum of the number of core eigenstates for each atom type multiplied by number of atoms of that type. This setting is recommended for large systems (greater than 5000 electrons). Default value is 0 i.e., no core eigenstates are excluded from the Rayleigh-Ritz projection step. Currently this optimization is not implemented for the complex executable.");

					prm.declare_entry("SPECTRUM SPLIT STARTING SCF ITER", "0",
							Patterns::Integer(0),
							"[Advanced] SCF iteration no beyond which spectrum splitting based can be used.");

					prm.declare_entry("RR GEP", "true",
							Patterns::Bool(),"[Advanced] Solve generalized eigenvalue problem instead of standard eignevalue problem in Rayleigh-Ritz step. This approach is not extended yet to complex executable. Default value is true for real executable and false for complex executable.");

					prm.declare_entry("RR GEP FULL MASS MATRIX", "false",
							Patterns::Bool(),"[Advanced] Solve generalized eigenvalue problem instead of standard eignevalue problem in Rayleigh-Ritz step with finite-element overlap matrix evaluated using full quadrature rule (Gauss quadrature rule) only during the solution of the generalized eigenvalue problem in the RR step.  Default value is false.");

					prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
							Patterns::Double(),
							"[Developer] The lower bound of the wanted eigen spectrum. It is only used for the first iteration of the Chebyshev filtered subspace iteration procedure. A rough estimate based on single atom eigen values can be used here. Default value is good enough for most problems.");

					prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "0",
							Patterns::Integer(0,2000),
							"[Advanced] Chebyshev polynomial degree to be employed for the Chebyshev filtering subspace iteration procedure to dampen the unwanted spectrum of the Kohn-Sham Hamiltonian. If set to 0, a default value depending on the upper bound of the eigen-spectrum is used. See Phani Motamarri et.al., J. Comp. Phys. 253, 308-343 (2013).");

					prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF", "1.34",
							Patterns::Double(0,2000),
							"[Advanced] Chebyshev polynomial degree first scf scaling factor.");


					prm.declare_entry("LOWER BOUND UNWANTED FRAC UPPER", "0",
							Patterns::Double(0,1),
							"[Developer] The value of the fraction of the upper bound of the unwanted spectrum, the lower bound of the unwanted spectrum will be set. Default value is 0.");

					prm.declare_entry("CHEBYSHEV FILTER TOLERANCE","5e-02",
							Patterns::Double(1e-10),
							"[Advanced] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure. Default value is sufficient for most purposes");

					prm.declare_entry("BATCH GEMM", "true",
							Patterns::Bool(),
							"[Advanced] Boolean parameter specifying whether to use gemm batch blas routines to perform matrix-matrix multiplication operations with groups of matrices, processing a number of groups at once using threads instead of the standard serial route. CAUTION: gemm batch blas routines will only be activated if the CHEBY WFC BLOCK SIZE is less than 1000, and only if intel mkl blas library is linked with the dealii installation. Default option is true.");

					prm.declare_entry("ORTHOGONALIZATION TYPE","Auto",
							Patterns::Selection("GS|LW|PGS|Auto"),
							"[Advanced] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization using SLEPc library), LW(Lowden Orthogonalization implemented using LAPACK/BLAS routines, extension to use ScaLAPACK library not implemented yet), PGS(Pseudo-Gram-Schmidt Orthogonalization: if you are using the real executable, parallel ScaLAPACK functions are used, otherwise serial LAPACK functions are used.) Auto is the default option, which chooses GS for all-electron case and PGS for pseudopotential case. GS and LW options are only available if RR GEP is set to false.");

					prm.declare_entry("ENABLE SWITCH TO GS", "true",
							Patterns::Bool(),
							"[Developer] Controls automatic switching to Gram-Schimdt orthogonalization if Lowden Orthogonalization or Pseudo-Gram-Schimdt orthogonalization are unstable. Default option is true.");


					prm.declare_entry("ENABLE SUBSPACE ROT PGS OPT", "true",
							Patterns::Bool(),
							"[Developer] Turns on subspace rotation optimization for Pseudo-Gram-Schimdt orthogonalization. Default option is true.");

					prm.declare_entry("CHEBY WFC BLOCK SIZE", "400",
							Patterns::Integer(1),
							"[Advanced] Chebyshev filtering procedure involves the matrix-matrix multiplication where one matrix corresponds to the discretized Hamiltonian and the other matrix corresponds to the wavefunction matrix. The matrix-matrix multiplication is accomplished in a loop over the number of blocks of the wavefunction matrix to reduce the memory footprint of the code. This parameter specifies the block size of the wavefunction matrix to be used in the matrix-matrix multiplication. The optimum value is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting CHEBY WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/CHEBY WFC BLOCK SIZE equals an integer value. Default value is 400.");

					prm.declare_entry("WFC BLOCK SIZE", "400",
							Patterns::Integer(1),
							"[Advanced]  This parameter specifies the block size of the wavefunction matrix to be used for memory optimization purposes in the orthogonalization, Rayleigh-Ritz, and density computation steps. The optimum block size is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/WFC BLOCK SIZE equals an integer value. Default value is 400.");

					prm.declare_entry("SUBSPACE ROT DOFS BLOCK SIZE", "5000",
							Patterns::Integer(1),
							"[Developer] This block size is used for memory optimization purposes in subspace rotation step in Pseudo-Gram-Schmidt orthogonalization and Rayleigh-Ritz steps. Default value is 5000.");

					prm.declare_entry("SCALAPACKPROCS", "0",
							Patterns::Integer(0,300),
							"[Advanced] Uses a processor grid of SCALAPACKPROCS times SCALAPACKPROCS for parallel distribution of the subspace projected matrix in the Rayleigh-Ritz step and the overlap matrix in the Pseudo-Gram-Schmidt step. Default value is 0 for which a thumb rule is used (see http://netlib.org/scalapack/slug/node106.html). If ELPA is used, twice the value obtained from the thumb rule is used as ELPA scales much better than ScaLAPACK.");

					prm.declare_entry("SCALAPACK BLOCK SIZE", "50",
							Patterns::Integer(1,300),
							"[Advanced] ScaLAPACK process grid block size.");

					prm.declare_entry("USE ELPA", "true",
							Patterns::Bool(),
							"[Standard] Use ELPA instead of ScaLAPACK for diagonalization of subspace projected Hamiltonian and Pseudo-Gram-Schmidt orthogonalization. Currently this setting is only available for real executable. Default setting is true.");

					prm.declare_entry("USE MIXED PREC PGS SR", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in subspace rotation step of PGS orthogonalization, if ORTHOGONALIZATION TYPE is set to PGS. Currently this optimization is only enabled for the real executable. Default setting is false.");

					prm.declare_entry("USE MIXED PREC PGS O", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in overlap matrix computation step of PGS orthogonalization, if ORTHOGONALIZATION TYPE is set to PGS. Currently this optimization is only enabled for the real executable. Default setting is false.");


					prm.declare_entry("USE MIXED PREC XTHX SPECTRUM SPLIT", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in computing subspace projected Kohn-Sham Hamiltonian when SPECTRUM SPLIT CORE EIGENSTATES>0. Currently this optimization is only enabled for the real executable. Default setting is false.");

					prm.declare_entry("USE MIXED PREC RR_SR SPECTRUM SPLIT", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in Rayleigh-Ritz subspace rotation step when SPECTRUM SPLIT CORE EIGENSTATES>0. Currently this optimization is only enabled for the real executable. Default setting is false.");

					prm.declare_entry("USE MIXED PREC RR_SR", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in Rayleigh-Ritz subspace rotation step. Currently this optimization is only enabled for the real executable. Default setting is false.");

					prm.declare_entry("USE MIXED PREC CHEBY", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in Chebyshev filtering. Currently this option is only available for real executable and USE ELPA=true for which DFT-FE also has to be linked to ELPA library. Default setting is false.");

					prm.declare_entry("USE SINGLE PREC XTHX OFF DIAGONAL", "false",
							Patterns::Bool(),
							"[Advanced] Use single precision arithmetic in the computation of XtHX in Rayleigh Ritz projection step for off-diagonal block entries");

					prm.declare_entry("USE MIXED PREC CHEBY NON LOCAL", "false",
							Patterns::Bool(),
							"[Advanced] Use mixed precision arithmetic in non-local HX of Chebyshev filtering. Currently this option is only available for real executable and USE ELPA=true for which DFT-FE also has to be linked to ELPA library. Default setting is false.");


					prm.declare_entry("OVERLAP COMPUTE COMMUN CHEBY", "true",
							Patterns::Bool(),
							"[Advanced] Overlap communication and computation in Chebyshev filtering. This option can only be activated for USE GPU=true. Default setting is true.");


					prm.declare_entry("OVERLAP CHEB PGS SR","false",
							Patterns::Bool(),
							"[Advanced] Overlap Chebyshev filtering and subspace rotation. This option can only be activated when RR step is skipped for certial problems. Default setting is false.");

					prm.declare_entry("COMMUN AVOIDANCE ALGO CHEBY", "false",
							Patterns::Bool(),
							"[Advanced] Communication avoidance algorithm. Not implemented for OVERLAP COMPUTE COMMUN CHEBY=true.");

					prm.declare_entry("OVERLAP COMPUTE COMMUN ORTHO RR", "true",
							Patterns::Bool(),
							"[Advanced] Overlap communication and computation in orthogonalization and Rayleigh-Ritz. This option can only be activated for USE GPU=true. Default setting is true.");

					prm.declare_entry("MIXED PREC XTHX FRAC STATES", "0",
							Patterns::Integer(0),
							"[Advanced] XTHX Mixed Precision. Temporary paramater- remove once spectrum splitting with RR GEP and mixed precision is implemented. Default value is 0.");

					prm.declare_entry("ALGO", "NORMAL",
							Patterns::Selection("NORMAL|FAST"),
							"[Standard] In the FAST mode, spectrum splitting technique is used in Rayleigh-Ritz step, and mixed precision arithmetic algorithms are used in Rayleigh-Ritz and Cholesky factorization based orthogonalization step. For spectrum splitting, 85 percent of the total number of wavefunctions are taken to be core states, which holds good for most systems including metallic systems assuming NUMBER OF KOHN-SHAM WAVEFUNCTIONS to be around 10 percent more than N/2. FAST setting is strongly recommended for large-scale (> 10k electrons) system sizes. Both NORMAL and FAST setting use Chebyshev filtered subspace iteration technique. Currently, FAST setting is only enabled for the real executable. If manual options for mixed precision and spectum splitting are being used, please use NORMAL setting for ALGO. Default setting is NORMAL.");


					prm.declare_entry("ADAPTIVE FILTER STATES", "0",
							Patterns::Integer(0),
							"[Advanced] Number of lowest Kohn-Sham eigenstates which are filtered with Chebyshev polynomial degree linearly varying from 50 percent (starting from the lowest) to 80 percent of the value specified by CHEBYSHEV POLYNOMIAL DEGREE. This imposes a step function filtering polynomial order on the ADAPTIVE FILTER STATES as filtering is done with blocks of size WFC BLOCK SIZE. This setting is recommended for large systems (greater than 5000 electrons). Default value is 0 i.e., all states are filtered with the same Chebyshev polynomial degree.");
				}
				prm.leave_subsection ();
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Poisson problem parameters");
			{
				prm.declare_entry("MAXIMUM ITERATIONS", "10000",
						Patterns::Integer(0,20000),
						"[Advanced] Maximum number of iterations to be allowed for Poisson problem convergence.");

				prm.declare_entry("TOLERANCE", "1e-10",
						Patterns::Double(0,1.0),
						"[Advanced] Absolute tolerance on the residual as stopping criterion for Poisson problem convergence.");
			}
			prm.leave_subsection ();


			prm.enter_subsection ("Helmholtz problem parameters");
			{
				prm.declare_entry("MAXIMUM ITERATIONS HELMHOLTZ", "10000",
						Patterns::Integer(0,20000),
						"[Advanced] Maximum number of iterations to be allowed for Helmholtz problem convergence.");

				prm.declare_entry("ABSOLUTE TOLERANCE HELMHOLTZ", "1e-10",
						Patterns::Double(0,1.0),
						"[Advanced] Absolute tolerance on the residual as stopping criterion for Helmholtz problem convergence.");
			}
			prm.leave_subsection ();


			prm.enter_subsection ("Molecular Dynamics");
			{
				prm.declare_entry("BOMD", "false",
						Patterns::Bool(),
						"[Standard] Perform Born-Oppenheimer NVE molecular dynamics. Input parameters for molecular dynamics have to be modified directly in the code in the file md/molecularDynamics.cc.");

				prm.declare_entry("XL BOMD", "false",
						Patterns::Bool(),
						"[Standard] Perform Extended Lagrangian Born-Oppenheimer NVE molecular dynamics. Currently not implemented for spin-polarization case.");

				prm.declare_entry("CHEBY TOL XL BOMD", "1e-6",
						Patterns::Double(0.0),
						"[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

				prm.declare_entry("CHEBY TOL XL BOMD RANK UPDATES FD", "1e-7",
						Patterns::Double(0.0),
						"[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

				prm.declare_entry("CHEBY TOL XL BOMD RESTART", "1e-9",
						Patterns::Double(0.0),
						"[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

				prm.declare_entry("MAX JACOBIAN RATIO FACTOR", "1.5",
						Patterns::Double(0.9,3.0),
						"[Developer] Maximum scaling factor for maximum jacobian ratio of FEM mesh when mesh is deformed."); 

				prm.declare_entry("STARTING TEMP NVE", "300.0",
						Patterns::Double(0.0),
						"[Developer] Starting temperature in K for NVE simulation.");   

				prm.declare_entry("TIME STEP", "0.5",
						Patterns::Double(0.0),
						"[Standard] Time step in femtoseconds."); 

				prm.declare_entry("NUMBER OF STEPS", "1000",
						Patterns::Integer(0,200000),
						"[Standard] Number of time steps.");

				prm.declare_entry("DIRAC DELTA KERNEL SCALING CONSTANT XL BOMD", "0.1",
						Patterns::Double(0.0),
						"[Developer] Dirac delta scaling kernel constant for XL BOMD.");  

				prm.declare_entry("KERNEL RANK XL BOMD", "0",
						Patterns::Integer(0,10),
						"[Standard] Maximum rank for low rank kernel update in XL BOMD.");

				prm.declare_entry("NUMBER DISSIPATION TERMS XL BOMD", "8",
						Patterns::Integer(6,8),
						"[Standard] Number of dissipation terms in XL BOMD.");

				prm.declare_entry("NUMBER PASSES RR SKIPPED XL BOMD", "0",
						Patterns::Integer(0),
						"[Standard] Number of starting chebsyev filtering passes without Rayleigh Ritz in XL BOMD.");

				prm.declare_entry("AUTO MESH STEP INTERPOLATE BOMD", "false",
						Patterns::Bool(),
						"[Standard] Perform interpolation of previous density to new auto mesh.");  

				prm.declare_entry("RATIO MESH MOVEMENT TO FORCE GAUSSIAN", "1.0",
						Patterns::Double(0.0),
						"[Standard] Ratio of mesh movement to force Gaussian."); 

				prm.declare_entry("USE ATOMIC RHO XL BOMD", "true",
						Patterns::Bool(),
						"[Standard] Use atomic rho xl bomd.");  

				prm.declare_entry("STARTING SINGLE FULL SCF XL BOMD", "false",
						Patterns::Bool(),
						"[Standard] Only do first ground-state in full scf for XL BOMD.");

				prm.declare_entry("XL BOMD RR PASS MIXED PREC", "false",
						Patterns::Bool(),
						"[Standard] Allow mixed precision for RR passes in XL BOMD.");

				prm.declare_entry("DENSITY MATRIX PERTURBATION RANK UPDATES XL BOMD", "false",
						Patterns::Bool(),
						"[Standard] Use density matrix perturbation theory for rank updates.");

				prm.declare_entry("XL BOMD STEP TIMING RUN", "false",
						Patterns::Bool(),
						"[Standard] Time one XL BOMD md step for performance measurements.");

				prm.declare_entry("XL BOMD KERNEL RANK UPDATE FD PARAMETER", "1e-2",
						Patterns::Double(0.0),
						"[Standard] Finite difference perturbation parameter.");

				prm.declare_entry("SKIP HARMONIC OSCILLATOR INITIAL STEPS XL BOMD", "false",
						Patterns::Bool(),
						"[Standard] Numerical strategy to remove oscillations in initial steps.");
			}
			prm.leave_subsection ();
		}

		void parse_parameters(ParameterHandler &prm)
		{
			dftParameters::verbosity                     = prm.get_integer("VERBOSITY");
			dftParameters::reproducible_output           = prm.get_bool("REPRODUCIBLE OUTPUT");
			dftParameters::electrostaticsHRefinement = prm.get_bool("H REFINED ELECTROSTATICS");
			dftParameters::electrostaticsPRefinement = prm.get_bool("P REFINED ELECTROSTATICS");

			prm.enter_subsection ("GPU");
			{ 
				dftParameters::useGPU= prm.get_bool("USE GPU");
				dftParameters::gpuFineGrainedTimings=prm.get_bool("FINE GRAINED GPU TIMINGS");
				dftParameters::allowFullCPUMemSubspaceRot=prm.get_bool("SUBSPACE ROT FULL CPU MEM");
				dftParameters::autoGPUBlockSizes=prm.get_bool("AUTO GPU BLOCK SIZES");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Postprocessing");
			{
				dftParameters::writeWfcSolutionFields           = prm.get_bool("WRITE WFC");
				dftParameters::writeDensitySolutionFields       = prm.get_bool("WRITE DENSITY");
				dftParameters::writeDosFile                     = prm.get_bool("WRITE DENSITY OF STATES");
				dftParameters::writeLdosFile                     = prm.get_bool("WRITE LOCAL DENSITY OF STATES");
				dftParameters::writeLocalizationLengths          = prm.get_bool("WRITE LOCALIZATION LENGTHS");
				dftParameters::readWfcForPdosPspFile            = prm.get_bool("READ ATOMIC WFC PDOS FROM PSP FILE");
				dftParameters::writeLocalizationLengths          = prm.get_bool("WRITE LOCALIZATION LENGTHS");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Parallelization");
			{
				dftParameters::npool             = prm.get_integer("NPKPT");
				dftParameters::nbandGrps         = prm.get_integer("NPBAND");
				dftParameters::bandParalOpt = prm.get_bool("BAND PARAL OPT");
				dftParameters::mpiAllReduceMessageBlockSizeMB = prm.get_double("MPI ALLREDUCE BLOCK SIZE");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Checkpointing and Restart");
			{
				chkType=prm.get_integer("CHK TYPE");
				restartFromChk=prm.get_bool("RESTART FROM CHK") && chkType!=0;
				restartMdFromChk=prm.get_bool("RESTART MD FROM CHK") && chkType!=0;
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Geometry");
			{
				dftParameters::natoms                        = prm.get_integer("NATOMS");
				dftParameters::natomTypes                    = prm.get_integer("NATOM TYPES");
				dftParameters::coordinatesFile               = prm.get("ATOMIC COORDINATES FILE");
				dftParameters::coordinatesGaussianDispFile   = prm.get("ATOMIC DISP COORDINATES FILE");
				dftParameters::domainBoundingVectorsFile     = prm.get("DOMAIN VECTORS FILE");

				prm.enter_subsection ("Optimization");
				{
					dftParameters::isIonOpt                      = prm.get_bool("ION OPT");
					dftParameters::ionOptSolver                  = prm.get("ION OPT SOLVER");
					dftParameters::nonSelfConsistentForce        = prm.get_bool("NON SELF CONSISTENT FORCE");
					dftParameters::isIonForce                    = dftParameters::isIonOpt || prm.get_bool("ION FORCE");
					dftParameters::forceRelaxTol                 = prm.get_double("FORCE TOL");
					dftParameters::ionRelaxFlagsFile             = prm.get("ION RELAX FLAGS FILE");
					dftParameters::isCellOpt                     = prm.get_bool("CELL OPT");
					dftParameters::isCellStress                  = dftParameters::isCellOpt || prm.get_bool("CELL STRESS");
					dftParameters::stressRelaxTol                = prm.get_double("STRESS TOL");
					dftParameters::cellConstraintType            = prm.get_integer("CELL CONSTRAINT TYPE");
					dftParameters::reuseWfcGeoOpt                = prm.get_bool("REUSE WFC");
					dftParameters::reuseDensityGeoOpt            = prm.get_bool("REUSE DENSITY");
				}
				prm.leave_subsection ();
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Boundary conditions");
			{
				dftParameters::radiusAtomBall                = prm.get_double("SELF POTENTIAL RADIUS");
				dftParameters::periodicX                     = prm.get_bool("PERIODIC1");
				dftParameters::periodicY                     = prm.get_bool("PERIODIC2");
				dftParameters::periodicZ                     = prm.get_bool("PERIODIC3");
				dftParameters::constraintsParallelCheck      = prm.get_bool("CONSTRAINTS PARALLEL CHECK");
				dftParameters::createConstraintsFromSerialDofhandler = prm.get_bool("CONSTRAINTS FROM SERIAL DOFHANDLER");
				dftParameters::pinnedNodeForPBC = prm.get_bool("POINT WISE DIRICHLET CONSTRAINT");
				dftParameters::smearedNuclearCharges = prm.get_bool("SMEARED NUCLEAR CHARGES");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Finite element mesh parameters");
			{
				dftParameters::finiteElementPolynomialOrder  = prm.get_integer("POLYNOMIAL ORDER");
				dftParameters::meshFileName                  = prm.get("MESH FILE");
				prm.enter_subsection ("Auto mesh generation parameters");
				{
					dftParameters::outerAtomBallRadius           = prm.get_double("ATOM BALL RADIUS");
					dftParameters::innerAtomBallRadius           = prm.get_double("INNER ATOM BALL RADIUS");
					dftParameters::meshSizeOuterDomain           = prm.get_double("BASE MESH SIZE");
					dftParameters::meshSizeInnerBall             = prm.get_double("MESH SIZE AT ATOM");
					dftParameters::meshSizeOuterBall             = prm.get_double("MESH SIZE AROUND ATOM");
					dftParameters::meshAdaption                  = prm.get_bool("MESH ADAPTION");
					dftParameters::autoUserMeshParams            = prm.get_bool("AUTO USER MESH PARAMS");
					dftParameters::topfrac                       = prm.get_double("TOP FRAC");
					dftParameters::numLevels                     = prm.get_double("NUM LEVELS");
					dftParameters::numberWaveFunctionsForEstimate = prm.get_integer("ERROR ESTIMATE WAVEFUNCTIONS");
					dftParameters::toleranceKinetic = prm.get_double("TOLERANCE FOR MESH ADAPTION");
					dftParameters::gaussianConstantForce = prm.get_double("GAUSSIAN CONSTANT FORCE GENERATOR");
					dftParameters::gaussianOrderForce = prm.get_double("GAUSSIAN ORDER FORCE GENERATOR");
					dftParameters::gaussianOrderMoveMeshToAtoms = prm.get_double("GAUSSIAN ORDER MOVE MESH TO ATOMS");
					dftParameters::useFlatTopGenerator = prm.get_bool("USE FLAT TOP GENERATOR");
					dftParameters::useMeshSizesFromAtomsFile = prm.get_bool("USE MESH SIZES FROM ATOM LOCATIONS FILE");
				}
				prm.leave_subsection ();
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Brillouin zone k point sampling options");
			{
				prm.enter_subsection ("Monkhorst-Pack (MP) grid generation");
				{
					dftParameters::nkx        = prm.get_integer("SAMPLING POINTS 1");
					dftParameters::nky        = prm.get_integer("SAMPLING POINTS 2");
					dftParameters::nkz        = prm.get_integer("SAMPLING POINTS 3");
					dftParameters::offsetFlagX        = prm.get_integer("SAMPLING SHIFT 1");
					dftParameters::offsetFlagY        = prm.get_integer("SAMPLING SHIFT 2");
					dftParameters::offsetFlagZ        = prm.get_integer("SAMPLING SHIFT 3");
				}
				prm.leave_subsection ();

				dftParameters::useSymm                  = prm.get_bool("USE GROUP SYMMETRY");
				dftParameters::timeReversal                   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
				dftParameters::kPointDataFile                = prm.get("kPOINT RULE FILE");

			}
			prm.leave_subsection ();

			prm.enter_subsection ("DFT functional parameters");
			{
				dftParameters::isPseudopotential             = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
				dftParameters::pseudoTestsFlag               = prm.get_bool("PSEUDO TESTS FLAG");
				dftParameters::pseudoPotentialFile           = prm.get("PSEUDOPOTENTIAL FILE NAMES LIST");
				dftParameters::xc_id                         = prm.get_integer("EXCHANGE CORRELATION TYPE");
				dftParameters::spinPolarized                 = prm.get_integer("SPIN POLARIZATION");
				dftParameters::start_magnetization           = prm.get_double("START MAGNETIZATION");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("SCF parameters");
			{
				dftParameters::TVal                          = prm.get_double("TEMPERATURE");
				dftParameters::numSCFIterations              = prm.get_integer("MAXIMUM ITERATIONS");
				dftParameters::selfConsistentSolverTolerance = prm.get_double("TOLERANCE");
				dftParameters::mixingHistory                 = prm.get_integer("MIXING HISTORY");
				dftParameters::mixingParameter               = prm.get_double("MIXING PARAMETER");
				dftParameters::kerkerParameter               = prm.get_double("KERKER MIXING PARAMETER");
				dftParameters::mixingMethod                  = prm.get("MIXING METHOD");
				dftParameters::constraintMagnetization       = prm.get_bool("CONSTRAINT MAGNETIZATION");
				dftParameters::startingWFCType               = prm.get("STARTING WFC");
				dftParameters::computeEnergyEverySCF         = prm.get_bool("COMPUTE ENERGY EACH ITER");
				dftParameters::useHigherQuadNLP              = prm.get_bool("HIGHER QUAD NLP");


				prm.enter_subsection ("Eigen-solver parameters");
				{
					dftParameters::numberEigenValues             = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
					dftParameters::numCoreWfcRR                  = prm.get_integer("SPECTRUM SPLIT CORE EIGENSTATES");
					dftParameters::spectrumSplitStartingScfIter  = prm.get_integer("SPECTRUM SPLIT STARTING SCF ITER");
					dftParameters::rrGEP= prm.get_bool("RR GEP");
					dftParameters::rrGEPFullMassMatrix = prm.get_bool("RR GEP FULL MASS MATRIX");
					dftParameters::lowerEndWantedSpectrum        = prm.get_double("LOWER BOUND WANTED SPECTRUM");
					dftParameters::lowerBoundUnwantedFracUpper   = prm.get_double("LOWER BOUND UNWANTED FRAC UPPER");
					dftParameters::chebyshevOrder                = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
					dftParameters::useELPA= prm.get_bool("USE ELPA");
					dftParameters::useBatchGEMM= prm.get_bool("BATCH GEMM");
					dftParameters::orthogType        = prm.get("ORTHOGONALIZATION TYPE");
					dftParameters::chebyshevTolerance = prm.get_double("CHEBYSHEV FILTER TOLERANCE");
					dftParameters::wfcBlockSize= prm.get_integer("WFC BLOCK SIZE");
					dftParameters::chebyWfcBlockSize= prm.get_integer("CHEBY WFC BLOCK SIZE");
					dftParameters::subspaceRotDofsBlockSize= prm.get_integer("SUBSPACE ROT DOFS BLOCK SIZE");
					dftParameters::enableSwitchToGS= prm.get_bool("ENABLE SWITCH TO GS");
					dftParameters::triMatPGSOpt= prm.get_bool("ENABLE SUBSPACE ROT PGS OPT");
					dftParameters::scalapackParalProcs= prm.get_integer("SCALAPACKPROCS");
					dftParameters::scalapackBlockSize= prm.get_integer("SCALAPACK BLOCK SIZE");
					dftParameters::useMixedPrecPGS_SR= prm.get_bool("USE MIXED PREC PGS SR");
					dftParameters::useMixedPrecPGS_O= prm.get_bool("USE MIXED PREC PGS O");
					dftParameters::useMixedPrecXTHXSpectrumSplit= prm.get_bool("USE MIXED PREC XTHX SPECTRUM SPLIT");
					dftParameters::useMixedPrecSubspaceRotSpectrumSplit= prm.get_bool("USE MIXED PREC RR_SR SPECTRUM SPLIT");
					dftParameters::useMixedPrecSubspaceRotRR= prm.get_bool("USE MIXED PREC RR_SR");
					dftParameters::useMixedPrecCheby= prm.get_bool("USE MIXED PREC CHEBY");
					dftParameters::useMixedPrecChebyNonLocal= prm.get_bool("USE MIXED PREC CHEBY NON LOCAL");
					dftParameters::chebyCommunAvoidanceAlgo= prm.get_bool("COMMUN AVOIDANCE ALGO CHEBY");
					dftParameters::useAsyncChebPGS_SR = prm.get_bool("OVERLAP CHEB PGS SR");
					dftParameters::useSinglePrecXtHXOffDiag=prm.get_bool("USE SINGLE PREC XTHX OFF DIAGONAL");
					dftParameters::overlapComputeCommunCheby= prm.get_bool("OVERLAP COMPUTE COMMUN CHEBY");
					dftParameters::overlapComputeCommunOrthoRR= prm.get_bool("OVERLAP COMPUTE COMMUN ORTHO RR");
					dftParameters::mixedPrecXtHXFracStates  = prm.get_integer("MIXED PREC XTHX FRAC STATES");
					dftParameters::algoType= prm.get("ALGO");
					dftParameters::numAdaptiveFilterStates= prm.get_integer("ADAPTIVE FILTER STATES");
					dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor=prm.get_double("CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF");
				}
				prm.leave_subsection ();
			}
			prm.leave_subsection ();


			prm.enter_subsection ("Poisson problem parameters");
			{
				dftParameters::maxLinearSolverIterations     = prm.get_integer("MAXIMUM ITERATIONS");
				dftParameters::absLinearSolverTolerance      = prm.get_double("TOLERANCE");
			}
			prm.leave_subsection ();

			prm.enter_subsection("Helmholtz problem parameters");
			{
				dftParameters::maxLinearSolverIterationsHelmholtz = prm.get_integer("MAXIMUM ITERATIONS HELMHOLTZ");
				dftParameters::absLinearSolverToleranceHelmholtz  = prm.get_double("ABSOLUTE TOLERANCE HELMHOLTZ");
			}
			prm.leave_subsection ();

			prm.enter_subsection ("Molecular Dynamics");
			{
				dftParameters::isBOMD                        = prm.get_bool("BOMD");
				dftParameters::maxJacobianRatioFactorForMD   = prm.get_double("MAX JACOBIAN RATIO FACTOR");
				dftParameters::isXLBOMD                      = prm.get_bool("XL BOMD");
				dftParameters::chebyshevFilterTolXLBOMD      = prm.get_double("CHEBY TOL XL BOMD");
				dftParameters::chebyshevFilterTolXLBOMDRankUpdates      = prm.get_double("CHEBY TOL XL BOMD RANK UPDATES FD");
				dftParameters::timeStepBOMD                  = prm.get_double("TIME STEP");
				dftParameters::numberStepsBOMD               = prm.get_integer("NUMBER OF STEPS"); 
				dftParameters::startingTempBOMDNVE           = prm.get_double("STARTING TEMP NVE");  
				dftParameters::diracDeltaKernelScalingConstant     = prm.get_double("DIRAC DELTA KERNEL SCALING CONSTANT XL BOMD"); 
				dftParameters::kernelUpdateRankXLBOMD        = prm.get_integer("KERNEL RANK XL BOMD");
				dftParameters::kmaxXLBOMD        = prm.get_integer("NUMBER DISSIPATION TERMS XL BOMD");
				dftParameters::numberPassesRRSkippedXLBOMD        = prm.get_integer("NUMBER PASSES RR SKIPPED XL BOMD");
				dftParameters::autoMeshStepInterpolateBOMD   = prm.get_bool("AUTO MESH STEP INTERPOLATE BOMD");
				dftParameters::ratioOfMeshMovementToForceGaussianBOMD       = prm.get_double("RATIO MESH MOVEMENT TO FORCE GAUSSIAN");    
				dftParameters::useAtomicRhoXLBOMD     = prm.get_bool("USE ATOMIC RHO XL BOMD");   
				dftParameters::useSingleFullScfXLBOMD = prm.get_bool("STARTING SINGLE FULL SCF XL BOMD");
				dftParameters::skipHarmonicOscillatorTermInitialStepsXLBOMD= prm.get_bool("SKIP HARMONIC OSCILLATOR INITIAL STEPS XL BOMD"); 
				dftParameters::xlbomdRestartChebyTol           = prm.get_double("CHEBY TOL XL BOMD RESTART");  
				dftParameters::xlbomdRRPassMixedPrec           = prm.get_bool("XL BOMD RR PASS MIXED PREC");
				dftParameters::useDensityMatrixPerturbationRankUpdates           = prm.get_bool("DENSITY MATRIX PERTURBATION RANK UPDATES XL BOMD");  
				dftParameters::xlbomdStepTimingRun           = prm.get_bool("XL BOMD STEP TIMING RUN");
				dftParameters::xlbomdKernelRankUpdateFDParameter=prm.get_double("XL BOMD KERNEL RANK UPDATE FD PARAMETER");
			}
			prm.leave_subsection ();

			if ((restartFromChk==true || dftParameters::restartMdFromChk) && (chkType==1 || chkType==3))
			{
				if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
					dftParameters::coordinatesFile="atomsFracCoordAutomesh.chk";
				else
					dftParameters::coordinatesFile="atomsCartCoordAutomesh.chk";

				dftParameters::domainBoundingVectorsFile="domainBoundingVectors.chk";

				dftParameters::coordinatesGaussianDispFile="atomsGaussianDispCoord.chk";
			}

			if (dftParameters::algoType=="FAST")
			{
				dftParameters::useMixedPrecPGS_O=true;
				dftParameters::useMixedPrecPGS_SR=true;
				dftParameters::useMixedPrecXTHXSpectrumSplit=true;
				dftParameters::useMixedPrecCheby=true;
				dftParameters::computeEnergyEverySCF=false;
			}
#ifdef USE_COMPLEX
			dftParameters::rrGEP=false;
			dftParameters::rrGEPFullMassMatrix=false;
#endif

			if (!dftParameters::isPseudopotential)
			{
				dftParameters::rrGEP=false;
				dftParameters::rrGEPFullMassMatrix=false;
			}

#ifndef DFTFE_WITH_ELPA
			dftParameters::useELPA=false;
#endif

			if (dftParameters::isCellStress)
				dftParameters::electrostaticsHRefinement=false;
			//
			check_print_parameters(prm);
			setHeuristicParameters();
		}



		void check_print_parameters(const dealii::ParameterHandler &prm)
		{
			if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
			{
				std::cout << "==========================================================================================================" << std::endl ;
				std::cout << "==========================================================================================================" << std::endl ;
				std::cout << "			Welcome to the Open Source program DFT-FE development version			        " << std::endl ;
				std::cout << "This is a C++ code for materials modeling from first principles using Kohn-Sham density functional theory." << std::endl ;
				std::cout << "This is a real-space code for periodic, semi-periodic and non-periodic pseudopotential" << std::endl ;
				std::cout << "and all-electron calculations, and is based on adaptive finite-element discretization." << std::endl ;
				std::cout << "For further details, and citing, please refer to our website: https://sites.google.com/umich.edu/dftfe" << std::endl ;
				std::cout << "==========================================================================================================" << std::endl ;
				std::cout << " DFT-FE Principal developers and Mentors (alphabetically) :									" << std::endl ;
				std::cout << "														" << std::endl ;
				std::cout << " Sambit Das               - University of Michigan, Ann Arbor" << std::endl ;
				std::cout << " Vikram Gavini (Mentor)   - University of Michigan, Ann Arbor" << std::endl ;
				std::cout << " Krishnendu Ghosh         - University of Michigan, Ann Arbor" << std::endl ;
				std::cout << " Phani Motamarri          - University of Michigan, Ann Arbor" << std::endl ;
				std::cout << " Shiva Rudraraju          - University of Wisconsin-Madison  " << std::endl ;
				std::cout << " (A complete list of the many authors that have contributed to DFT-FE can be found in the authors file)"<< std::endl;
				std::cout <<  "==========================================================================================================" << std::endl ;
				std::cout << " 	     Copyright (c) 2017-2019 The Regents of the University of Michigan and DFT-FE authors         " << std::endl ;
				std::cout << " 			DFT-FE is published under [LGPL v2.1 or newer] 				" << std::endl ;
				std::cout << "==========================================================================================================" << std::endl ;
				std::cout << "==========================================================================================================" << std::endl ;
			}

			const bool printParametersToFile=false;
			if (printParametersToFile && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
			{
				prm.print_parameters (std::cout, ParameterHandler::OutputStyle::LaTeX);
				exit(0);
			}

			if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0 &&  dftParameters::verbosity>=1)
			{
				prm.print_parameters (std::cout, ParameterHandler::ShortText);
			}

			AssertThrow((dftParameters::smearedNuclearCharges && dftParameters::useFlatTopGenerator) || (!dftParameters::smearedNuclearCharges)  
					,ExcMessage("DFT-FE Error: USE FLAT TOP GENERATOR must be set to true if SMEARED NUCLEAR CHARGES is set to true."));

			AssertThrow(!((dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ) && (dftParameters::writeLdosFile || dftParameters::writePdosFile)),ExcMessage("DFT-FE Error: LOCAL DENSITY OF STATES and PROJECTED DENSITY OF STATES are currently not implemented in the case of periodic and semi-periodic boundary conditions."));

#ifdef USE_COMPLEX
			if (dftParameters::isIonForce || dftParameters::isCellStress)
				AssertThrow(!dftParameters::useSymm,ExcMessage("DFT-FE Error: USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));


			if (dftParameters::numCoreWfcRR>0)
				AssertThrow(false,ExcMessage("DFT-FE Error: SPECTRUM SPLIT CORE EIGENSTATES cannot be set to a non-zero value when using complex executable. This optimization will be added in a future release"));
#else
			AssertThrow( dftParameters::nkx==1 &&  dftParameters::nky==1 &&  dftParameters::nkz==1
					&& dftParameters::offsetFlagX==0 &&  dftParameters::offsetFlagY==0 &&  dftParameters::offsetFlagZ==0
					,ExcMessage("DFT-FE Error: Real executable cannot be used for non-zero k point."));

			AssertThrow(!dftParameters::isCellStress,ExcMessage("DFT-FE Error: Currently CELL STRESS cannot be set true if using real executable for a periodic Gamma point problem. This functionality will be added soon."));
#endif
			AssertThrow(!(dftParameters::chkType==2 && (dftParameters::isIonOpt || dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=2 cannot be used if geometry optimization is being performed."));

			AssertThrow(!(dftParameters::chkType==1 && (dftParameters::isIonOpt && dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=1 cannot be used if both ION OPT and CELL OPT are set to true."));

			AssertThrow(dftParameters::nbandGrps<=dftParameters::numberEigenValues
					,ExcMessage("DFT-FE Error: NPBAND is greater than NUMBER OF KOHN-SHAM WAVEFUNCTIONS."));

			if (dftParameters::nonSelfConsistentForce)
				AssertThrow(false,ExcMessage("DFT-FE Error: Implementation of this feature is not completed yet."));

			AssertThrow(!dftParameters::coordinatesFile.empty()
					,ExcMessage("DFT-FE Error: ATOMIC COORDINATES FILE not given."));

			AssertThrow(!dftParameters::domainBoundingVectorsFile.empty()
					,ExcMessage("DFT-FE Error: DOMAIN VECTORS FILE not given."));

			if (dftParameters::isPseudopotential)
				AssertThrow(!dftParameters::pseudoPotentialFile.empty(),
						ExcMessage("DFT-FE Error: PSEUDOPOTENTIAL FILE NAMES LIST not given."));

			if (dftParameters::spinPolarized==0)
				AssertThrow(!dftParameters::constraintMagnetization,
						ExcMessage("DFT-FE Error: This is a SPIN UNPOLARIZED calculation. Can't have CONSTRAINT MAGNETIZATION ON."));

			if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
				if (dftParameters::constraintMagnetization)
					std::cout << " WARNING: CONSTRAINT MAGNETIZATION is ON. A fixed occupation will be used no matter what temperature is provided at input" << std::endl;

			AssertThrow(dftParameters::numberEigenValues!=0
					,ExcMessage("DFT-FE Error: Number of wavefunctions not specified or given value of zero, which is not allowed."));

			AssertThrow(dftParameters::natoms!=0
					,ExcMessage("DFT-FE Error: Number of atoms not specified or given a value of zero, which is not allowed."));

			AssertThrow(dftParameters::natomTypes!=0
					,ExcMessage("DFT-FE Error: Number of atom types not specified or given a value of zero, which is not allowed."));

			if(dftParameters::meshAdaption)
				AssertThrow(!(dftParameters::isIonOpt && dftParameters::isCellOpt),ExcMessage("DFT-FE Error: Currently Atomic relaxation does not work with automatic mesh adaption scheme."));

			if(dftParameters::nbandGrps>1)
				AssertThrow(dftParameters::wfcBlockSize==dftParameters::chebyWfcBlockSize,ExcMessage("DFT-FE Error: WFC BLOCK SIZE and CHEBY WFC BLOCK SIZE must be same for band parallelization."));

#ifndef WITH_MKL;
			dftParameters::useBatchGEMM=false;
			if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
				std::cout <<"Setting USE BATCH GEMM=false as intel mkl blas library is not being linked to."<<std::endl;
#endif

#ifndef USE_PETSC;
			AssertThrow(dftParameters::isPseudopotential,ExcMessage("DFT-FE Error: Please link to dealii installed with petsc and slepc for all-electron calculations."));
#endif

#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
			{
				if (dftParameters::nbandGrps>1)
					AssertThrow(dftParameters::rrGEP
							,ExcMessage("DFT-FE Error: if band parallelization is used, RR GEP must be set to true."));
			}
#endif
			if (dftParameters::useMixedPrecCheby)
				AssertThrow(dftParameters::useELPA
						,ExcMessage("DFT-FE Error: USE ELPA must be set to true for USE MIXED PREC CHEBY."));

		}


		//FIXME: move this to triangulation manager, and set data members there
		//without changing the global dftParameters
		void setHeuristicParameters()
		{
			if (dftParameters::meshSizeOuterDomain<1.0e-6)
				if (dftParameters::periodicX ||dftParameters::periodicY ||dftParameters::periodicZ)
					dftParameters::meshSizeOuterDomain=4.0;
				else
					dftParameters::meshSizeOuterDomain=13.0;

			if (dftParameters::meshSizeInnerBall<1.0e-6)
				if (dftParameters::isPseudopotential)
					dftParameters::meshSizeInnerBall=2.0*dftParameters::meshSizeOuterBall;
				else
					dftParameters::meshSizeInnerBall=0.1*dftParameters::meshSizeOuterBall;

			if (dftParameters::isPseudopotential && dftParameters::orthogType=="Auto")
			{
				if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
					std::cout <<"Setting ORTHOGONALIZATION TYPE=PGS for pseudopotential calculations "<<std::endl;
				dftParameters::orthogType="PGS";
			}
			else if (!dftParameters::isPseudopotential && dftParameters::orthogType=="Auto")
			{
				if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
					std::cout <<"Setting ORTHOGONALIZATION TYPE=GS for all-electron calculations "<<std::endl;

				dftParameters::orthogType="GS";
			}

			if (!(dftParameters::periodicX ||dftParameters::periodicY ||dftParameters::periodicZ)
					&&!dftParameters::reproducible_output)
			{
				dftParameters::constraintsParallelCheck=false;
				dftParameters::createConstraintsFromSerialDofhandler=false;
			}
			else if (dftParameters::reproducible_output)
				dftParameters::createConstraintsFromSerialDofhandler=true;

			if (dftParameters::reproducible_output)
			{
				dftParameters::gaussianOrderMoveMeshToAtoms=4.0;
			}

      //if (dftParameters::smearedNuclearCharges)
      //{
      //  dftParameters::gaussianOrderForce=3.0;
      //}

		}

	}

}
