// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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

/** @file force.h
 *  @brief computes configurational forces in KSDFT
 *
 *  This class computes and stores the configurational forces corresponding to geometry optimization.
 *  It uses the formulation in the paper by Motamarri et.al. (https://arxiv.org/abs/1712.05535) 
 *  which provides an unified approach to atomic forces corresponding to internal atomic relaxation 
 *  and cell stress corresponding to cell relaxation.
 *
 *  @author Sambit Das
 */

#ifndef force_H_
#define force_H_
#include "headers.h"
#include "constants.h"
#include "geoOptIon.h"
#include "meshMovementGaussian.h"

using namespace dealii;
typedef dealii::parallel::distributed::Vector<double> vectorType;
template <unsigned int T> class dftClass;

template <unsigned int FEOrder>
class forceClass
{
  template <unsigned int T>  friend class dftClass;
  template <unsigned int T>  friend class geoOptIon;
public:
/** @brief Constructor.
 *
 *  @param _dftPtr pointer to dftClass
 *  @param mpi_comm_replica mpi_communicator of the current pool
 */   
  forceClass(dftClass<FEOrder>* _dftPtr,  MPI_Comm &mpi_comm_replica);

/** @brief initializes data structures inside forceClass assuming unmoved triangulation.
 *
 *  initUnmoved is the first step of the initialization/reinitialization of force class when
 *  starting from a new unmoved triangulation. It creates the dofHandler with linear finite elements and
 *  three components corresponding to the three force components. It also creates the corresponding constraint
 *  matrices which is why an unmoved triangulation is necessary. Finally this function also initializes the 
 *  gaussianMovePar data member.
 *
 *  @param triangulation reference to unmoved triangulation where the mesh nodes have not been manually moved.
 *  @return void.
 */    
  void initUnmoved(Triangulation<3,3> & triangulation);

/** @brief initializes data structures inside forceClass which depend on the moved mesh.
 *
 *  initMoved is the second step (first step call initUnmoved) of the initialization/reinitialization of force class when
 *  starting from a new mesh, and the first step when recomputing forces on a
 *  perturbed mesh. initMoved assumes that the triangulation whose reference was passed to the forceClass object
 *  in the initUnmoved call now has its nodes moved such that all atomic positions lie on nodes.
 *
 *  @return void. 
 */  
  void initMoved();

/** @brief initializes and precomputes pseudopotential related data structuers required for configurational force
 *  and stress computation.
 *
 *  This function is only activated for pseudopotential calculations and is currently called when initializing/reinitializing
 *  the dftClass object. This function initializes and precomputes the pseudopotential datastructures for local and non-local
 *  parts. Separate internal function calls are made for KB and ONCV projectors.
 *
 *  @return void. 
 */  
  void initPseudoData();

/** @brief computes the configurational force on all atoms corresponding to a Gaussian generator,
 *  which represents perturbation of the underlying space.
 *
 *  The Gaussian generator is taken to be exp(-d_gaussianConstant*r^2), r being the distance from the atom.
 *  Currently d_gaussianConstant is hardcoded to be 4.0. To get the computed atomic forces use
 *  getAtomsForces
 *
 *  @return void. 
 */  
  void computeAtomsForces();

/** @brief returns a copy of the configurational force on all global atoms.
 *
 *  computeAtomsForces must be called prior to this function call.
 *
 *  @return std::vector<double> flattened array of the configurational force on all atoms,
 *  the three force components on each atom being the leading dimension. Units- Hartree/Bohr
 */ 
  std::vector<double> getAtomsForces(); 

/** @brief prints the currently stored configurational forces on atoms and the Gaussian generator constant 
 *  used to compute them.
 *
 *  @return void. 
 */  
  void printAtomsForces();

#ifdef ENABLE_PERIODIC_BC
/** @brief computes the configurational stress on the domain corresponding to 
 *  affine deformation of the periodic cell.
 *
 *  This function cannot be called for fully non-periodic calculations.
 *
 *  @return void. 
 */    
  void computeStress();

/** @brief returns a copy of the current stress tensor value.
 *
 *  computeStress must be call prior to this function call.
 *
 *  @return Tensor<2,C_DIM,double>  second order stress Tensor in Hartree/Bohr^3
 */    
  Tensor<2,C_DIM,double> getStress();

/** @brief prints the currently stored configurational stress tensor.
 *
 *  @return void. 
 */   
  void printStress();  
#endif

/** @brief Updates atom positions, remeshes/moves mesh and calls appropriate reinits. 
 *  
 *  Function to update the atom positions and mesh based on the provided displacement input.
 *  Depending on the maximum displacement magnitude this function decides wether to do auto remeshing
 *  or move mesh using Gaussian functions. Additionaly this function also wraps the atom position across the 
 *  periodic boundary if the atom moves across it.
 *
 *  @param globalAtomsDisplacements vector containing the displacements (from current position) of all atoms (global). 
 *  @return void.
 */    
  void updateAtomPositionsAndMoveMesh(const std::vector<Point<C_DIM> > & globalAtomsDisplacements);

private:

/** @brief Locates and stores the global dof indices of d_dofHandlerForce whose cooridinates match
 *  with the atomic positions.
 *
 *  @return void. 
 */     
  void locateAtomCoreNodesForce(); 

  void createBinObjectsForce();

  void configForceLinFEInit();

  void configForceLinFEFinalize();

  void computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(); 

  void computeConfigurationalForceSpinPolarizedEEshelbyTensorFPSPFnlLinFE();    

  void computeConfigurationalForcePhiExtLinFE();

  void computeConfigurationalForceEselfLinFE();

  void computeConfigurationalForceEselfNoSurfaceLinFE();

  void computeConfigurationalForceTotalLinFE();

  void FPSPLocalGammaAtomsElementalContribution(std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
		                                FEValues<C_DIM> & feVselfValues,
                                                FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,					    
				                const unsigned int cell,
			                        const std::vector<VectorizedArray<double> > & rhoQuads);

  void distributeForceContributionFPSPLocalGammaAtoms(const std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms); 

#ifdef ENABLE_PERIODIC_BC   
  void FnlGammaAtomsElementalContributionPeriodic
      (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
       FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
       const unsigned int cell,
       const std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > > & pspnlGammaAtomsQuads,
       const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiTimesV,       
       const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiQuads);  

  void FnlGammaAtomsElementalContributionPeriodicSpinPolarized
      (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
       FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
       const unsigned int cell,
       const std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > > & pspnlGammaAtomsQuads,
       const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiSpin0TimesV,
       const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiSpin1TimesV,    
       const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiSpin0Quads,
       const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiSpin1Quads); 
#else  

  void FnlGammaAtomsElementalContributionNonPeriodicSpinPolarized
      (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
       FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
       const unsigned int cell,
       const std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > pspnlGammaAtomQuads,
       const std::vector<std::vector<double> >  & projectorKetTimesPsiSpin0TimesV,
       const std::vector<std::vector<double> >  & projectorKetTimesPsiSpin1TimesV,       
       const std::vector< VectorizedArray<double> > & psiSpin0Quads,
       const std::vector< VectorizedArray<double> > & psiSpin1Quads);

  void FnlGammaAtomsElementalContributionNonPeriodic
      (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
       FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
       const unsigned int cell,
       const std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > pspnlGammaAtomQuads,
       const std::vector<std::vector<double> >  & projectorKetTimesPsiTimesV,			       
       const std::vector< VectorizedArray<double> > & psiQuads);  

#endif 

  void distributeForceContributionFnlGammaAtoms(const std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms);  

  void computeAtomsForcesGaussianGenerator(bool allowGaussianOverlapOnAtoms=false);

#ifdef ENABLE_PERIODIC_BC 
  void computeStressEself();

  void computeStressEEshelbyEPSPEnlEk();

  void computeStressSpinPolarizedEEshelbyEPSPEnlEk();

  void addEPSPStressContribution
                               (FEValues<C_DIM> & feVselfValues,
			        FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
			        const unsigned int cell,
			        const std::vector<VectorizedArray<double> > & rhoQuads);  
#endif

  void initLocalPseudoPotentialForce();

  void computeElementalNonLocalPseudoDataForce();

  void computeElementalNonLocalPseudoOVDataForce();  

  void computeNonLocalProjectorKetTimesPsiTimesV(const std::vector<vectorType*> &src,
                                                 std::vector<std::vector<double> > & projectorKetTimesPsiTimesVReal,
                                                 std::vector<std::vector<std::complex<double> > > & projectorKetTimesPsiTimesVComplex,
						 const unsigned int kPointIndex);


  /// Parallel distributed vector field which stores the configurational force for each fem node corresponding
  /// to linear shape function generator (see equations 52-53 in (https://arxiv.org/abs/1712.05535)).
  /// This vector doesn't contain contribution from terms which have sums over k points.
  vectorType d_configForceVectorLinFE;

#ifdef ENABLE_PERIODIC_BC 
  /// Parallel distributed vector field which stores the configurational force for each fem node corresponding
  /// to linear shape function generator (see equations 52-53 in (https://arxiv.org/abs/1712.05535)).
  /// This vector only contains contribution from terms which have sums over k points.
  vectorType d_configForceVectorLinFEKPoints;
#endif


#ifdef ENABLE_PERIODIC_BC
  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational force computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points*2>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_ZetalmDeltaVl;

  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational force computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points*num_k_points*3*2>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */  
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl_KPoint;

  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational force computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points*num_k_points*3*2>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */  
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint; 

  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational stress computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points*num_k_points*3*3*2>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */   
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint;

#else

  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational stress computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */    
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_ZetalmDeltaVl;

  /* Storage for precomputed nonlocal pseudopotential quadrature data. This is to speedup the 
   * configurational stress computation. Data format: vector(numNonLocalAtomsCurrentProcess with 
   * non-zero compact support, vector(number pseudo wave functions,map<cellid,num_quad_points*3>)).
   * Refer to (https://arxiv.org/abs/1712.05535) for details of the expression of the configurational force terms 
   * for the norm-conserving Troullier-Martins pseudopotential in the Kleinman-Bylander form. 
   * The same expressions also extend to the Optimized Norm-Conserving Vanderbilt (ONCV) pseudopotentials.
   */    
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl;
#endif

  /// Internal data: map for cell id to sum Vpseudo local of all atoms whose psp tail intersects the local domain. 
  std::map<dealii::CellId, std::vector<double> > d_gradPseudoVLoc;

  /// Internal data:: map for cell id to gradient of Vpseudo local of individual atoms. Only for atoms
  /// whose psp tail intersects the local domain.
  std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > d_gradPseudoVLocAtoms;

  /// meshMovementGaussianClass object  								       
  meshMovementGaussianClass gaussianMovePar;

  /// Gaussian generator constant. Gaussian generator: Gamma(r)= exp(-d_gaussianConstant*r^2)
  /// FIXME: Until the hanging nodes surface integral issue is fixed use a value >=4.0
  const double d_gaussianConstant=5.0;

  /// Storage for configurational force on all global atoms.
  std::vector<double> d_globalAtomsGaussianForces;

#ifdef ENABLE_PERIODIC_BC 
  /* Part of the configurational force which is summed over k points. 
   * It is a temporary data structure required for force evaluation (d_globalAtomsGaussianForces)
   * when parallization over k points is on.
   */  
  std::vector<double> d_globalAtomsGaussianForcesKPoints;
#endif  

#ifdef ENABLE_PERIODIC_BC

  /// Storage for configurational stress tensor 
  Tensor<2,C_DIM,double> d_stress;

  /* Part of the stress tensor which is summed over k points. 
   * It is a temporary data structure required for stress evaluation (d_stress)
   * when parallization over k points is on.
   */
  Tensor<2,C_DIM,double> d_stressKPoints;
#endif 
  /* Dont use true except for debugging forces only without mesh movement, as gaussian ovelap 
   * on atoms for move mesh is by default set to false
   */
  const bool d_allowGaussianOverlapOnAtoms=false;

  /// pointer to dft class
  dftClass<FEOrder>* dftPtr;

  /// Finite element object for configurational force computation. Linear finite elements with three force field components are used.
  FESystem<C_DIM>  FEForce;

  /* DofHandler on which we define the configurational force field. Each geometric fem node has 
   * three dofs corresponding the the three force components. The generator for the configurational 
   * force on the fem node is the linear shape function attached to it. This DofHandler is based on the same
   * triangulation on which we solve the dft problem.
   */
  DoFHandler<C_DIM> d_dofHandlerForce;

  /// Index of the d_dofHandlerForce in the MatrixFree object stored in dftClass. This is required to correctly use FEEvaluation class.
  unsigned int d_forceDofHandlerIndex;

  /// Map of locally relevant global dof index in dofHandlerForce to the cartesian coordinates of the dof 
  std::map<types::global_dof_index, Point<C_DIM> > d_supportPointsForce;

  /// IndexSet of locally owned dofs of in d_dofHandlerForce the current processor
  IndexSet   d_locally_owned_dofsForce;

  /// IndexSet of locally relevant dofs of in d_dofHandlerForce the current processor
  IndexSet   d_locally_relevant_dofsForce;

  /// Constraint matrix for hanging node and periodic constaints on d_dofHandlerForce.
  ConstraintMatrix d_constraintsNoneForce;

  /// Internal data: map < <atomId,force component>, globaldof in d_dofHandlerForce>  
  std::map<std::pair<unsigned int,unsigned int>, unsigned int>  d_atomsForceDofs;

  /// Internal data: stores cell iterators of all cells in dftPtr->d_dofHandler which are part of the vself ball. Outer vector is over vself bins.
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator> > d_cellsVselfBallsDofHandler;

  /// Internal data: stores cell iterators of all cells in d_dofHandlerForce which are part of the vself ball. Outer vector is over vself bins.
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator> > d_cellsVselfBallsDofHandlerForce;

  /// Internal data: stores map of vself ball cell Id  to the closest atom Id of that cell. Outer vector over vself bins.
  std::vector<std::map<dealii::CellId , unsigned int> > d_cellsVselfBallsClosestAtomIdDofHandler;

  /// Internal data: stores the map of atom Id (only in the local processor) to the vself bin Id.
  std::map<unsigned int, unsigned int> d_AtomIdBinIdLocalDofHandler;  
  
  /* Internal data: stores the face ids of dftPtr->d_dofHandler (single component field) on which to 
   * evaluate the vself ball surface integral in the configurational force expression. Outer vector is over
   * the vself bins. Inner map is between the cell iterator and the vector of face ids to integrate on for that
   * cell iterator.
   */  
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > > > d_cellFacesVselfBallSurfacesDofHandler;

  /* Internal data: stores the face ids of d_dofHandlerForce (three component field) on which to 
   * evaluate the vself ball surface integral in the configurational force expression. Outer vector is over
   * the vself bins. Inner map is between the cell iterator and the vector of face ids to integrate on for that
   * cell iterator.
   */
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > > > d_cellFacesVselfBallSurfacesDofHandlerForce; 

  /// mpi_communicator in the current pool
  MPI_Comm mpi_communicator;

  /// number of mpi processes in the current pool
  const unsigned int n_mpi_processes;

  /// current mpi process id in the current pool
  const unsigned int this_mpi_process;

  /// conditional stream object to enable printing only on root processor across all pools
  dealii::ConditionalOStream   pcout;

  /// compute-time logger
  dealii::TimerOutput computing_timer;

};

#endif
