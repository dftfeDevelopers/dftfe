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


#ifndef force_H_
#define force_H_
#include "vselfBinsManager.h"
#include "dftParameters.h"

#include "constants.h"
#include "headers.h"
#include "meshMovementGaussian.h"
#include "kohnShamDFTOperator.h"
#ifdef DFTFE_WITH_DEVICE
#  include "kohnShamDFTOperatorDevice.h"
#endif
#include <dftd.h>



namespace dftfe
{
  // forward declaration
  template <unsigned int T1, unsigned int T2>
  class dftClass;

  /**
   * @brief computes configurational forces in KSDFT
   *
   * This class computes and stores the configurational forces corresponding to
   * geometry optimization. It uses the formulation in the paper by Motamarri
   * et.al. (https://link.aps.org/doi/10.1103/PhysRevB.97.165132) which provides
   * an unified approach to atomic forces corresponding to internal atomic
   * relaxation and cell stress corresponding to cell relaxation.
   *
   * @author Sambit Das
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class forceClass
  {
    friend class dftClass<FEOrder, FEOrderElectro>;

  public:
    /** @brief Constructor.
     *
     *  @param _dftPtr pointer to dftClass
     *  @param mpi_comm_parent parent mpi_communicator
     *  @param mpi_comm_domain domain decomposition mpi_communicator
     */
    forceClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
               const MPI_Comm &                   mpi_comm_parent,
               const MPI_Comm &                   mpi_comm_domain,
               const dftParameters &              dftParams);

    /** @brief initializes data structures inside forceClass assuming unmoved triangulation.
     *
     *  initUnmoved is the first step of the initialization/reinitialization of
     * force class when starting from a new unmoved triangulation. It creates
     * the dofHandler with linear finite elements and three components
     * corresponding to the three force components. It also creates the
     * corresponding constraint matrices which is why an unmoved triangulation
     * is necessary. Finally this function also initializes the gaussianMovePar
     * data member.
     *
     *  @param triangulation reference to unmoved triangulation where the mesh nodes have not
     *  been manually moved.
     *  @param isElectrostaticsMesh boolean parameter specifying whether this triangulatio is to be used for
     *  for the electrostatics part of the configurational force.
     *  @return void.
     */
    void
    initUnmoved(const dealii::Triangulation<3, 3> &     triangulation,
                const dealii::Triangulation<3, 3> &     serialTriangulation,
                const std::vector<std::vector<double>> &domainBoundingVectors,
                const bool                              isElectrostaticsMesh);

    /** @brief initializes data structures inside forceClass which depend on the moved mesh.
     *
     *  initMoved is the second step (first step call initUnmoved) of the
     * initialization/reinitialization of force class when starting from a new
     * mesh, and the first step when recomputing forces on a perturbed mesh.
     * initMoved assumes that the triangulation whose reference was passed to
     * the forceClass object in the initUnmoved call now has its nodes moved
     * such that all atomic positions lie on nodes.
     *
     *  @return void.
     */
    void
    initMoved(
      std::vector<const dealii::DoFHandler<3> *> &dofHandlerVectorMatrixFree,
      std::vector<const dealii::AffineConstraints<double> *>
        &        constraintsVectorMatrixFree,
      const bool isElectrostaticsMesh);

    /** @brief initializes and precomputes pseudopotential related data structuers required for configurational force
     *  and stress computation.
     *
     *  This function is only activated for pseudopotential calculations and is
     * currently called when initializing/reinitializing the dftClass object.
     * This function initializes and precomputes the pseudopotential
     * datastructures for local and non-local parts. Separate internal function
     * calls are made for KB and ONCV projectors.
     *
     *  @return void.
     */
    void
    initPseudoData();

    /** @brief computes the configurational force on all atoms corresponding to a Gaussian generator,
     *  which represents perturbation of the underlying space.
     *
     *  The Gaussian generator is taken to be exp(-d_gaussianConstant*r^2), r
     * being the distance from the atom. Currently d_gaussianConstant is
     * hardcoded to be 4.0. To get the computed atomic forces use getAtomsForces
     *
     *  @return void.
     */
    void
    computeAtomsForces(
      const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                  kohnShamDFTEigenOperator,
      const dispersionCorrection &         dispersionCorr,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoOutValues,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,      
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                                  hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                      pseudoVLocAtomsElectro,
      const dealii::AffineConstraints<double> &hangingPlusPBCConstraintsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
      const std::map<dealii::CellId, std::vector<double>>
        &                              shadowKSGradRhoMinValues,
      const distributedCPUVec<double> &phiRhoMinusApproxRho,
      const bool                       shadowPotentialForce = false);

    /** @brief returns a copy of the configurational force on all global atoms.
     *
     *  computeAtomsForces must be called prior to this function call.
     *
     *  @return std::vector<double> flattened array of the configurational force on all atoms,
     *  the three force components on each atom being the leading dimension.
     * Units- Hartree/Bohr
     */
    std::vector<double>
    getAtomsForces();

    /** @brief prints the currently stored configurational forces on atoms and the Gaussian generator constant
     *  used to compute them.
     *
     *  @return void.
     */
    void
    printAtomsForces();

    /** @brief Update force generator Gaussian constant.
     *
     *  @return void.
     */
    // void updateGaussianConstant(const double newGaussianConstant);

    /** @brief computes the configurational stress on the domain corresponding to
     *  affine deformation of the periodic cell.
     *
     *  This function cannot be called for fully non-periodic calculations.
     *
     *  @return void.
     */
    void
    computeStress(
      const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                  kohnShamDFTEigenOperator,
      const dispersionCorrection &         dispersionCorr,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoOutValues,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,      
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &pseudoVLocAtomsElectro,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                      hessianRhoCoreAtoms,
      const dealii::AffineConstraints<double> &hangingPlusPBCConstraintsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro);

    /** @brief prints the currently stored configurational stress tensor.
     *
     *  @return void.
     */
    void
    printStress();

    /** @brief returns a copy of the current stress tensor value.
     *
     *  computeStress must be call prior to this function call.
     *
     *  @return dealii::Tensor<2,3,double>  second order stress Tensor in Hartree/Bohr^3
     */
    dealii::Tensor<2, 3, double>
    getStress();

    /** @brief get the value of Gaussian generator parameter (d_gaussianConstant).
     * Gaussian generator: Gamma(r)= exp(-d_gaussianConstant*r^2).
     *
     */
    // double getGaussianGeneratorParameter() const;

  private:
    /** @brief Locates and stores the global dof indices of d_dofHandlerForce whose cooridinates match
     *  with the atomic positions.
     *
     *  @return void.
     */
    void
    locateAtomCoreNodesForce(const dealii::DoFHandler<3> &dofHandlerForce,
                             const dealii::IndexSet &locally_owned_dofsForce,
                             std::map<std::pair<unsigned int, unsigned int>,
                                      unsigned int> &atomsForceDofs);

    void
    createBinObjectsForce(
      const dealii::DoFHandler<3> &            dofHandler,
      const dealii::DoFHandler<3> &            dofHandlerForce,
      const dealii::AffineConstraints<double> &hangingPlusPBCConstraints,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManager,
      std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
        &cellsVselfBallsDofHandler,
      std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
        &cellsVselfBallsDofHandlerForce,
      std::vector<std::map<dealii::CellId, unsigned int>>
        &cellsVselfBallsClosestAtomIdDofHandler,
      std::map<unsigned int, unsigned int> &AtomIdBinIdLocalDofHandler,
      std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                           std::vector<unsigned int>>>
        &cellFacesVselfBallSurfacesDofHandler,
      std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                           std::vector<unsigned int>>>
        &cellFacesVselfBallSurfacesDofHandlerForce);

    void
    configForceLinFEInit(
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro);

    void
    configForceLinFEFinalize();

    void
    computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(
      const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                  kohnShamDFTEigenOperator,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoOutValues,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                                  hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
      const std::map<dealii::CellId, std::vector<double>>
        &                              shadowKSGradRhoMinValues,
      const distributedCPUVec<double> &phiRhoMinusApproxRho,
      const bool                       shadowPotentialForce = false);

    void
    computeConfigurationalForceEEshelbyEElectroPhiTot(
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &rhoTotalOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
      const distributedCPUVec<double> &                    phiRhoMinusApproxRho,
      const bool shadowPotentialForce = false);

    void
    computeConfigurationalForcePhiExtLinFE();

    void
    computeConfigurationalForceEselfLinFE(
      const dealii::DoFHandler<3> &                    dofHandlerElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const dealii::MatrixFree<3, double> &            matrixFreeDataElectro,
      const unsigned int smearedChargeQuadratureId);

    void
    computeConfigurationalForceEselfNoSurfaceLinFE();

    void
    computeConfigurationalForceTotalLinFE(
      const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                  kohnShamDFTEigenOperator,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoOutValues,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                                  hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
      const std::map<dealii::CellId, std::vector<double>>
        &                              shadowKSGradRhoMinValues,
      const distributedCPUVec<double> &phiRhoMinusApproxRho,
      const bool                       shadowPotentialForce = false);

    void
    FPSPLocalGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &                                  forceContributionFPSPLocalGammaAtoms,
      dealii::FEValues<3> &                feValues,
      dealii::FEFaceValues<3> &            feFaceValues,
      dealii::FEEvaluation<3,
                           1,
                           C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                           3> &            forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>> &rhoQuads,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradRhoQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtoms,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManager,
      const std::vector<std::map<dealii::CellId, unsigned int>>
        &cellsVselfBallsClosestAtomIdDofHandler);

    void
    FPhiTotSmearedChargesGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionSmearedChargesGammaAtoms,
      dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradPhiTotQuads,
      const std::vector<unsigned int> &nonTrivialAtomIdsMacroCell,
      const std::map<dealii::CellId, std::vector<int>> &bQuadAtomIdsAllAtoms,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &smearedbQuads);

    void
    FVselfSmearedChargesGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionSmearedChargesGammaAtoms,
      dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &                              gradVselfBinQuads,
      const std::vector<unsigned int> &nonTrivialAtomIdsMacroCell,
      const std::map<dealii::CellId, std::vector<int>> &bQuadAtomIdsAllAtoms,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &smearedbQuads);

    void
    FShadowLocalGammaAtomsElementalContributionElectronic(
      std::map<unsigned int, std::vector<double>>
        &forceContributionLocalGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &derVxcWithRhoTimesRhoDiffQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoAtomsQuadsSeparate,
      const dealii::AlignedVector<
        dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
        &der2ExcWithGradRhoOutQuads,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derVxcWithGradRhoOutQuads,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &shadowKSRhoMinMinusGradRhoQuads,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &shadowKSGradRhoMinMinusGradRhoQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &hessianRhoAtomsQuadsSeparate,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &        hessianRhoCoreAtoms,
      const bool isAtomicRhoSplitting = false,
      const bool isXCGGA              = false,
      const bool isNLCC               = false);

    void
    FShadowLocalGammaAtomsElementalContributionElectrostatic(
      std::map<unsigned int, std::vector<double>>
        &forceContributionLocalGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &gradPhiRhoMinusApproxRhoElectroQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &rhoAtomsQuadsSeparate);


    void
    FNonlinearCoreCorrectionGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>> &vxcQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms);


    void
    FNonlinearCoreCorrectionGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &derExcGradRho,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &hessianRhoCoreAtoms);

    void
    FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin0,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin1,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin0,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin1,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &        hessianRhoCoreAtoms,
      const bool isXCGGA = false);

    void
    distributeForceContributionFPSPLocalGammaAtoms(
      const std::map<unsigned int, std::vector<double>>
        &forceContributionFPSPLocalGammaAtoms,
      const std::map<std::pair<unsigned int, unsigned int>, unsigned int>
        &                                      atomsForceDofs,
      const dealii::AffineConstraints<double> &constraintsNoneForce,
      distributedCPUVec<double> &              configForceVectorLinFE);

    void
    accumulateForceContributionGammaAtomsFloating(
      const std::map<unsigned int, std::vector<double>>
        &                  forceContributionLocalGammaAtoms,
      std::vector<double> &accumForcesVector);


    void
    FnlGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &                                  forceContributionFnlGammaAtoms,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      dealii::FEEvaluation<3,
                           1,
                           C_num1DQuadNLPSP<FEOrder>() *
                             C_numCopies1DQuadNLPSP(),
                           3> &            forceEvalNLP,
      const unsigned int                   cell,
      const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
#ifdef USE_COMPLEX
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
      const std::vector<dataTypes::number> &zetaDeltaVQuadsFlattened,
      const std::vector<dataTypes::number> &
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);


    void FnlGammaxElementalContribution(
      dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &FVectQuads,
      const dealii::MatrixFree<3, double> &                     matrixFreeData,
      const unsigned int                                        numQuadPoints,
      const unsigned int                                        cell,
      const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
      const std::vector<dataTypes::number> &        zetaDeltaVQuadsFlattened,
      const std::vector<dataTypes::number> &
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);

    void
    distributeForceContributionFnlGammaAtoms(
      const std::map<unsigned int, std::vector<double>>
        &forceContributionFnlGammaAtoms);

    void stressEnlElementalContribution(
      dealii::Tensor<2, 3, double> &                stressContribution,
      const dealii::MatrixFree<3, double> &         matrixFreeData,
      const unsigned int                            numQuadPoints,
      const std::vector<double> &                   jxwQuadsSubCells,
      const unsigned int                            cell,
      const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
      const std::vector<dataTypes::number> &zetalmDeltaVlProductDistImageAtoms,
#ifdef USE_COMPLEX
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened,
      const bool isSpinPolarized);

    void
    computeAtomsForcesGaussianGenerator(
      bool allowGaussianOverlapOnAtoms = false);

    void
    computeFloatingAtomsForces();

    void
    computeStressEself(
      const dealii::DoFHandler<3> &                    dofHandlerElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
      const dealii::MatrixFree<3, double> &            matrixFreeDataElectro,
      const unsigned int smearedChargeQuadratureId);

    void
    computeStressEEshelbyEPSPEnlEk(
      const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                  kohnShamDFTEigenOperator,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoOutValues,
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &pseudoVLocAtomsElectro,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              hessianRhoCoreAtoms,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro);

    void
    computeStressEEshelbyEElectroPhiTot(
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &rhoTotalOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesElectro,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesElectroLpsp,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro);

    void addEPSPStressContribution(
      dealii::FEValues<3> &                feValues,
      dealii::FEFaceValues<3> &            feFaceValues,
      dealii::FEEvaluation<3,
                           1,
                           C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                           3> &            forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>> &rhoQuads,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradRhoQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtoms,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManager,
      const std::vector<std::map<dealii::CellId, unsigned int>>
        &cellsVselfBallsClosestAtomIdDofHandler);

    void addENonlinearCoreCorrectionStressContribution(
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>> &vxcQuads,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &derExcGradRho,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &hessianRhoCoreAtoms);

    void addENonlinearCoreCorrectionStressContributionSpinPolarized(
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin0,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin1,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin0,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin1,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &        hessianRhoCoreAtoms,
      const bool isXCGGA = false);

    void addEPhiTotSmearedStressContribution(
      dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradPhiTotQuads,
      const std::vector<unsigned int> &nonTrivialAtomImageIdsMacroCell,
      const std::map<dealii::CellId, std::vector<int>>
        &bQuadAtomIdsAllAtomsImages,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &smearedbQuads);

    void addEVselfSmearedStressContribution(
      dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradVselfQuads,
      const std::vector<unsigned int> &nonTrivialAtomImageIdsMacroCell,
      const std::map<dealii::CellId, std::vector<int>>
        &bQuadAtomIdsAllAtomsImages,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &smearedbQuads);

    void
    computeElementalNonLocalPseudoOVDataForce();


    /// Parallel distributed vector field which stores the configurational force
    /// for each fem node corresponding to linear shape function generator (see
    /// equations 52-53 in
    /// (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)). This vector
    /// doesn't contain contribution from terms which have sums over k points.
    distributedCPUVec<double> d_configForceVectorLinFE;

    /// Parallel distributed vector field which stores the configurational force
    /// for each fem node corresponding to linear shape function generator (see
    /// equations 52-53 in
    /// (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)). This vector only
    /// containts contribution from the electrostatic part.
    distributedCPUVec<double> d_configForceVectorLinFEElectro;

#ifdef USE_COMPLEX
    /// Parallel distributed vector field which stores the configurational force
    /// for each fem node corresponding to linear shape function generator (see
    /// equations 52-53 in
    /// (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)). This vector only
    /// contains contribution from terms which have sums over k points.
    distributedCPUVec<double> d_configForceVectorLinFEKPoints;
#endif


    std::vector<double> d_forceAtomsFloating;

#ifdef USE_COMPLEX
    std::vector<double> d_forceAtomsFloatingKPoints;
#endif



    /// Gaussian generator constant. Gaussian generator: Gamma(r)=
    /// exp(-d_gaussianConstant*r^2)
    /// FIXME: Until the hanging nodes surface integral issue is fixed use a
    /// value >=4.0
    // double d_gaussianConstant;

    /// Storage for configurational force on all global atoms.
    std::vector<double> d_globalAtomsForces;

    /// Storage for configurational stress tensor
    dealii::Tensor<2, 3, double> d_stress;



    /* Part of the stress tensor which is summed over k points.
     * It is a temporary data structure required for stress evaluation
     * (d_stress) when parallization over k points is on.
     */
    dealii::Tensor<2, 3, double> d_stressKPoints;

    /* Dont use true except for debugging forces only without mesh movement, as
     * gaussian ovelap on atoms for move mesh is by default set to false
     */
    const bool d_allowGaussianOverlapOnAtoms = false;

    /// pointer to dft class
    dftClass<FEOrder, FEOrderElectro> *dftPtr;

    /// Finite element object for configurational force computation. Linear
    /// finite elements with three force field components are used.
    dealii::FESystem<3> FEForce;

    /* DofHandler on which we define the configurational force field. Each
     * geometric fem node has three dofs corresponding the the three force
     * components. The generator for the configurational force on the fem node
     * is the linear shape function attached to it. This DofHandler is based on
     * the same triangulation on which we solve the dft problem.
     */
    dealii::DoFHandler<3> d_dofHandlerForce;

    /* DofHandler on which we define the configurational force field from
     * electrostatic part (without psp). Each geometric fem node has three dofs
     * corresponding the the three force components. The generator for the
     * configurational force on the fem node is the linear shape function
     * attached to it. This DofHandler is based on the same triangulation on
     * which we solve the dft problem.
     */
    dealii::DoFHandler<3> d_dofHandlerForceElectro;

    /// Index of the d_dofHandlerForce in the MatrixFree object stored in
    /// dftClass. This is required to correctly use FEEvaluation class.
    unsigned int d_forceDofHandlerIndex;

    /// Index of the d_dofHandlerForceElectro in the MatrixFree object stored in
    /// dftClass. This is required to correctly use FEEvaluation class.
    unsigned int d_forceDofHandlerIndexElectro;

    /// dealii::IndexSet of locally owned dofs of in d_dofHandlerForce the
    /// current processor
    dealii::IndexSet d_locally_owned_dofsForce;

    /// dealii::IndexSet of locally owned dofs of in d_dofHandlerForceElectro
    /// the current processor
    dealii::IndexSet d_locally_owned_dofsForceElectro;

    /// dealii::IndexSet of locally relevant dofs of in d_dofHandlerForce the
    /// current processor
    dealii::IndexSet d_locally_relevant_dofsForce;

    /// dealii::IndexSet of locally relevant dofs of in d_dofHandlerForceElectro
    /// the current processor
    dealii::IndexSet d_locally_relevant_dofsForceElectro;

    /// Constraint matrix for hanging node and periodic constaints on
    /// d_dofHandlerForce.
    dealii::AffineConstraints<double> d_constraintsNoneForce;

    /// Constraint matrix for hanging node and periodic constaints on
    /// d_dofHandlerForceElectro.
    dealii::AffineConstraints<double> d_constraintsNoneForceElectro;

    /// Internal data: map < <atomId,force component>, globaldof in
    /// d_dofHandlerForce>
    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_atomsForceDofs;

    /// Internal data: map < <atomId,force component>, globaldof in
    /// d_dofHandlerForceElectro>
    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_atomsForceDofsElectro;

    /// Internal data: stores cell iterators of all cells in
    /// dftPtr->d_dofHandler which are part of the vself ball. Outer vector is
    /// over vself bins.
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      d_cellsVselfBallsDofHandler;

    /// Internal data: stores cell iterators of all cells in d_dofHandlerForce
    /// which are part of the vself ball. Outer vector is over vself bins.
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      d_cellsVselfBallsDofHandlerForce;

    /// Internal data: stores map of vself ball cell Id  to the closest atom Id
    /// of that cell. Outer vector over vself bins.
    std::vector<std::map<dealii::CellId, unsigned int>>
      d_cellsVselfBallsClosestAtomIdDofHandler;

    /// Internal data: stores the map of atom Id (only in the local processor)
    /// to the vself bin Id.
    std::map<unsigned int, unsigned int> d_AtomIdBinIdLocalDofHandler;

    /* Internal data: stores the face ids of dftPtr->d_dofHandler (single
     * component field) on which to evaluate the vself ball surface integral in
     * the configurational force expression. Outer vector is over the vself
     * bins. Inner map is between the cell iterator and the vector of face ids
     * to integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>>
      d_cellFacesVselfBallSurfacesDofHandler;

    /* Internal data: stores the face ids of d_dofHandlerForce (three component
     * field) on which to evaluate the vself ball surface integral in the
     * configurational force expression. Outer vector is over the vself bins.
     * Inner map is between the cell iterator and the vector of face ids to
     * integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>>
      d_cellFacesVselfBallSurfacesDofHandlerForce;

    /// Internal data: stores cell iterators of all cells in
    /// dftPtr->d_dofHandler which are part of the vself ball. Outer vector is
    /// over vself bins.
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      d_cellsVselfBallsDofHandlerElectro;

    /// Internal data: stores cell iterators of all cells in d_dofHandlerForce
    /// which are part of the vself ball. Outer vector is over vself bins.
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      d_cellsVselfBallsDofHandlerForceElectro;

    /// Internal data: stores map of vself ball cell Id  to the closest atom Id
    /// of that cell. Outer vector over vself bins.
    std::vector<std::map<dealii::CellId, unsigned int>>
      d_cellsVselfBallsClosestAtomIdDofHandlerElectro;

    /// Internal data: stores the map of atom Id (only in the local processor)
    /// to the vself bin Id.
    std::map<unsigned int, unsigned int> d_AtomIdBinIdLocalDofHandlerElectro;

    /* Internal data: stores the face ids of dftPtr->d_dofHandler (single
     * component field) on which to evaluate the vself ball surface integral in
     * the configurational force expression. Outer vector is over the vself
     * bins. Inner map is between the cell iterator and the vector of face ids
     * to integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>>
      d_cellFacesVselfBallSurfacesDofHandlerElectro;

    /* Internal data: stores the face ids of d_dofHandlerForce (three component
     * field) on which to evaluate the vself ball surface integral in the
     * configurational force expression. Outer vector is over the vself bins.
     * Inner map is between the cell iterator and the vector of face ids to
     * integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>>
      d_cellFacesVselfBallSurfacesDofHandlerForceElectro;

    std::map<dealii::CellId, dealii::DoFHandler<3>::active_cell_iterator>
      d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro;

    std::vector<distributedCPUVec<double>> d_gaussianWeightsVecAtoms;

    /// map from cell id to set of non local atom ids (local numbering)
    // std::map<dealii::CellId,std::set<unsigned int>>
    // d_cellIdToNonlocalAtomIdsLocalCompactSupportMap;

    /// domain decomposition mpi_communicator
    const MPI_Comm d_mpiCommParent;

    /// domain decomposition mpi_communicator
    const MPI_Comm mpi_communicator;

    const dftParameters &d_dftParams;

    /// number of mpi processes in the current pool
    const unsigned int n_mpi_processes;

    /// current mpi process id in the current pool
    const unsigned int this_mpi_process;

    /// conditional stream object to enable printing only on root processor
    /// across all pools
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif
