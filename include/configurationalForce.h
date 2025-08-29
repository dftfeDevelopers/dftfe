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


#ifndef configurationalForce_H_
#define configurationalForce_H_

#include <dftd.h>
#include <oncvClass.h>
#include <AtomicCenteredNonLocalOperator.h>
#include <FEBasisOperations.h>
#include <BLASWrapper.h>
#include <vselfBinsManager.h>
#include <groupSymmetry.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class configurationalForceClass
  {
  public:
    configurationalForceClass(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<double, double, memorySpace>>
        basisOperationsPtrElectro,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrElectroHost,
      std::shared_ptr<
        dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                               pseudopotentialClassPtr,
      std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      const dftfe::uInt                        densityQuadratureId,
      const dftfe::uInt                        densityQuadratureIdElectro,
      const dftfe::uInt                        lpspQuadratureId,
      const dftfe::uInt                        lpspQuadratureIdElectro,
      const dftfe::uInt                        smearedChargeQuadratureIdElectro,
      const MPI_Comm                          &mpi_comm_parent,
      const MPI_Comm                          &mpi_comm_domain,
      const MPI_Comm                          &interpoolcomm,
      const MPI_Comm                          &interBandGroupComm,
      const dftParameters                     &dftParams);

    void
    computeForceAndStress(
      const dftfe::uInt                         &numEigenValues,
      const std::vector<double>                 &kPointCoords,
      const std::vector<double>                 &kPointWeights,
      const std::vector<std::vector<double>>    &domainBoundingVectors,
      const double                               domainVolume,
      const std::shared_ptr<groupSymmetryClass> &groupSymmetryPtr,
      const dispersionCorrection                &dispersionCorr,
      const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                             &eigenVectors,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<double>              &imageCharges,
      const std::vector<std::vector<double>> &imagePositions,
      const distributedCPUVec<double>        &phiTotRhoOutValues,
      const distributedCPUVec<double>        &rhoOutNodalValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCOutRepresentationPtr,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                                          &hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocValues,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                  &pseudoVLocAtoms,
      const dealii::DoFHandler<3> &dofHandlerRhoNodal,
      const vselfBinsManager      &vselfBinsManager,
      const std::vector<distributedCPUVec<double>>
                        &vselfFieldGateauxDerStrainFDBins,
      const dftfe::uInt &binsStartDofHandlerIndexElectro,
      const std::map<dealii::CellId, std::vector<dftfe::Int>>
        &bQuadAtomIdsAllAtoms,
      const std::map<dealii::CellId, std::vector<dftfe::Int>>
        &bQuadAtomIdsAllAtomsImages,
      const std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
      const std::vector<double> &gaussianConstantsForce,
      const std::vector<double> &generatorFlatTopWidths,
      const bool                 floatingNuclearCharges,
      const bool                 computeForce,
      const bool                 computeStress);

    void
    printStress();

    void
    printAtomsForces();

    std::vector<double> &
    getAtomsForces();

    dealii::Tensor<2, 3, double> &
    getStress();


  private:
    void
    computeWfcContribNloc(
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
                              nonLocalOperator,
      const CouplingStructure couplingtype,
      const std::vector<
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> *>
                                              &couplingMatrixPtrs,
      const std::map<dftfe::uInt, dftfe::uInt> nonlocalAtomIdToGlobalIdMap,
      const dftfe::uInt                       &numEigenValues,
      const std::vector<double>               &kPointCoords,
      const std::vector<double>               &kPointWeights,
      const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                             &eigenVectors,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const bool                              floatingNuclearCharges,
      const bool                              computeForce,
      const bool                              computeStress);

    void
    computeWfcContribNlocAtomOnNode(
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
                              nonLocalOperator,
      const CouplingStructure couplingtype,
      const std::vector<
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> *>
                                              &couplingMatrixPtrs,
      const std::map<dftfe::uInt, dftfe::uInt> nonlocalAtomIdToGlobalIdMap,
      const dftfe::uInt                       &numEigenValues,
      const std::vector<double>               &kPointCoords,
      const std::vector<double>               &kPointWeights,
      const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                             &eigenVectors,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const bool                              floatingNuclearCharges,
      const bool                              computeForce,
      const bool                              computeStress);

    void
    computeWfcContribLocal(
      const dftfe::uInt         &numEigenValues,
      const std::vector<double> &kPointCoords,
      const std::vector<double> &kPointWeights,
      const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                             &eigenVectors,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const bool                              floatingNuclearCharges,
      const bool                              computeForce,
      const bool                              computeStress);

    void
    computeElectroContribEshelby(
      const distributedCPUVec<double> &phiTotRhoOutValues,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                &rhooutValues,
      const bool floatingNuclearCharges,
      const bool computeForce,
      const bool computeStress);

    void
    computeESelfContribEshelby(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<double>              &imageCharges,
      const std::vector<std::vector<double>> &imagePositions,
      const vselfBinsManager                 &vselfBinsManager,
      const bool                              floatingNuclearCharges,
      const bool                              computeForce,
      const bool                              computeStress);

    void
    computeSmearedContribAll(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imagePositions,
      const vselfBinsManager                 &vselfBinsManager,
      const dftfe::uInt                      &binsStartDofHandlerIndexElectro,
      const distributedCPUVec<double>        &phiTotRhoOutValues,
      const std::map<dealii::CellId, std::vector<dftfe::Int>>
        &bQuadAtomIdsAllAtoms,
      const std::map<dealii::CellId, std::vector<dftfe::Int>>
        &bQuadAtomIdsAllAtomsImages,
      const std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
      const bool floatingNuclearCharges,
      const bool computeForce,
      const bool computeStress);

    void
    computeLPSPContribAll(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<double>              &imageCharges,
      const std::vector<std::vector<double>> &imagePositions,
      const distributedCPUVec<double>        &rhoOutNodalValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocValues,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                  &pseudoVLocAtoms,
      const dealii::DoFHandler<3> &dofHandlerRhoNodal,
      const vselfBinsManager      &vselfBinsManager,
      const std::vector<distributedCPUVec<double>>
                &vselfFieldGateauxDerStrainFDBins,
      const bool floatingNuclearCharges,
      const bool computeForce,
      const bool computeStress);

    void
    computeXCContribAll(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauOutValues,
      const std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCOutRepresentationPtr,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                &hessianRhoCoreAtoms,
      const bool floatingNuclearCharges,
      const bool computeForce,
      const bool computeStress);


    void
    createBinObjectsForce(
      const dealii::DoFHandler<3> &dofHandlerRhoNodal,
      const vselfBinsManager      &vselfBinsManager,
      std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
        &cellsVselfBallsDofHandler,
      std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
        &cellsVselfBallsDofHandlerForce,
      std::vector<std::map<dealii::CellId, dftfe::uInt>>
        &cellsVselfBallsClosestAtomIdDofHandler,
      std::map<dftfe::uInt, dftfe::uInt> &AtomIdBinIdLocalDofHandler,
      std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                           std::vector<dftfe::uInt>>>
        &cellFacesVselfBallSurfacesDofHandler,
      std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                           std::vector<dftfe::uInt>>>
        &cellFacesVselfBallSurfacesDofHandlerForce);

    void
    computeAtomsForcesGaussianGenerator(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<double>              &gaussianConstantsForce,
      const std::vector<double>              &generatorFlatTopWidths,
      const distributedCPUVec<double>        &configForceVectorLinFE,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &forceContrib);
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperPtrHost;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperationsPtr;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>>
      d_basisOperationsPtrElectro;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrElectroHost;
    std::shared_ptr<
      dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
      d_pseudopotentialClassPtr;

    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_forceTotal;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_stressTotal;

    std::vector<double>          d_forceVector;
    dealii::Tensor<2, 3, double> d_stressTensor;

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             d_mpiCommDomain;
    const MPI_Comm             d_mpiCommInterPool;
    const MPI_Comm             d_mpiCommInterBandGroup;
    const dftParameters       &d_dftParams;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;

    const dftfe::uInt d_densityQuadratureId;
    const dftfe::uInt d_densityQuadratureIdElectro;
    const dftfe::uInt d_lpspQuadratureId;
    const dftfe::uInt d_lpspQuadratureIdElectro;
    const dftfe::uInt d_smearedChargeQuadratureIdElectro;


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
    std::vector<std::map<dealii::CellId, dftfe::uInt>>
      d_cellsVselfBallsClosestAtomIdDofHandlerElectro;

    /// Internal data: stores the map of atom Id (only in the local processor)
    /// to the vself bin Id.
    std::map<dftfe::uInt, dftfe::uInt> d_AtomIdBinIdLocalDofHandlerElectro;

    /* Internal data: stores the face ids of dftPtr->d_dofHandler (single
     * component field) on which to evaluate the vself ball surface integral in
     * the configurational force expression. Outer vector is over the vself
     * bins. Inner map is between the cell iterator and the vector of face ids
     * to integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      d_cellFacesVselfBallSurfacesDofHandlerElectro;

    /* Internal data: stores the face ids of d_dofHandlerForce (three component
     * field) on which to evaluate the vself ball surface integral in the
     * configurational force expression. Outer vector is over the vself bins.
     * Inner map is between the cell iterator and the vector of face ids to
     * integrate on for that cell iterator.
     */
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      d_cellFacesVselfBallSurfacesDofHandlerForceElectro;

    std::map<dealii::CellId, dealii::DoFHandler<3>::active_cell_iterator>
      d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro;

    /// Finite element object for configurational force computation. Linear
    /// finite elements with three force field components are used.
    dealii::FESystem<3>               FEForce;
    dealii::AffineConstraints<double> d_affineConstraintsForce;
    dealii::DoFHandler<3>             d_dofHandlerForce;
    dealii::IndexSet                  d_locally_owned_dofsForce;
    distributedCPUVec<double>         d_configForceContribsLinFE;

    dealii::TimerOutput computing_timer;
  };
  void
  computeWavefuncEshelbyContributionLocal(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    const double                              kcoordx,
    const double                              kcoordy,
    const double                              kcoordz,
    double                                   *partialOccupVec,
    double                                   *eigenValuesVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double                                   *eshelbyContributions,
    double                                   *eshelbyTensor,
    const bool                                floatingNuclearCharges,
    const bool                                computeForce,
    const bool                                computeStress);

} // namespace dftfe
#endif
