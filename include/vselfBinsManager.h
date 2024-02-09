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

#include <headers.h>
#include "constraintMatrixInfo.h"
#include "dftParameters.h"
#include "FEBasisOperations.h"
#if defined(DFTFE_WITH_DEVICE)
#  include <operatorDevice.h>
#endif

#ifndef vselfBinsManager_H_
#  define vselfBinsManager_H_

namespace dftfe
{
  /**
   * @brief Categorizes atoms into bins for efficient solution of nuclear electrostatic self-potential.
   * template parameter FEOrderElectro is the finite element polynomial order.
   *
   * @author Sambit Das, Phani Motamarri
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class vselfBinsManager
  {
  public:
    /**
     * @brief Constructor
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain domain decomposition mpi communicator
     */
    vselfBinsManager(const MPI_Comm &     mpi_comm_parent,
                     const MPI_Comm &     mpi_comm_domain,
                     const MPI_Comm &     mpi_intercomm_kpts,
                     const dftParameters &dftParams);


    /**
     * @brief Categorize atoms into bins based on self-potential ball radius
     * around each atom such that no two atoms in each bin has overlapping
     * balls.
     *
     * @param[out] constraintsVector constraintsVector to which the vself bins
     * solve constraint matrices will be pushed back
     * @param[in] dofHandler DofHandler object
     * @param[in] constraintMatrix dealii::AffineConstraints<double> which was
     * used for the total electrostatics solve
     * @param[in] atomLocations global atom locations and charge values data
     * @param[in] imagePositions image atoms positions data
     * @param[in] imageIds image atoms Ids data
     * @param[in] imageCharges image atoms charge values data
     * @param[in] radiusAtomBall self-potential ball radius
     */
    void
    createAtomBins(
      std::vector<const dealii::AffineConstraints<double> *> &constraintsVector,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
      const dealii::DoFHandler<3> &            dofHandler,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const std::vector<std::vector<double>> & atomLocations,
      const std::vector<std::vector<double>> & imagePositions,
      const std::vector<int> &                 imageIds,
      const std::vector<double> &              imageCharges,
      const double                             radiusAtomBall);

    /**
     * @brief Categorize atoms into bins based on self-potential ball radius
     * around each atom such that no two atoms in each bin has overlapping
     * balls.
     *
     * @param[out] constraintsVector constraintsVector to which the vself bins
     * solve constraint matrices will be pushed back
     * @param[in] dofHandler DofHandler object
     * @param[in] constraintMatrix dealii::AffineConstraints<double> which was
     * used for the total electrostatics solve
     * @param[in] atomLocations global atom locations and charge values data
     * @param[in] imagePositions image atoms positions data
     * @param[in] imageIds image atoms Ids data
     * @param[in] imageCharges image atoms charge values data
     */
    void
    updateBinsBc(
      std::vector<const dealii::AffineConstraints<double> *> &constraintsVector,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
      const dealii::DoFHandler<3> &            dofHandler,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const std::vector<std::vector<double>> & atomLocations,
      const std::vector<std::vector<double>> & imagePositions,
      const std::vector<int> &                 imageIds,
      const std::vector<double> &              imageCharges,
      const bool vselfPerturbationUpdateForStress = false);


    /**
     * @brief Solve nuclear electrostatic self-potential of atoms in each bin one-by-one
     *
     * @param[in] matrix_free_data MatrixFree object
     * @param[in] offset MatrixFree object starting offset for vself bins solve
     * @param[out] phiExt sum of the self-potential fields of all atoms and
     * image atoms
     * @param[in] phiExtConstraintMatrix constraintMatrix corresponding to
     * phiExt
     * @param[in] imagePositions image atoms positions data
     * @param[in] imageIds image atoms Ids data
     * @param[in] imageCharges image atoms charge values data	   *
     * @param[out] localVselfs peak self-potential values of atoms in the local
     * processor
     */
    void
    solveVselfInBins(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                                      basisOperationsPtr,
      const unsigned int                       offset,
      const unsigned int                       matrixFreeQuadratureIdAX,
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const std::vector<std::vector<double>> & imagePositions,
      const std::vector<int> &                 imageIds,
      const std::vector<double> &              imageCharges,
      std::vector<std::vector<double>> &       localVselfs,
      std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
      std::map<dealii::CellId, std::vector<int>> &   bQuadAtomIdsAllAtoms,
      std::map<dealii::CellId, std::vector<int>> &   bQuadAtomIdsAllAtomsImages,
      std::map<dealii::CellId, std::vector<unsigned int>>
        &bCellNonTrivialAtomIds,
      std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
        &bCellNonTrivialAtomIdsBins,
      std::map<dealii::CellId, std::vector<unsigned int>>
        &bCellNonTrivialAtomImageIds,
      std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
        &                        bCellNonTrivialAtomImageIdsBins,
      const std::vector<double> &smearingWidths,
      std::vector<double> &      smearedChargeScaling,
      const unsigned int         smearedChargeQuadratureId,
      const bool                 useSmearedCharges        = false,
      const bool                 isVselfPerturbationSolve = false);

#  ifdef DFTFE_WITH_DEVICE
    /**
     * @brief Solve nuclear electrostatic self-potential of atoms in each bin one-by-one
     *
     * @param[in] matrix_free_data MatrixFree object
     * @param[in] offset MatrixFree object starting offset for vself bins solve
     * @param[out] phiExt sum of the self-potential fields of all atoms and
     * image atoms
     * @param[in] phiExtConstraintMatrix constraintMatrix corresponding to
     * phiExt
     * @param[in] imagePositions image atoms positions data
     * @param[in] imageIds image atoms Ids data
     * @param[in] imageCharges image atoms charge values data	   *
     * @param[out] localVselfs peak self-potential values of atoms in the local
     * processor
     */
    void
    solveVselfInBinsDevice(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int mfBaseDofHandlerIndex,
      const unsigned int matrixFreeQuadratureIdAX,
      const unsigned int offset,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellGradNIGradNJIntergralDevice,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                      BLASWrapperPtr,
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const std::vector<std::vector<double>> & imagePositions,
      const std::vector<int> &                 imageIds,
      const std::vector<double> &              imageCharges,
      std::vector<std::vector<double>> &       localVselfs,
      std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
      std::map<dealii::CellId, std::vector<int>> &   bQuadAtomIdsAllAtoms,
      std::map<dealii::CellId, std::vector<int>> &   bQuadAtomIdsAllAtomsImages,
      std::map<dealii::CellId, std::vector<unsigned int>>
        &bCellNonTrivialAtomIds,
      std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
        &bCellNonTrivialAtomIdsBins,
      std::map<dealii::CellId, std::vector<unsigned int>>
        &bCellNonTrivialAtomImageIds,
      std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
        &                        bCellNonTrivialAtomImageIdsBins,
      const std::vector<double> &smearingWidths,
      std::vector<double> &      smearedChargeScaling,
      const unsigned int         smearedChargeQuadratureId,
      const bool                 useSmearedCharges        = false,
      const bool                 isVselfPerturbationSolve = false);


#  endif

    /**
     * @brief Solve nuclear electrostatic self-potential of atoms in each bin in a perturbed domain (used for cell stress calculation)
     *
     */
    void
    solveVselfInBinsPerturbedDomain(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int mfBaseDofHandlerIndex,
      const unsigned int matrixFreeQuadratureIdAX,
      const unsigned int offset,
#  ifdef DFTFE_WITH_DEVICE
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellGradNIGradNJIntergralDevice,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
#  endif
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const std::vector<std::vector<double>> & imagePositions,
      const std::vector<int> &                 imageIds,
      const std::vector<double> &              imageCharges,
      const std::vector<double> &              smearingWidths,
      const unsigned int                       smearedChargeQuadratureId,
      const bool                               useSmearedCharges = false);

    /// get const reference map of binIds and atomIds
    const std::map<int, std::set<int>> &
    getAtomIdsBins() const;

    /// get const reference map of binIds and atomIds
    const std::map<int, std::set<int>> &
    getAtomImageIdsBins() const;

    /// get const reference to map of global dof index and vself solve boundary
    /// flag in each bin
    const std::vector<std::map<dealii::types::global_dof_index, int>> &
    getBoundaryFlagsBins() const;

    /// get const reference to map of global dof index and vself solve boundary
    /// flag in each bin
    const std::vector<std::map<dealii::types::global_dof_index, int>> &
    getBoundaryFlagsBinsOnlyChargeId() const;

    /// get const reference to map of global dof index and vself field initial
    /// value in each bin
    const std::vector<std::map<dealii::types::global_dof_index, int>> &
    getClosestAtomIdsBins() const;

    /// get const reference to map of global dof index and vself field initial
    /// value in each bin
    const std::vector<
      std::map<dealii::types::global_dof_index, dealii::Point<3>>> &
    getClosestAtomLocationsBins() const;

    /// get const reference to solved vself fields
    const std::vector<distributedCPUVec<double>> &
    getVselfFieldBins() const;

    /// get const reference to del{vself}/del{R_i} fields
    const std::vector<distributedCPUVec<double>> &
    getVselfFieldDerRBins() const;

    /// perturbation of vself solution field to be used to evaluate the
    /// Gateaux derivative of vself field with respect to affine strain
    /// components using central finite difference
    const std::vector<distributedCPUVec<double>> &
    getPerturbedVselfFieldBins() const;

    /// get const reference to d_atomIdBinIdMapLocalAllImages
    const std::map<unsigned int, unsigned int> &
    getAtomIdBinIdMapLocalAllImages() const;

    /// get stored adaptive ball radius
    double
    getStoredAdaptiveBallRadius() const;


  private:
    /**
     * @brief locate underlying fem nodes for atoms in bins.
     *
     */
    void
    locateAtomsInBins(const dealii::DoFHandler<3> &dofHandler);

    /**
     * @brief sanity check for Dirichlet boundary conditions on the vself balls in each bin
     *
     */
    void
    createAtomBinsSanityCheck(
      const dealii::DoFHandler<3> &            dofHandler,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints);

    /// storage for input atomLocations argument in createAtomBins function
    std::vector<std::vector<double>> d_atomLocations;

    /// storage for optimized constraints handling object
    dftUtils::constraintMatrixInfo d_constraintsOnlyHangingInfo;

    /// vector of constraint matrices for vself bins
    std::vector<dealii::AffineConstraints<double>> d_vselfBinConstraintMatrices;

    /// map of binIds and atomIds
    std::map<int, std::set<int>> d_bins;

    /// map of binIds and atomIds and imageIds
    std::map<int, std::set<int>> d_binsImages;

    /// map of global dof index and vself solve boundary flag (chargeId or
    //  imageId+numberGlobalCharges) in each bin
    std::vector<std::map<dealii::types::global_dof_index, int>> d_boundaryFlag;

    /// map of global dof index and vself solve boundary flag (only chargeId)in
    /// each bin
    std::vector<std::map<dealii::types::global_dof_index, int>>
      d_boundaryFlagOnlyChargeId;

    /// map of global dof index to location of closest charge
    std::vector<std::map<dealii::types::global_dof_index, dealii::Point<3>>>
      d_dofClosestChargeLocationMap;

    /// map of global dof index and vself field initial value in each bin
    std::vector<std::map<dealii::types::global_dof_index, double>>
      d_vselfBinField;

    /// map of global dof index and vself field initial value in each bin
    std::vector<std::map<dealii::types::global_dof_index, int>>
      d_closestAtomBin;

    /// Internal data: stores the map of atom Id (only in the local processor)
    /// to the vself bin Id. Populated in solve vself in Bins
    std::map<unsigned int, unsigned int> d_atomIdBinIdMapLocalAllImages;

    /// solved vself solution field for each bin
    std::vector<distributedCPUVec<double>> d_vselfFieldBins;

    /// solved del{vself}/del{R_i} solution field for each bin
    std::vector<distributedCPUVec<double>> d_vselfFieldDerRBins;

    /// perturbation of vself solution field to be used to evaluate the
    /// Gateaux derivative of vself field with respect to affine strain
    /// components using central finite difference
    std::vector<distributedCPUVec<double>> d_vselfFieldPerturbedBins;

    // std::vector<double> d_inhomoIdsColoredVecFlattened;

    /// Map of locally relevant global dof index and the atomic charge in each
    /// bin
    std::vector<std::map<dealii::types::global_dof_index, double>> d_atomsInBin;

    /// Vself ball radius. This is stored after the first call to createAtomBins
    /// and reused for subsequent calls
    double d_storedAdaptiveBallRadius;

    const dftParameters &d_dftParams;

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const MPI_Comm             d_mpiInterCommKpts;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // vselfBinsManager_H_
