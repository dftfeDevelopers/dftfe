// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian
//

#include "functionalTest.h"
#include "dftParameters.h"
#include "triangulationManager.h"
#include "TransferBetweenMeshesIncompatiblePartitioning.h"
#include "MPIPatternP2P.h"
#include "vectorUtilities.h"

namespace functionalTest
{
  namespace
  {
    double
    value(double x, double y, double z, unsigned int index)
    {
      double val = 1;
      val        = x * x + y * y + z * z;
      return val;
    }

    double
    real_part(double &a)
    {
      return a;
    }

    double
    imaginary_part(double &a)
    {
      return 0.0;
    }

    double
    real_part(std::complex<double> &a)
    {
      return std::real(a);
    }

    double
    imaginary_part(std::complex<double> &a)
    {
      return std::imag(a);
    }

    double
    conj_compl(double &a)
    {
      return a;
    }

    std::complex<double>
    conj_compl(std::complex<double> &a)
    {
      return std::conj(a);
    }
  } // namespace

  void
  testAccumulateInsert(const MPI_Comm &mpiComm)
  {
    // Works with just two processors.
    // if you have more. you are screwed.

    std::pair<dftfe::global_size_type, dftfe::global_size_type> localRange;
    std::vector<dftfe::global_size_type>                        ghostRange;

    dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      vecData;

    dftfe::size_type thisRankId =
      dealii::Utilities::MPI::this_mpi_process(mpiComm);

    dftfe::size_type numMPIRank =
      dealii::Utilities::MPI::n_mpi_processes(mpiComm);

    if (numMPIRank != 2)
      {
        std::cout
          << " Errrorrrr num procs is not equal to 2. This function is specifically written for 2 procs \n";
      }

    if (thisRankId == 0)
      {
        localRange =
          std::make_pair<dftfe::global_size_type, dftfe::global_size_type>(0,
                                                                           10);
        ghostRange.resize(3, 0);
        ghostRange[0] = 11;
        ghostRange[1] = 14;
        ghostRange[2] = 16;
      }

    if (thisRankId == 1)
      {
        localRange =
          std::make_pair<dftfe::global_size_type, dftfe::global_size_type>(10,
                                                                           17);
        ghostRange.resize(0, 0);
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);


    std::shared_ptr<
      dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      mpiP2PObjPtr = std::make_shared<
        dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>(
        localRange, ghostRange, mpiComm);

    dftfe::utils::mpi::MPICommunicatorP2P<dftfe::dataTypes::number,
                                          dftfe::utils::MemorySpace::HOST>
      mpiCommP2PObj(mpiP2PObjPtr, 1);


    if (thisRankId == 1)
      {
        vecData.resize(7, 0.0);

        for (dftfe::size_type iPoint = 0; iPoint < 7; iPoint++)
          {
            vecData[iPoint] = 2.0;
          }
      }

    if (thisRankId == 0)
      {
        vecData.resize(13, 0.0);

        for (dftfe::size_type iPoint = 0; iPoint < 10; iPoint++)
          {
            vecData[iPoint] = iPoint;
          }
        vecData[10] = 17;
        vecData[11] = 24;
        vecData[12] = 12;
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);


    if (thisRankId == 0)
      {
        std::cout << " This rank = 0 \n";
        std::cout << " Vec data = ";
        for (dftfe::size_type iPoint = 0; iPoint < 10; iPoint++)
          std::cout << vecData[iPoint] << "  ";
        std::cout << "\n";
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);

    if (thisRankId == 1)
      {
        std::cout << " This rank = 1 \n";
        std::cout << " Vec data = ";
        for (dftfe::size_type iPoint = 0; iPoint < 7; iPoint++)
          std::cout << vecData[iPoint] << "  ";
        std::cout << "\n";
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);

    mpiCommP2PObj.accumulateInsertLocallyOwned(vecData);

    std::cout << std::flush;
    MPI_Barrier(mpiComm);


    if (thisRankId == 0)
      {
        std::cout << " This rank = 0 \n";
        std::cout << " Vec data = ";
        for (dftfe::size_type iPoint = 0; iPoint < 10; iPoint++)
          std::cout << vecData[iPoint] << "  ";
        std::cout << "\n";
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);

    if (thisRankId == 1)
      {
        std::cout << " This rank = 1 \n";
        std::cout << " Vec data = ";
        for (dftfe::size_type iPoint = 0; iPoint < 7; iPoint++)
          std::cout << vecData[iPoint] << "  ";
        std::cout << "\n";
      }

    std::cout << std::flush;
    MPI_Barrier(mpiComm);
  }

  void
  testTransferFromParentToChildIncompatiblePartitioning(
    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
      dftfe::utils::MemorySpace::HOST>>     BLASWrapperPtr,
    const MPI_Comm &                        mpi_comm_parent,
    const MPI_Comm &                        mpi_comm_domain,
    const MPI_Comm &                        interpoolcomm,
    const MPI_Comm &                        interbandgroup_comm,
    const unsigned int                      FEOrder,
    const dftfe::dftParameters &            dftParams,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria)
  {
    // create triangulation


    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);

    dftfe::triangulationManager dftMesh(mpi_comm_parent,
                                        mpi_comm_domain,
                                        interpoolcomm,
                                        interbandgroup_comm,
                                        FEOrder,
                                        dftParams);

    dftfe::dftParameters dftParamsVxc(dftParams);

    dftParamsVxc.innerAtomBallRadius = dftParams.innerAtomBallRadius * 2;
    dftParamsVxc.meshSizeInnerBall   = dftParams.meshSizeInnerBall / 2;
    dftParamsVxc.meshSizeOuterDomain = dftParams.meshSizeOuterDomain;
    dftParamsVxc.outerAtomBallRadius = dftParams.outerAtomBallRadius;
    dftParamsVxc.meshSizeOuterBall   = dftParams.meshSizeOuterBall;

    dftfe::triangulationManager VxcMesh(mpi_comm_parent,
                                        mpi_comm_domain,
                                        interpoolcomm,
                                        interbandgroup_comm,
                                        FEOrder,
                                        dftParamsVxc);
    //
    dftMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false); // generateSerialTria

    const dealii::parallel::distributed::Triangulation<3> &parallelMeshUnmoved =
      dftMesh.getParallelMeshUnmoved();
    const dealii::parallel::distributed::Triangulation<3> &parallelMeshMoved =
      dftMesh.getParallelMeshMoved();

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);



    // create Vxc mesh

    VxcMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false); // generateSerialTria

    const dealii::parallel::distributed::Triangulation<3>
      &parallelMeshMovedVxc = VxcMesh.getParallelMeshMoved();

    const dealii::parallel::distributed::Triangulation<3>
      &parallelMeshUnmovedVxc = VxcMesh.getParallelMeshUnmoved();

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);
    // construct dofHandler and constraints matrix

    dealii::DoFHandler<3> dofHandlerTria(parallelMeshMoved);

    dealii::DoFHandler<3> dofHandlerTriaVxc(parallelMeshMovedVxc);
    //    dealii::DoFHandler<3> dofHandlerTriaVxc(parallelMeshMoved);
    const dealii::FE_Q<3> finite_elementHigh(FEOrder + 2);
    const dealii::FE_Q<3> finite_elementLow(FEOrder);

    dofHandlerTria.distribute_dofs(finite_elementHigh);

    dofHandlerTriaVxc.distribute_dofs(finite_elementLow);

    dealii::AffineConstraints<double> constraintMatrix, constraintMatrixVxc;

    dealii::IndexSet locallyRelevantDofs, locallyRelevantDofsVxc;

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTria,
                                                    locallyRelevantDofs);

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTriaVxc,
                                                    locallyRelevantDofsVxc);


    constraintMatrix.clear();
    constraintMatrix.reinit(locallyRelevantDofs);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTria,
                                                    constraintMatrix);
    constraintMatrix.close();

    constraintMatrixVxc.clear();
    constraintMatrix.reinit(locallyRelevantDofsVxc);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTriaVxc,
                                                    constraintMatrixVxc);


    constraintMatrixVxc.close();

    // create quadrature

    dealii::QGauss<3>             gaussQuadHigh(FEOrder + 3);
    dealii::QGauss<3>             gaussQuadLow(FEOrder + 1);
    unsigned int                  numQuadPointsHigh = gaussQuadHigh.size();
    unsigned int                  numQuadPointsLow  = gaussQuadLow.size();
    dealii::MatrixFree<3, double> matrixFreeData, matrixFreeDataVxc;


    matrixFreeData.reinit(dealii::MappingQ1<3, 3>(),
                          dofHandlerTria,
                          constraintMatrix,
                          gaussQuadHigh);

    matrixFreeDataVxc.reinit(dealii::MappingQ1<3, 3>(),
                             dofHandlerTriaVxc,
                             constraintMatrixVxc,
                             gaussQuadLow);

    dftfe::distributedCPUMultiVec<dftfe::dataTypes::number> parentVec, childVec;

    unsigned int blockSize = 1;


    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(0), blockSize, parentVec);

    dftfe::dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      multiVectorConstraintsParent;
    multiVectorConstraintsParent.initialize(
      matrixFreeData.get_vector_partitioner(0), constraintMatrix);

    dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      quadValuesChildAnalytical, quadValuesChildComputed;

    unsigned int totalLocallyOwnedCellsVxc =
      matrixFreeDataVxc.n_physical_cells();
    quadValuesChildAnalytical.resize(totalLocallyOwnedCellsVxc *
                                     numQuadPointsLow * blockSize);
    quadValuesChildComputed.resize(totalLocallyOwnedCellsVxc *
                                   numQuadPointsLow * blockSize);


    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), dofHandlerTria, dof_coord);


    dealii::types::global_dof_index numberDofsParent = dofHandlerTria.n_dofs();

    std::shared_ptr<
      const dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      parentMPIPattern = parentVec.getMPIPatternP2P();
    const std::pair<dftfe::global_size_type, dftfe::global_size_type>
      &locallyOwnedRangeParent = parentMPIPattern->getLocallyOwnedRange();

    for (dealii::types::global_dof_index iNode = locallyOwnedRangeParent.first;
         iNode < locallyOwnedRangeParent.second;
         iNode++)
      {
        for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
          {
            unsigned int indexVec =
              (iNode - locallyOwnedRangeParent.first) * blockSize + iBlock;
            if ((!constraintMatrix.is_constrained(iNode)))
              *(parentVec.data() + indexVec) = value(dof_coord[iNode][0],
                                                     dof_coord[iNode][1],
                                                     dof_coord[iNode][2],
                                                     iBlock);
          }
      }

    parentVec.updateGhostValues();
    multiVectorConstraintsParent.distribute(parentVec);

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);



    dftfe::TransferDataBetweenMeshesIncompatiblePartitioning<
      dftfe::utils::MemorySpace::HOST>
      inverseDftDoFManagerObj(matrixFreeData,
                              0,
                              0,
                              matrixFreeDataVxc,
                              0,
                              0,
                              dftParams.verbosity,
                              mpi_comm_domain);

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapParent;
    dftfe::vectorTools::computeCellLocalIndexSetMap(
      parentVec.getMPIPatternP2P(),
      matrixFreeData,
      0,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParent);

    dftfe::global_size_type numPointsChild = (dftfe::global_size_type)(
      totalLocallyOwnedCellsVxc * numQuadPointsLow * blockSize);

    MPI_Allreduce(MPI_IN_PLACE,
                  &numPointsChild,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numPointsChild),
                  MPI_SUM,
                  mpi_comm_domain);

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);


    double startTimeMesh1ToMesh2 = MPI_Wtime();
    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
      fullFlattenedArrayCellLocalProcIndexIdMapParentMemStorage;
    fullFlattenedArrayCellLocalProcIndexIdMapParentMemStorage.resize(
      fullFlattenedArrayCellLocalProcIndexIdMapParent.size());
    fullFlattenedArrayCellLocalProcIndexIdMapParentMemStorage.copyFrom(
      fullFlattenedArrayCellLocalProcIndexIdMapParent);
    inverseDftDoFManagerObj.interpolateMesh1DataToMesh2QuadPoints(
      BLASWrapperPtr,
      parentVec,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParentMemStorage,
      quadValuesChildComputed,
      true);


    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);
    double endTimeMesh1ToMesh2 = MPI_Wtime();

    if ((dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0) &&
        (dftParams.verbosity > 2))
      {
        std::cout << " Num of points  = " << numPointsChild << "\n";
        std::cout << " Time taken to transfer from Mesh 1 to Mesh 2 = "
                  << endTimeMesh1ToMesh2 - startTimeMesh1ToMesh2 << "\n";
      }

    dealii::FEValues<3> fe_valuesChild(dofHandlerTriaVxc.get_fe(),
                                       gaussQuadLow,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellChild = dofHandlerTriaVxc.begin_active(),
      endcChild = dofHandlerTriaVxc.end();

    unsigned int iCellChildIndex = 0;
    for (; cellChild != endcChild; cellChild++)
      {
        if (cellChild->is_locally_owned())
          {
            fe_valuesChild.reinit(cellChild);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsLow; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesChild.quadrature_point(iQuad);
                for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
                  {
                    quadValuesChildAnalytical
                      [(iCellChildIndex * numQuadPointsLow + iQuad) *
                         blockSize +
                       iBlock] =
                        value(qPointVal[0], qPointVal[1], qPointVal[2], iBlock);
                  }
              }
            iCellChildIndex++;
          }
      }

    double l2Error = 0.0;
    for (unsigned int iQuad = 0; iQuad < quadValuesChildAnalytical.size();
         iQuad++)
      {
        dftfe::dataTypes::number diff =
          quadValuesChildComputed[iQuad] - quadValuesChildAnalytical[iQuad];
        dftfe::dataTypes::number errorVal = conj_compl(diff) * diff;
        l2Error += real_part(errorVal);
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &l2Error, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_domain);
    l2Error = std::sqrt(l2Error);
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0)
      {
        if (l2Error > 1e-9)
          {
            std::cout << " Error while interpolating to quad points of child = "
                      << l2Error << "\n";
          }
        else
          {
            std::cout
              << " Interpolation to quad points of child of successful\n";
          }
      }



    // test transfer child to parent

    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeDataVxc.get_vector_partitioner(0), blockSize, childVec);

    dftfe::dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      multiVectorConstraintsChild;
    multiVectorConstraintsChild.initialize(
      matrixFreeDataVxc.get_vector_partitioner(0), constraintMatrixVxc);

    dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      quadValuesParentAnalytical, quadValuesParentComputed;

    unsigned int totalLocallyOwnedCellsParent =
      matrixFreeData.n_physical_cells();
    quadValuesParentAnalytical.resize(totalLocallyOwnedCellsParent *
                                      numQuadPointsHigh * blockSize);
    quadValuesParentComputed.resize(totalLocallyOwnedCellsParent *
                                    numQuadPointsHigh * blockSize);


    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord_child;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), dofHandlerTriaVxc, dof_coord_child);


    dealii::types::global_dof_index numberDofsChild =
      dofHandlerTriaVxc.n_dofs();

    std::shared_ptr<
      const dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      childMPIPattern = childVec.getMPIPatternP2P();
    const std::pair<dftfe::global_size_type, dftfe::global_size_type>
      &locallyOwnedRangeChild = childMPIPattern->getLocallyOwnedRange();

    for (dealii::types::global_dof_index iNode = locallyOwnedRangeChild.first;
         iNode < locallyOwnedRangeChild.second;
         iNode++)
      {
        for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
          {
            unsigned int indexVec =
              (iNode - locallyOwnedRangeChild.first) * blockSize + iBlock;
            if ((!constraintMatrixVxc.is_constrained(iNode)))
              *(childVec.data() + indexVec) = value(dof_coord_child[iNode][0],
                                                    dof_coord_child[iNode][1],
                                                    dof_coord_child[iNode][2],
                                                    iBlock);
          }
      }

    childVec.updateGhostValues();
    multiVectorConstraintsChild.distribute(childVec);

    dftfe::global_size_type numPointsParent =
      totalLocallyOwnedCellsParent * numQuadPointsHigh * blockSize;

    MPI_Allreduce(MPI_IN_PLACE,
                  &numPointsParent,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numPointsParent),
                  MPI_SUM,
                  mpi_comm_domain);
    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapChild;
    dftfe::vectorTools::computeCellLocalIndexSetMap(
      childVec.getMPIPatternP2P(),
      matrixFreeDataVxc,
      0,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapChild);

    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage;
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.resize(
      fullFlattenedArrayCellLocalProcIndexIdMapChild.size());
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.copyFrom(
      fullFlattenedArrayCellLocalProcIndexIdMapChild);

    double startTimeMesh2ToMesh1 = MPI_Wtime();
    inverseDftDoFManagerObj.interpolateMesh2DataToMesh1QuadPoints(
      BLASWrapperPtr,
      childVec,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage,
      quadValuesParentComputed,
      true);

    std::cout << std::flush;
    MPI_Barrier(mpi_comm_domain);
    double endTimeMesh2ToMesh1 = MPI_Wtime();

    if ((dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0) &&
        (dftParams.verbosity > 2))
      {
        std::cout << " Number of points parent = " << numPointsParent << "\n";
        std::cout << " Time taken to transfer from Mesh 2 to Mesh 1 = "
                  << endTimeMesh2ToMesh1 - startTimeMesh2ToMesh1 << "\n";
      }

    dealii::FEValues<3> fe_valuesParent(dofHandlerTria.get_fe(),
                                        gaussQuadHigh,
                                        dealii::update_values |
                                          dealii::update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellParent = dofHandlerTria.begin_active(),
      endcParent = dofHandlerTria.end();

    unsigned int iCellParentIndex = 0;
    for (; cellParent != endcParent; cellParent++)
      {
        if (cellParent->is_locally_owned())
          {
            fe_valuesParent.reinit(cellParent);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsHigh; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesParent.quadrature_point(iQuad);
                for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
                  {
                    quadValuesParentAnalytical
                      [(iCellParentIndex * numQuadPointsHigh + iQuad) *
                         blockSize +
                       iBlock] =
                        value(qPointVal[0], qPointVal[1], qPointVal[2], iBlock);
                  }
              }
            iCellParentIndex++;
          }
      }

    l2Error = 0.0;
    for (unsigned int iQuad = 0; iQuad < quadValuesParentAnalytical.size();
         iQuad++)
      {
        dftfe::dataTypes::number diff =
          (quadValuesParentComputed[iQuad] - quadValuesParentAnalytical[iQuad]);

        dftfe::dataTypes::number errorVal = conj_compl(diff) * diff;
        l2Error += real_part(errorVal);
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &l2Error, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_domain);
    l2Error = std::sqrt(l2Error);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0)
      {
        if (l2Error > 1e-9)
          {
            std::cout
              << " Error while interpolating to quad points of parent = "
              << l2Error << "\n";
          }
        else
          {
            std::cout
              << " Interpolation to quad points of parent of successful\n";
          }
      }
  }
} // end of namespace functionalTest
