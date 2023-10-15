// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri, Sambit Das
//

#include <dftUtils.h>
#include <vectorUtilities.h>

#include <exception>

namespace dftfe
{
  namespace vectorTools
  {
    void
    createParallelConstraintMatrixFromSerial(
      const dealii::Triangulation<3, 3> &     serTria,
      const dealii::DoFHandler<3> &           dofHandlerPar,
      const MPI_Comm &                        mpi_comm_parent,
      const MPI_Comm &                        mpi_comm_domain,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      dealii::AffineConstraints<double> &     periodicHangingConstraints,
      dealii::AffineConstraints<double> &     onlyHangingConstraints,
      const int                               verbosity,
      const bool                              periodicX,
      const bool                              periodicY,
      const bool                              periodicZ)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0));
      dealii::TimerOutput computing_timer(mpi_comm_domain,
                                          pcout,
                                          verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      computing_timer.enter_subsection(
        "Create constraints from serial dofHandler");

      const dealii::IndexSet &locally_owned_dofs_par =
        dofHandlerPar.locally_owned_dofs();

      dealii::IndexSet locally_relevant_dofs_par;
      dealii::DoFTools::extract_locally_relevant_dofs(
        dofHandlerPar, locally_relevant_dofs_par);

      dealii::DoFHandler<3> dofHandlerSer(serTria);
      dofHandlerSer.distribute_dofs(dofHandlerPar.get_fe());

      const dealii::types::global_dof_index numGlobalDofs =
        dofHandlerSer.n_locally_owned_dofs();
      std::vector<dealii::types::global_dof_index> newDofNumbers(numGlobalDofs,
                                                                 0);

      std::map<dealii::CellId, dealii::DoFHandler<3>::active_cell_iterator>
        cellIdToCellIterMapSer;
      std::map<dealii::CellId, dealii::DoFHandler<3>::active_cell_iterator>
        cellIdToCellIterMapPar;

      for (const auto &cell : dofHandlerPar.active_cell_iterators())
        if (!cell->is_artificial())
          cellIdToCellIterMapPar[cell->id()] = cell;

      for (const auto &cell : dofHandlerSer.active_cell_iterators())
        if (cellIdToCellIterMapPar.find(cell->id()) !=
            cellIdToCellIterMapPar.end())
          cellIdToCellIterMapSer[cell->id()] = cell;

      const unsigned int dofs_per_cell = dofHandlerPar.get_fe().dofs_per_cell;
      std::vector<dealii::types::global_dof_index> cell_dof_indices_par(
        dofs_per_cell);
      std::vector<dealii::types::global_dof_index> cell_dof_indices_ser(
        dofs_per_cell);

      for (const auto &cell : dofHandlerPar.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(cell_dof_indices_par);
            cellIdToCellIterMapSer[cell->id()]->get_dof_indices(
              cell_dof_indices_ser);
            for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
              if (locally_owned_dofs_par.is_element(
                    cell_dof_indices_par[iNode]))
                newDofNumbers[cell_dof_indices_ser[iNode]] =
                  cell_dof_indices_par[iNode];
          }

      MPI_Allreduce(MPI_IN_PLACE,
                    &newDofNumbers[0],
                    numGlobalDofs,
                    DEAL_II_DOF_INDEX_MPI_TYPE,
                    MPI_SUM,
                    mpi_comm_domain);

      dofHandlerSer.renumber_dofs(newDofNumbers);

      if (verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(mpi_comm_domain,
                                          "Renumbered serial dofHandler");

      dealii::AffineConstraints<double> constraintsHangingSer;

#ifdef DFTFE_WITH_CUSTOMIZED_DEALII
      dealii::DoFTools::make_hanging_node_constraints_from_serial(
        dofHandlerSer,
        dofHandlerPar,
        cellIdToCellIterMapSer,
        constraintsHangingSer);
#else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: attempted to use dealii::DoFTools::make_hanging_node_constraints_from_serial which can only be used by linking to the customized dealii (https://github.com/dftfeDevelopers/dealii)."));

#endif
      if (verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpi_comm_domain, "Created hanging node constraints serial");

      dealii::AffineConstraints<double> constraintsPeriodicHangingSer;
      constraintsPeriodicHangingSer.merge(
        constraintsHangingSer,
        dealii::AffineConstraints<
          double>::MergeConflictBehavior::right_object_wins);
      constraintsHangingSer.close();

      // create unitVectorsXYZ
      std::vector<std::vector<double>> unitVectorsXYZ(3,
                                                      std::vector<double>(3,
                                                                          0.0));

      std::vector<dealii::Tensor<1, 3>> offsetVectors;
      // resize offset vectors
      offsetVectors.resize(3);

      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          offsetVectors[i][j] =
            unitVectorsXYZ[i][j] - domainBoundingVectors[i][j];

      std::vector<dealii::GridTools::PeriodicFacePair<
        typename dealii::DoFHandler<3>::cell_iterator>>
        periodicity_vector2;

      std::vector<int>         periodicDirectionVector;
      const std::array<int, 3> periodic = {periodicX, periodicY, periodicZ};
      for (unsigned int d = 0; d < 3; ++d)
        if (periodic[d] == 1)
          periodicDirectionVector.push_back(d);


      for (unsigned int i = 0;
           i < std::accumulate(periodic.begin(), periodic.end(), 0);
           ++i)
        dealii::GridTools::collect_periodic_faces(
          dofHandlerSer,
          /*b_id1*/ 2 * i + 1,
          /*b_id2*/ 2 * i + 2,
          /*direction*/ periodicDirectionVector[i],
          periodicity_vector2,
          offsetVectors[periodicDirectionVector[i]]);

      dealii::DoFTools::make_periodicity_constraints<3, 3>(
        periodicity_vector2, constraintsPeriodicHangingSer);

      constraintsPeriodicHangingSer.close();

      if (verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpi_comm_domain, "Created periodic constraints serial");

      periodicHangingConstraints.clear();
      periodicHangingConstraints.reinit(locally_relevant_dofs_par);

      onlyHangingConstraints.clear();
      onlyHangingConstraints.reinit(locally_relevant_dofs_par);

      for (auto index : locally_relevant_dofs_par)
        {
          if (constraintsPeriodicHangingSer.is_constrained(index))
            {
              periodicHangingConstraints.add_line(index);
              periodicHangingConstraints.add_entries(
                index,
                *constraintsPeriodicHangingSer.get_constraint_entries(index));
            }

          if (constraintsHangingSer.is_constrained(index))
            {
              onlyHangingConstraints.add_line(index);
              onlyHangingConstraints.add_entries(
                index, *constraintsHangingSer.get_constraint_entries(index));
            }
        }

      periodicHangingConstraints.close();
      onlyHangingConstraints.close();

      computing_timer.leave_subsection(
        "Create constraints from serial dofHandler");
    }

    template <typename T>
    void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                   partitioner,
      const unsigned int    blockSize,
      distributedCPUVec<T> &flattenedArray)
    {
      const MPI_Comm &mpi_communicator = partitioner->get_mpi_communicator();
      //
      // Get required sizes
      //
      const unsigned int n_ghosts  = partitioner->n_ghost_indices();
      const unsigned int localSize = partitioner->local_size();
      const unsigned int totalSize = localSize + n_ghosts;
      const dealii::types::global_dof_index globalNumberDegreesOfFreedom =
        partitioner->size();

      //
      // create data for new parallel layout
      //
      dealii::IndexSet locallyOwnedFlattenedNodesSet, ghostFlattenedNodesSet;
      locallyOwnedFlattenedNodesSet.clear();
      ghostFlattenedNodesSet.clear();

      //
      // Set the maximal size of the indices upon which this object operates.
      //
      locallyOwnedFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);
      ghostFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);


      for (unsigned int ilocaldof = 0; ilocaldof < totalSize; ++ilocaldof)
        {
          std::vector<dealii::types::global_dof_index>
                                                       newLocallyOwnedGlobalNodeIds;
          std::vector<dealii::types::global_dof_index> newGhostGlobalNodeIds;
          const dealii::types::global_dof_index        globalIndex =
            partitioner->local_to_global(ilocaldof);
          const bool isGhost = partitioner->is_ghost_entry(globalIndex);
          if (isGhost)
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newGhostGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }
          else
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newLocallyOwnedGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }

          // insert into dealii index sets
          locallyOwnedFlattenedNodesSet.add_indices(
            newLocallyOwnedGlobalNodeIds.begin(),
            newLocallyOwnedGlobalNodeIds.end());
          ghostFlattenedNodesSet.add_indices(newGhostGlobalNodeIds.begin(),
                                             newGhostGlobalNodeIds.end());
        }

      // compress index set ranges
      locallyOwnedFlattenedNodesSet.compress();
      ghostFlattenedNodesSet.compress();

      bool print = false;
      if (print)
        {
          std::cout << "Number of Wave Functions per Block: " << blockSize
                    << std::endl;
          std::stringstream ss1;
          locallyOwnedFlattenedNodesSet.print(ss1);
          std::stringstream ss2;
          ghostFlattenedNodesSet.print(ss2);
          std::string s1(ss1.str());
          s1.pop_back();
          std::string s2(ss2.str());
          s2.pop_back();
          std::cout << "procId: "
                    << dealii::Utilities::MPI::this_mpi_process(
                         mpi_communicator)
                    << " new owned: " << s1 << " new ghost: " << s2
                    << std::endl;
        }

      //
      // sanity check
      //
      AssertThrow(
        locallyOwnedFlattenedNodesSet.is_ascending_and_one_to_one(
          mpi_communicator),
        dealii::ExcMessage(
          "Incorrect renumbering and/or partitioning of flattened wave function matrix"));

      //
      // create flattened wave function matrix
      //
      flattenedArray.reinit(locallyOwnedFlattenedNodesSet,
                            ghostFlattenedNodesSet,
                            mpi_communicator);
    }


    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<dealii::types::global_dof_index>
        &flattenedArrayCellLocalProcIndexIdMap)

    {
      //
      // get FE cell data
      //
      const unsigned int numberMacroCells = matrix_free_data.n_cell_batches();
      const unsigned int numberNodesPerElement =
        matrix_free_data.get_dofs_per_cell(mfDofHandlerIndex);
      const unsigned int totalLocallyOwnedCells =
        matrix_free_data.n_physical_cells();


      std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
        numberNodesPerElement);
      //
      // create map for all locally owned cells in the same order
      //
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          matrix_free_data.get_dof_handler(mfDofHandlerIndex).begin_active(),
        endc = matrix_free_data.get_dof_handler(mfDofHandlerIndex).end();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        numberNodesPerElement);

      flattenedArrayCellLocalProcIndexIdMap.clear();
      flattenedArrayCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells *
                                                   numberNodesPerElement);

      unsigned int iElemCount = 0;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];

                  // Think about variable blockSize
                  // dealii::types::global_dof_index globalIndexFlattenedArray =
                  //   (dealii::types::global_dof_index)blockSize * globalIndex;
                  dealii::types::global_dof_index localIndexFlattenedArray =
                    (partitioner->globalToLocal(globalIndex)) *
                    (dealii::types::global_dof_index)blockSize;
                  flattenedArrayCellLocalProcIndexIdMap
                    [iElemCount * numberNodesPerElement + iNode] =
                      localIndexFlattenedArray;
                }
              ++iElemCount;
            }
        }
    }


    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<dealii::types::global_dof_index>
        &                        flattenedArrayMacroCellLocalProcIndexIdMap,
      std::vector<unsigned int> &normalCellIdToMacroCellIdMap,
      std::vector<unsigned int> &macroCellIdToNormalCellIdMap,
      std::vector<dealii::types::global_dof_index>
        &flattenedArrayCellLocalProcIndexIdMap)

    {
      //
      // get FE cell data
      //
      const unsigned int numberMacroCells = matrix_free_data.n_cell_batches();
      const unsigned int numberNodesPerElement =
        matrix_free_data.get_dofs_per_cell(mfDofHandlerIndex);
      const unsigned int totalLocallyOwnedCells =
        matrix_free_data.n_physical_cells();


      std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
        numberNodesPerElement);

      flattenedArrayMacroCellLocalProcIndexIdMap.clear();
      flattenedArrayMacroCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells *
                                                        numberNodesPerElement);

      //
      // create map for all locally owned cells in the order of macrocell,
      // subcell order
      //
      unsigned int                                         iElem = 0;
      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
      for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
           ++iMacroCell)
        {
          const unsigned int n_sub_cells =
            matrix_free_data.n_active_entries_per_cell_batch(iMacroCell);
          for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
            {
              cellPtr = matrix_free_data.get_cell_iterator(iMacroCell,
                                                           iCell,
                                                           mfDofHandlerIndex);
              cellPtr->get_dof_indices(cell_dof_indicesGlobal);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indicesGlobal[iNode];

                  // Think about variable blockSize
                  // dealii::types::global_dof_index globalIndexFlattenedArray =
                  //   (dealii::types::global_dof_index)blockSize * globalIndex;
                  dealii::types::global_dof_index localIndexFlattenedArray =
                    (partitioner->globalToLocal(globalIndex)) *
                    (dealii::types::global_dof_index)blockSize;
                  flattenedArrayMacroCellLocalProcIndexIdMap
                    [iElem * numberNodesPerElement + iNode] =
                      localIndexFlattenedArray;
                } // idof loop
              ++iElem;
            } // subcell loop
        }     // macrocell loop

      //
      // create map for all locally owned cells in the same order
      //
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          matrix_free_data.get_dof_handler(mfDofHandlerIndex).begin_active(),
        endc = matrix_free_data.get_dof_handler(mfDofHandlerIndex).end();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        numberNodesPerElement);

      flattenedArrayCellLocalProcIndexIdMap.clear();
      flattenedArrayCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells *
                                                   numberNodesPerElement);

      unsigned int iElemCount = 0;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];

                  // Think about variable blockSize
                  // dealii::types::global_dof_index globalIndexFlattenedArray =
                  //   (dealii::types::global_dof_index)blockSize * globalIndex;
                  dealii::types::global_dof_index localIndexFlattenedArray =
                    (partitioner->globalToLocal(globalIndex)) *
                    (dealii::types::global_dof_index)blockSize;
                  flattenedArrayCellLocalProcIndexIdMap
                    [iElemCount * numberNodesPerElement + iNode] =
                      localIndexFlattenedArray;
                }
              ++iElemCount;
            }
        }


      // create map between macro cell id and normal cell id
      normalCellIdToMacroCellIdMap.clear();
      normalCellIdToMacroCellIdMap.resize(totalLocallyOwnedCells);

      macroCellIdToNormalCellIdMap.clear();
      macroCellIdToNormalCellIdMap.resize(totalLocallyOwnedCells);
      unsigned int iElemNormal = 0;
      cell = matrix_free_data.get_dof_handler(mfDofHandlerIndex).begin_active();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              bool         isFound        = false;
              unsigned int iElemMacroCell = 0;
              for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
                   ++iMacroCell)
                {
                  const unsigned int n_sub_cells =
                    matrix_free_data.n_active_entries_per_cell_batch(
                      iMacroCell);
                  for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
                    {
                      cellPtr =
                        matrix_free_data.get_cell_iterator(iMacroCell,
                                                           iCell,
                                                           mfDofHandlerIndex);
                      if (cell->id() == cellPtr->id())
                        {
                          normalCellIdToMacroCellIdMap[iElemNormal] =
                            iElemMacroCell;
                          macroCellIdToNormalCellIdMap[iElemMacroCell] =
                            iElemNormal;
                          isFound = true;
                          break;
                        }
                      iElemMacroCell++;
                    }

                  if (isFound)
                    break;
                }
              iElemNormal++;
            }
        }
    }



    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<std::vector<dealii::types::global_dof_index>>
        &flattenedArrayMacroCellLocalProcIndexIdMap,
      std::vector<std::vector<dealii::types::global_dof_index>>
        &flattenedArrayCellLocalProcIndexIdMap)

    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(
           matrix_free_data.get_vector_partitioner()->get_mpi_communicator()) ==
         0));

      //
      // get FE cell data
      //
      const unsigned int numberMacroCells = matrix_free_data.n_cell_batches();
      const unsigned int numberNodesPerElement =
        matrix_free_data.get_dofs_per_cell(mfDofHandlerIndex);


      std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
        numberNodesPerElement);

      //
      // get total locally owned cells
      //
      int totalLocallyOwnedCells = 0;
      for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
           ++iMacroCell)
        {
          const unsigned int n_sub_cells =
            matrix_free_data.n_active_entries_per_cell_batch(iMacroCell);
          for (unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
            {
              totalLocallyOwnedCells++;
            }
        }

      flattenedArrayMacroCellLocalProcIndexIdMap.clear();
      flattenedArrayMacroCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells);

      //
      // create map for all locally owned cells in the order of macrocell,
      // subcell order
      //
      unsigned int                                         iElem = 0;
      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
      for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
           ++iMacroCell)
        {
          const unsigned int n_sub_cells =
            matrix_free_data.n_active_entries_per_cell_batch(iMacroCell);
          for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
            {
              cellPtr = matrix_free_data.get_cell_iterator(iMacroCell,
                                                           iCell,
                                                           mfDofHandlerIndex);
              cellPtr->get_dof_indices(cell_dof_indicesGlobal);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indicesGlobal[iNode];

                  // Think about variable blockSize
                  // dealii::types::global_dof_index globalIndexFlattenedArray =
                  //   (dealii::types::global_dof_index)blockSize * globalIndex;
                  dealii::types::global_dof_index localIndexFlattenedArray =
                    (partitioner->globalToLocal(globalIndex)) *
                    (dealii::types::global_dof_index)blockSize;
                  flattenedArrayMacroCellLocalProcIndexIdMap[iElem].push_back(
                    localIndexFlattenedArray);
                } // idof loop
              ++iElem;
            } // subcell loop
        }     // macrocell loop


      //
      // create map for all locally owned cells in the same order
      //
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          matrix_free_data.get_dof_handler(mfDofHandlerIndex).begin_active(),
        endc = matrix_free_data.get_dof_handler(mfDofHandlerIndex).end();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        numberNodesPerElement);

      flattenedArrayCellLocalProcIndexIdMap.clear();
      flattenedArrayCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells);

      unsigned int iElemCount = 0;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];

                  // Think about variable blockSize
                  // dealii::types::global_dof_index globalIndexFlattenedArray =
                  //   (dealii::types::global_dof_index)blockSize * globalIndex;
                  dealii::types::global_dof_index localIndexFlattenedArray =
                    (partitioner->globalToLocal(globalIndex)) *
                    (dealii::types::global_dof_index)blockSize;
                  flattenedArrayCellLocalProcIndexIdMap[iElemCount].push_back(
                    localIndexFlattenedArray);
                }
              ++iElemCount;
            }
        }
    }

#ifdef USE_COMPLEX
    void
    copyFlattenedSTLVecToSingleCompVec(
      const std::complex<double> *                flattenedArray,
      const unsigned int                          totalNumberComponents,
      const unsigned int                          localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                     localProcDofIndicesImag,
      std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        for (unsigned int icomp = componentIndexRange.first;
             icomp < componentIndexRange.second;
             ++icomp)
          {
            const unsigned int flattenedArrayLocalIndex =
              totalNumberComponents * iNode + icomp;

            componentVectors[icomp - componentIndexRange.first].local_element(
              localProcDofIndicesReal[iNode]) =
              flattenedArray[flattenedArrayLocalIndex].real();
            componentVectors[icomp - componentIndexRange.first].local_element(
              localProcDofIndicesImag[iNode]) =
              flattenedArray[flattenedArrayLocalIndex].imag();
          }
    }

    void
    copyFlattenedSTLVecToSingleCompVec(
      const std::complex<double> *                flattenedArray,
      const unsigned int                          totalNumberComponents,
      const unsigned int                          localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      std::vector<distributedCPUVec<double>> &    componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        for (unsigned int icomp = componentIndexRange.first;
             icomp < componentIndexRange.second;
             ++icomp)
          {
            const unsigned int flattenedArrayLocalIndex =
              totalNumberComponents * iNode + icomp;

            componentVectors[icomp - componentIndexRange.first].local_element(
              iNode) = flattenedArray[flattenedArrayLocalIndex].real();
          }
    }
#else
    void
    copyFlattenedSTLVecToSingleCompVec(
      const double *flattenedArray,
      const unsigned int totalNumberComponents,
      const unsigned int localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));
      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        for (unsigned int icomp = componentIndexRange.first;
             icomp < componentIndexRange.second;
             ++icomp)
          {
            const unsigned int flattenedArrayLocalIndex =
              totalNumberComponents * iNode + icomp;
            componentVectors[icomp - componentIndexRange.first].local_element(
              iNode) = flattenedArray[flattenedArrayLocalIndex];
          }
    }
#endif

#ifdef USE_COMPLEX
    void
    copyFlattenedDealiiVecToSingleCompVec(
      const distributedCPUVec<std::complex<double>> &flattenedArray,
      const unsigned int                             totalNumberComponents,
      const std::pair<unsigned int, unsigned int>    componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                     localProcDofIndicesImag,
      std::vector<distributedCPUVec<double>> &componentVectors,
      const bool isFlattenedDealiiGhostValuesUpdated)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                partitioner = flattenedArray.get_partitioner();
      const unsigned int localSize =
        partitioner->local_size() / totalNumberComponents;
      const unsigned int n_ghosts =
        partitioner->n_ghost_indices() / totalNumberComponents;
      const unsigned int totalSize = localSize + n_ghosts;

      if (!isFlattenedDealiiGhostValuesUpdated)
        {
          for (unsigned int iNode = 0; iNode < localSize; ++iNode)
            for (unsigned int icomp = componentIndexRange.first;
                 icomp < componentIndexRange.second;
                 ++icomp)
              {
                const unsigned int flattenedArrayLocalIndex =
                  totalNumberComponents * iNode + icomp;

                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesReal[iNode]) =
                  flattenedArray.local_element(flattenedArrayLocalIndex).real();
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesImag[iNode]) =
                  flattenedArray.local_element(flattenedArrayLocalIndex).imag();
              }

          for (unsigned int i = 0; i < componentVectors.size(); ++i)
            componentVectors[i].update_ghost_values();
        }
      else
        {
          for (unsigned int iNode = 0; iNode < totalSize; ++iNode)
            for (unsigned int icomp = componentIndexRange.first;
                 icomp < componentIndexRange.second;
                 ++icomp)
              {
                const unsigned int flattenedArrayLocalIndex =
                  totalNumberComponents * iNode + icomp;

                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesReal[iNode]) =
                  flattenedArray.local_element(flattenedArrayLocalIndex).real();
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesImag[iNode]) =
                  flattenedArray.local_element(flattenedArrayLocalIndex).imag();
              }
        }
    }
#else
    void
    copyFlattenedDealiiVecToSingleCompVec(
      const distributedCPUVec<double> &flattenedArray,
      const unsigned int totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      std::vector<distributedCPUVec<double>> &componentVectors,
      const bool isFlattenedDealiiGhostValuesUpdated)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &partitioner = flattenedArray.get_partitioner();
      const unsigned int localSize =
        partitioner->local_size() / totalNumberComponents;
      const unsigned int n_ghosts =
        partitioner->n_ghost_indices() / totalNumberComponents;
      const unsigned int totalSize = localSize + n_ghosts;

      if (!isFlattenedDealiiGhostValuesUpdated)
        {
          for (unsigned int iNode = 0; iNode < localSize; ++iNode)
            for (unsigned int icomp = componentIndexRange.first;
                 icomp < componentIndexRange.second;
                 ++icomp)
              {
                const unsigned int flattenedArrayLocalIndex =
                  totalNumberComponents * iNode + icomp;
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(iNode) =
                  flattenedArray.local_element(flattenedArrayLocalIndex);
              }

          for (unsigned int i = 0; i < componentVectors.size(); ++i)
            componentVectors[i].update_ghost_values();
        }
      else
        {
          for (unsigned int iNode = 0; iNode < totalSize; ++iNode)
            for (unsigned int icomp = componentIndexRange.first;
                 icomp < componentIndexRange.second;
                 ++icomp)
              {
                const unsigned int flattenedArrayLocalIndex =
                  totalNumberComponents * iNode + icomp;
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(iNode) =
                  flattenedArray.local_element(flattenedArrayLocalIndex);
              }
        }
    }
#endif

#ifdef USE_COMPLEX
    void
    copySingleCompVecToFlattenedDealiiVec(
      distributedCPUVec<std::complex<double>> &   flattenedArray,
      const unsigned int                          totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                           localProcDofIndicesImag,
      const std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      const unsigned int localVectorSize =
        flattenedArray.local_size() / totalNumberComponents;
      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        {
          for (unsigned int icomp = componentIndexRange.first;
               icomp < componentIndexRange.second;
               ++icomp)
            {
              const unsigned int flattenedArrayLocalIndex =
                totalNumberComponents * iNode + icomp;

              const double real =
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesReal[iNode]);
              const double imag =
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(localProcDofIndicesImag[iNode]);

              flattenedArray.local_element(flattenedArrayLocalIndex) =
                std::complex<double>(real, imag);
            }
        }

      flattenedArray.update_ghost_values();
    }
#else
    void
    copySingleCompVecToFlattenedDealiiVec(
      distributedCPUVec<double> &flattenedArray,
      const unsigned int totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));
      const unsigned int localVectorSize =
        flattenedArray.local_size() / totalNumberComponents;
      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        {
          for (unsigned int icomp = componentIndexRange.first;
               icomp < componentIndexRange.second;
               ++icomp)
            {
              const unsigned int flattenedArrayLocalIndex =
                totalNumberComponents * iNode + icomp;
              flattenedArray.local_element(flattenedArrayLocalIndex) =
                componentVectors[icomp - componentIndexRange.first]
                  .local_element(iNode);
            }
        }

      flattenedArray.update_ghost_values();
    }
#endif

#ifdef USE_COMPLEX
    void
    copySingleCompVecToFlattenedSTLVec(
      std::vector<std::complex<double>> &         flattenedArray,
      const unsigned int                          totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                           localProcDofIndicesImag,
      const std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));

      const unsigned int localVectorSize =
        flattenedArray.size() / totalNumberComponents;
      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        for (unsigned int icomp = componentIndexRange.first;
             icomp < componentIndexRange.second;
             ++icomp)
          {
            const unsigned int flattenedArrayLocalIndex =
              totalNumberComponents * iNode + icomp;

            const double real =
              componentVectors[icomp - componentIndexRange.first].local_element(
                localProcDofIndicesReal[iNode]);
            const double imag =
              componentVectors[icomp - componentIndexRange.first].local_element(
                localProcDofIndicesImag[iNode]);

            flattenedArray[flattenedArrayLocalIndex] =
              std::complex<double>(real, imag);
          }
    }
#else
    void
    copySingleCompVecToFlattenedSTLVec(
      std::vector<double> &flattenedArray,
      const unsigned int totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<distributedCPUVec<double>> &componentVectors)
    {
      Assert(componentVectors.size() ==
               (componentIndexRange.second - componentIndexRange.first),
             dealii::ExcMessage("Incorrect dimensions of componentVectors"));
      Assert(componentIndexRange.first < totalNumberComponents &&
               componentIndexRange.second <= totalNumberComponents,
             dealii::ExcMessage(
               "componentIndexRange doesn't lie within totalNumberComponents"));
      const unsigned int localVectorSize =
        flattenedArray.size() / totalNumberComponents;
      for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
        for (unsigned int icomp = componentIndexRange.first;
             icomp < componentIndexRange.second;
             ++icomp)
          {
            const unsigned int flattenedArrayLocalIndex =
              totalNumberComponents * iNode + icomp;
            flattenedArray[flattenedArrayLocalIndex] =
              componentVectors[icomp - componentIndexRange.first].local_element(
                iNode);
          }
    }
#endif

    void
    classifyInteriorSurfaceNodesInCell(
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      std::vector<unsigned int> &          nodesPerCellClassificationMap)
    {
      const std::vector<dealii::Point<3>> &nodalCoordinatesRefCell =
        matrix_free_data.get_dof_handler(mfDofHandlerIndex)
          .get_fe()
          .get_unit_support_points();
      unsigned int numberNodesPerCell = nodalCoordinatesRefCell.size();
      nodesPerCellClassificationMap.resize(numberNodesPerCell, 0);

      double tol = 1e-05;

      for (unsigned int iNode = 0; iNode < numberNodesPerCell; ++iNode)
        {
          if (std::abs(nodalCoordinatesRefCell[iNode][0] - 1.0) < tol ||
              std::abs(nodalCoordinatesRefCell[iNode][0]) < tol)
            {
              nodesPerCellClassificationMap[iNode] = 1;
            }
          else if (std::abs(nodalCoordinatesRefCell[iNode][1] - 1.0) < tol ||
                   std::abs(nodalCoordinatesRefCell[iNode][1]) < tol)
            {
              nodesPerCellClassificationMap[iNode] = 1;
            }
          else if (std::abs(nodalCoordinatesRefCell[iNode][2] - 1.0) < tol ||
                   std::abs(nodalCoordinatesRefCell[iNode][2]) < tol)
            {
              nodesPerCellClassificationMap[iNode] = 1;
            }
        }
    }


    void
    classifyInteriorSurfaceNodesInGlobalArray(
      const dealii::MatrixFree<3, double> &    matrix_free_data,
      const unsigned int                       mfDofHandlerIndex,
      const dealii::AffineConstraints<double> &constraintMatrix,
      std::vector<unsigned int> &              nodesPerCellClassificationMap,
      std::vector<unsigned int> &              globalArrayClassificationMap)
    {
      distributedCPUVec<double> dummyVector;
      matrix_free_data.initialize_dof_vector(dummyVector, mfDofHandlerIndex);

      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          matrix_free_data.get_dof_handler(mfDofHandlerIndex).begin_active(),
        endc = matrix_free_data.get_dof_handler(mfDofHandlerIndex).end();

      const unsigned int numberNodesPerElement =
        matrix_free_data.get_dofs_per_cell(mfDofHandlerIndex);
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        numberNodesPerElement);
      const unsigned int numberDoFs = dummyVector.local_size();
      globalArrayClassificationMap.resize(numberDoFs, 0);

      //
      // create map
      //
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];
                  dealii::types::global_dof_index localIndex =
                    dummyVector.get_partitioner()->global_to_local(globalIndex);
                  if (nodesPerCellClassificationMap[iNode] == 1 &&
                      !constraintMatrix.is_constrained(globalIndex) &&
                      dummyVector.get_partitioner()->in_local_range(
                        globalIndex))
                    {
                      globalArrayClassificationMap[localIndex] = 1;
                    }
                }
            }
        }
    }



    std::pair<dealii::Point<3>, dealii::Point<3>>
    createBoundingBoxTriaLocallyOwned(const dealii::DoFHandler<3> &dofHandler)
    {
      dealii::Point<3>   corner1, corner2;
      const unsigned int vertices_per_cell =
        dealii::GeometryInfo<3>::vertices_per_cell;
      corner1[0] = 0.0;
      corner1[1] = 0.0;
      corner1[2] = 0.0;

      corner2[0] = 0.0;
      corner2[1] = 0.0;
      corner2[2] = 0.0;

      dealii::DoFHandler<3>::active_cell_iterator cell, endc;
      cell = dofHandler.begin_active(), endc = dofHandler.end();

      std::vector<bool> vertex_touched(
        dofHandler.get_triangulation().n_vertices(), false);

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            for (unsigned int i = 0; i < vertices_per_cell; ++i)
              {
                const unsigned global_vertex_no = cell->vertex_index(i);

                if (vertex_touched[global_vertex_no])
                  continue;
                vertex_touched[global_vertex_no] = true;

                dealii::Point<3> nodalCoor = cell->vertex(i);

                if (nodalCoor[0] < corner1[0])
                  corner1[0] = nodalCoor[0];

                if (nodalCoor[1] < corner1[1])
                  corner1[1] = nodalCoor[1];

                if (nodalCoor[2] < corner1[2])
                  corner1[2] = nodalCoor[2];

                if (nodalCoor[0] > corner2[0])
                  corner2[0] = nodalCoor[0];

                if (nodalCoor[1] > corner2[1])
                  corner2[1] = nodalCoor[1];

                if (nodalCoor[2] > corner2[2])
                  corner2[2] = nodalCoor[2];
              }
          }
      return std::pair<dealii::Point<3>, dealii::Point<3>>(corner1, corner2);
    }

    template void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &,
      const unsigned int,
      distributedCPUVec<double> &);

    template void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &,
      const unsigned int,
      distributedCPUVec<float> &);


    template void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &,
      const unsigned int,
      distributedCPUVec<std::complex<double>> &);

    template void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &,
      const unsigned int,
      distributedCPUVec<std::complex<float>> &);
  } // namespace vectorTools

} // namespace dftfe
