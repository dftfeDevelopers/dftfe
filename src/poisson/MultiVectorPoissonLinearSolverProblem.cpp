// ---------------------------------------------------------------------
//
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

#include "MultiVectorPoissonLinearSolverProblem.h"
#include "dftUtils.h"
#include "vectorUtilities.h"
#include "poissonSolverProblem.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  MultiVectorPoissonLinearSolverProblem<memorySpace>::
    MultiVectorPoissonLinearSolverProblem(const MPI_Comm &mpi_comm_parent,
                                          const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , d_mpi_parent(mpi_comm_parent)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr                      = NULL;
    d_blockedNDBCPtr                   = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
    d_diagonalA.resize(0);
    d_diagonalSqrtA.resize(0);
    d_isMeanValueConstraintComputed = false;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  MultiVectorPoissonLinearSolverProblem<
    memorySpace>::~MultiVectorPoissonLinearSolverProblem()
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::clear()
  {
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr                      = NULL;
    d_blockedNDBCPtr                   = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
    d_diagonalA.resize(0);
    d_diagonalSqrtA.resize(0);
    d_isMeanValueConstraintComputed = false;
    d_cellBlockSize                 = 100;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::reinit(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                             basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int                       matrixFreeQuadratureComponentRhs,
    const unsigned int                       matrixFreeQuadratureComponentAX,
    bool                                     isComputeMeanValueConstraint)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double time = MPI_Wtime();

    d_BLASWrapperPtr            = BLASWrapperPtr;
    d_basisOperationsPtr        = basisOperationsPtr;
    d_matrixFreeDataPtr         = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPtr       = &constraintMatrix;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentRhs = matrixFreeQuadratureComponentRhs;
    d_matrixFreeQuadratureComponentAX  = matrixFreeQuadratureComponentAX;


    d_basisOperationsPtr->reinit(1,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 false,  // TODO should this be set to true
                                 false); // TODO should this be set to true

    d_locallyOwnedSize     = d_basisOperationsPtr->nOwnedDofs();
    d_numberDofsPerElement = d_basisOperationsPtr->nDofsPerCell();
    d_numCells             = d_basisOperationsPtr->nCells();
    d_nQuadsPerCell        = d_basisOperationsPtr->nQuadsPerCell();

    // d_cellBlockSize = std::min(d_cellBlockSize, d_numCells);
    d_cellBlockSize = d_numCells;
    d_dofHandler =
      &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    if (isComputeMeanValueConstraint)
      {
        computeMeanValueConstraint();
        d_isMeanValueConstraintComputed = true;
      }


    d_basisOperationsPtr->computeStiffnessVector(true, true);

    preComputeShapeFunction();
    computeDiagonalA();
    d_isComputeDiagonalA = true;


    d_basisOperationsPtr->computeCellStiffnessMatrix(
      matrixFreeQuadratureComponentAX,
      d_numCells,
      true,
      false); // TODO setting the coeff to false
    d_cellStiffnessMatrixPtr =
      &(d_basisOperationsPtr->cellStiffnessMatrixBasisData());



    double l2NormStiff = 0.0;
    for (unsigned int iNode = 0;
         iNode < d_numCells * d_numberDofsPerElement * d_numberDofsPerElement;
         iNode++)
      {
        double diff = d_cellShapeFunctionGradientIntegral[iNode] -
                      d_cellStiffnessMatrixPtr->data()[iNode];
        l2NormStiff += diff * diff;
      }

    MPI_Allreduce(
      MPI_IN_PLACE, &l2NormStiff, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);


    d_constraintsInfo.initialize(d_matrixFreeDataPtr->get_vector_partitioner(
                                   matrixFreeVectorComponent),
                                 constraintMatrix);

    d_inc                 = 1;
    d_negScalarCoeffAlpha = -1.0 / (4.0 * M_PI);
    d_scalarCoeffAlpha    = 1.0 / (4.0 * M_PI);
    d_beta                = 0.0;
    d_alpha               = 1.0;
    d_transA              = 'N';
    d_transB              = 'N';
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::distributeX()
  {
    d_BLASWrapperPtr->axpby(d_locallyOwnedSize * d_blockSize,
                            d_alpha,
                            d_blockedNDBCPtr->data(),
                            d_alpha,
                            d_blockedXPtr->data());

    d_blockedXPtr->updateGhostValues();
    d_constraintsInfo.distribute(*d_blockedXPtr);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<
    memorySpace>::computeMeanValueConstraint()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::computeDiagonalA()
  {
    d_diagonalA = d_basisOperationsPtr->inverseStiffnessVectorBasisData();
    d_diagonalSqrtA =
      d_basisOperationsPtr->inverseSqrtStiffnessVectorBasisData();

    d_blockSize = 1;

    dftfe::poissonSolverProblem<2, 2> phiTotalSolverProblem(mpi_communicator);


    dftfe::distributedCPUVec<double> expectedOutput;

    dftfe::vectorTools::createDealiiVector<double>(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      1,
      expectedOutput);

    std::map<dealii::types::global_dof_index, double> atoms;
    std::map<dealii::CellId, std::vector<double>>     smearedChargeValues;

    phiTotalSolverProblem.reinit(d_basisOperationsPtr,
                                 expectedOutput,
                                 *d_constraintMatrixPtr,
                                 d_matrixFreeVectorComponent,
                                 d_matrixFreeQuadratureComponentRhs,
                                 d_matrixFreeQuadratureComponentAX,
                                 atoms,
                                 smearedChargeValues,
                                 d_matrixFreeQuadratureComponentAX,
                                 *d_rhsQuadDataPtr,
                                 true,  // isComputeDiagonalA
                                 false, // isComputeMeanValueConstraint
                                 false, // smearedNuclearCharges
                                 true,  // isRhoValues
                                 false, // isGradSmearedChargeRhs
                                 0,     // smearedChargeGradientComponentId
                                 false, // storeSmearedChargeRhs
                                 false, // reuseSmearedChargeRhs
                                 true); // reinitializeFastConstraints


    distributedCPUVec<double> rhsTempVec, outputVec;

    dftfe::vectorTools::createDealiiVector<double>(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      1,
      rhsTempVec);

    dftfe::vectorTools::createDealiiVector<double>(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      1,
      outputVec);

    rhsTempVec = 1.0;

    outputVec = 0.0;
    phiTotalSolverProblem.precondition_Jacobi(outputVec, rhsTempVec, 0.3);

    double l2NormDiag     = 0.0;
    double l2NormSqrtDiag = 0.0;

    double scalarVal = std::sqrt(4.0 * M_PI);
    for (unsigned int iNode = 0; iNode < d_diagonalA.size(); iNode++)
      {
        double diff = ((4.0 * M_PI) * d_diagonalA.data()[iNode]) -
                      outputVec.local_element(iNode);
        l2NormDiag += diff * diff;

        double diff1 = (scalarVal * d_diagonalSqrtA.data()[iNode]) -
                       std::sqrt(std::abs(outputVec.local_element(iNode)));
        l2NormSqrtDiag += diff1 * diff1;
      }

    d_blockSize = 0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::vmult(
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &Ax,
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &x,
    unsigned int                                            blockSize)
  {
    Ax.setValue(0.0);
    d_AxCellLLevelNodalData.setValue(0.0);
    x.updateGhostValues();
    d_constraintsInfo.distribute(x);

    d_basisOperationsPtr->reinit(d_blockSize,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 false,  // TODO should this be set to true
                                 false); // TODO should this be set to true
    //
    d_basisOperationsPtr->extractToCellNodalData(x,
                                                 d_xCellLLevelNodalData.data());

    for (size_type iCell = 0; iCell < d_numCells; iCell += d_cellBlockSize)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellBlockSize, d_numCells));

        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          d_blockSize,
          d_numberDofsPerElement,
          d_numberDofsPerElement,
          &d_scalarCoeffAlpha,
          d_xCellLLevelNodalData.data() +
            cellRange.first * d_numberDofsPerElement * d_blockSize,
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          d_cellStiffnessMatrixPtr->data() +
            cellRange.first * d_numberDofsPerElement * d_numberDofsPerElement,
          d_numberDofsPerElement,
          d_numberDofsPerElement * d_numberDofsPerElement,
          &d_beta,
          d_AxCellLLevelNodalData.data(),
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          cellRange.second - cellRange.first);
      }

    d_basisOperationsPtr->accumulateFromCellNodalData(
      d_AxCellLLevelNodalData.data(), Ax);
    d_constraintsInfo.distribute_slave_to_master(Ax);
    Ax.accumulateAddLocallyOwned();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::precondition_JacobiSqrt(
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
    const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
    const double                                                  omega) const
  {
    double scaleValue = (4.0 * M_PI);
    scaleValue        = std::sqrt(scaleValue);
    d_BLASWrapperPtr->stridedBlockScaleCopy(d_blockSize,
                                            d_locallyOwnedSize,
                                            scaleValue,
                                            d_diagonalSqrtA.data(),
                                            src.data(),
                                            dst.data(),
                                            d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::precondition_Jacobi(
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
    const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
    const double                                                  omega) const
  {
    double scaleValue = (4.0 * M_PI);
    d_BLASWrapperPtr->stridedBlockScaleCopy(d_blockSize,
                                            d_locallyOwnedSize,
                                            scaleValue,
                                            d_diagonalA.data(),
                                            src.data(),
                                            dst.data(),
                                            d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::setDataForRhsVec(
    dftfe::utils::MemoryStorage<double, memorySpace> &inputQuadData)
  {
    d_rhsQuadDataPtr = &inputQuadData;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::preComputeShapeFunction()
  {
    //
    // get FE data
    //
    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataPtr->n_physical_cells();

    // Quadrature for AX multiplication will FEOrderElectro+1
    const dealii::Quadrature<3> &quadratureAX =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);
    dealii::FEValues<3> fe_valuesAX(d_dofHandler->get_fe(),
                                    quadratureAX,
                                    dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values);
    const unsigned int  numberQuadraturePointsAX = quadratureAX.size();

    // Quadrature for the integration of the rhs should be higher
    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentRhs);
    dealii::FEValues<3> fe_valuesRhs(d_dofHandler->get_fe(),
                                     quadratureRhs,
                                     dealii::update_values |
                                       dealii::update_gradients |
                                       dealii::update_JxW_values);
    const unsigned int  numberDofsPerElement =
      d_dofHandler->get_fe().dofs_per_cell;
    const unsigned int numberQuadraturePointsRhs = quadratureRhs.size();

    //
    // resize data members
    //
    d_cellShapeFunctionGradientIntegral.resize(totalLocallyOwnedCells *
                                                 numberDofsPerElement *
                                                 numberDofsPerElement,
                                               0.0);
    d_cellShapeFunctionJxW.resize(totalLocallyOwnedCells *
                                  numberQuadraturePointsRhs);
    d_shapeFunctionValue.resize(numberDofsPerElement * numberDofsPerElement);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // compute cell-level shapefunctiongradientintegral generator by going over
    // dealii macrocells which allows efficient integration of cell-level matrix
    // integrals using dealii vectorized arrays
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = d_dofHandler->begin_active(),
      endc             = d_dofHandler->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_valuesRhs.reinit(cell);
          fe_valuesAX.reinit(cell);
          if (iElem == 0)
            {
              // For the reference cell initalize the shape function values
              d_shapeFunctionValue.resize(numberDofsPerElement *
                                          numberQuadraturePointsRhs);

              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                {
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsRhs;
                       ++q_point)
                    {
                      d_shapeFunctionValue[numberQuadraturePointsRhs * iNode +
                                           q_point] =
                        fe_valuesRhs.shape_value(iNode, q_point);
                    }
                }
            }

          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = 0; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  double shapeFunctionGradientValue = 0.0;
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsAX;
                       ++q_point)
                    shapeFunctionGradientValue +=
                      (fe_valuesAX.shape_grad(iNode, q_point) *
                       fe_valuesAX.shape_grad(jNode, q_point)) *
                      fe_valuesAX.JxW(q_point);

                  d_cellShapeFunctionGradientIntegral
                    [iElem * numberDofsPerElement * numberDofsPerElement +
                     numberDofsPerElement * iNode + jNode] =
                      shapeFunctionGradientValue;
                } // j node



            } // i node loop
          for (unsigned int q_point = 0; q_point < numberQuadraturePointsRhs;
               ++q_point)
            {
              d_cellShapeFunctionJxW[(iElem * numberQuadraturePointsRhs) +
                                     q_point] = fe_valuesRhs.JxW(q_point);
            }
          iElem++;
        }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::tempRhsVecCalc(
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &rhs)
  {
    //
    // get FE data
    //
    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataPtr->n_physical_cells();
    // Quadrature for the integration of the rhs should be higher
    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentRhs);
    const unsigned int numberDofsPerElement =
      d_dofHandler->get_fe().dofs_per_cell;
    const unsigned      numberQuadraturePointsRhs = quadratureRhs.size();
    std::vector<double> quadPointsValues, cellLevelJxW, cellLevelShapeFunction,
      cellLevelRhsInput;
    quadPointsValues.resize(numberQuadraturePointsRhs);
    cellLevelJxW.resize(numberQuadraturePointsRhs * numberQuadraturePointsRhs);
    cellLevelShapeFunction.resize(numberDofsPerElement *
                                  numberQuadraturePointsRhs);
    cellLevelRhsInput.resize(numberDofsPerElement * d_blockSize);
    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0;
    char               transposeMat      = 'T';
    char               doNotTransposeMat = 'N';

    // storage for precomputing index maps
    std::vector<dealii::types::global_dof_index>
      flattenedArrayMacroCellLocalProcIndexIdMap,
      flattenedArrayCellLocalProcIndexIdMap;

    vectorTools::computeCellLocalIndexSetMap(
      rhs.getMPIPatternP2P(),
      *d_matrixFreeDataPtr,
      d_matrixFreeVectorComponent,
      d_blockSize,
      flattenedArrayCellLocalProcIndexIdMap);

    d_basisOperationsPtr->reinit(d_blockSize,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 false,  // TODO should this be set to true
                                 false); // TODO should this be set to true
                                         //

    double l2ErrorIndex = 0.0;
    for (unsigned int i = 0; i < flattenedArrayCellLocalProcIndexIdMap.size();
         i++)
      {
        double diff =
          (flattenedArrayCellLocalProcIndexIdMap.data()[i] -
           d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
             .data()[i]);
        l2ErrorIndex += diff * diff;
      }


    // Calculating the rhs from the quad points
    // multiVectorInput is stored on the quad points
    unsigned int iElem = 0;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();

    // rhs += \int N_i cellLevelQuad
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          std::fill(cellLevelRhsInput.begin(), cellLevelRhsInput.end(), 0.0);

          for (unsigned int iBlock = 0; iBlock < d_blockSize; iBlock++)
            {
              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   iNode++)
                {
                  for (unsigned int jQuad = 0;
                       jQuad < numberQuadraturePointsRhs;
                       jQuad++)
                    {
                      cellLevelRhsInput[iNode * d_blockSize + iBlock] +=
                        alpha *
                        (*(d_rhsQuadDataPtr->data() +
                           iElem * numberQuadraturePointsRhs * d_blockSize +
                           jQuad * d_blockSize + iBlock)) *
                        d_shapeFunctionValue[iNode * numberQuadraturePointsRhs +
                                             jQuad] *
                        d_cellShapeFunctionJxW[iElem *
                                                 numberQuadraturePointsRhs +
                                               jQuad];
                    }
                }
            }


          // Copy to the cell level rhs to the global rhs
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              dealii::types::global_dof_index localNodeId =
                flattenedArrayCellLocalProcIndexIdMap[iElem *
                                                        numberDofsPerElement +
                                                      iNode];
              for (unsigned int iBlock = 0; iBlock < d_blockSize; iBlock++)
                {
                  *(rhs.data() + localNodeId + iBlock) +=
                    cellLevelRhsInput[d_blockSize * iNode + iBlock];
                }
            }
          iElem++;
        }
    d_constraintsInfo.distribute_slave_to_master(rhs);

    // MPI operation to sync data
    //    rhs.compress(dealii::VectorOperation::add);
    rhs.accumulateAddLocallyOwned();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<double, memorySpace> &
  MultiVectorPoissonLinearSolverProblem<memorySpace>::computeRhs(
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &NDBCVec,
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &outputVec,
    unsigned int                                            blockSizeInput)
  {
    d_basisOperationsPtr->reinit(blockSizeInput,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true,  // TODO should this be set to true
                                 true); // TODO should this be set to true
                                        //



    d_basisOperationsPtr->initializeShapeFunctionAndJacobianBasisData();
    d_basisOperationsPtr->initializeFlattenedIndexMaps();
    if (d_blockSize != blockSizeInput)
      {
        d_blockSize = blockSizeInput;
        dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                    dftfe::utils::MemorySpace::HOST>
          nodeIds, quadIds;
        nodeIds.resize(d_locallyOwnedSize);
        quadIds.resize(d_numCells * d_nQuadsPerCell);
        for (size_type i = 0; i < d_locallyOwnedSize; i++)
          {
            nodeIds.data()[i] = i * d_blockSize;
          }
        d_mapNodeIdToProcId.resize(d_locallyOwnedSize);
        d_mapNodeIdToProcId.copyFrom(nodeIds);

        for (size_type i = 0; i < d_numCells * d_nQuadsPerCell; i++)
          {
            quadIds.data()[i] = i * d_blockSize;
          }

        d_mapQuadIdToProcId.resize(d_numCells * d_nQuadsPerCell);
        d_mapQuadIdToProcId.copyFrom(quadIds);


        d_xCellLLevelNodalData.resize(d_numCells * d_numberDofsPerElement *
                                      d_blockSize);
        d_AxCellLLevelNodalData.resize(d_numCells * d_numberDofsPerElement *
                                       d_blockSize);


        d_basisOperationsPtr->createMultiVector(d_blockSize, d_rhsVec);
      }

    d_blockedXPtr    = &outputVec;
    d_blockedNDBCPtr = &NDBCVec;


    dftfe::utils::MemoryStorage<double, memorySpace> xCellLLevelNodalData,
      rhsCellLLevelNodalData;

    xCellLLevelNodalData.resize(d_numCells * d_numberDofsPerElement *
                                d_blockSize);
    rhsCellLLevelNodalData.resize(d_numCells * d_numberDofsPerElement *
                                  d_blockSize);
    //     Adding the Non homogeneous Dirichlet boundary conditions
    d_rhsVec.setValue(0.0);

    // Calculating the rhs from the quad points
    // multiVectorInput is stored on the quad points

    // Assumes that NDBC is constraints distribute is called
    // rhs  = - ( 1.0 / 4 \pi ) \int \nabla N_j \nabla N_i  d_NDBC

    d_basisOperationsPtr->extractToCellNodalData(*d_blockedNDBCPtr,
                                                 xCellLLevelNodalData.data());

    for (size_type iCell = 0; iCell < d_numCells; iCell += d_numCells)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_numCells, d_numCells));

        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          d_blockSize,
          d_numberDofsPerElement,
          d_numberDofsPerElement,
          &d_negScalarCoeffAlpha,
          xCellLLevelNodalData.data() +
            cellRange.first * d_numberDofsPerElement * d_blockSize,
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          d_cellStiffnessMatrixPtr->data() +
            cellRange.first * d_numberDofsPerElement * d_numberDofsPerElement,
          d_numberDofsPerElement,
          d_numberDofsPerElement * d_numberDofsPerElement,
          &d_beta,
          rhsCellLLevelNodalData.data(),
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          cellRange.second - cellRange.first);
      }

    d_basisOperationsPtr->accumulateFromCellNodalData(
      rhsCellLLevelNodalData.data(), d_rhsVec);


    d_basisOperationsPtr->reinit(d_blockSize,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true,   // TODO should this be set to true
                                 false); // TODO should this be set to true
                                         //


    std::pair<unsigned int, unsigned int> cellRange =
      std::make_pair(0, d_numCells);
    d_basisOperationsPtr->integrateWithBasis(d_rhsQuadDataPtr->data(),
                                             NULL,
                                             d_rhsVec,
                                             d_mapQuadIdToProcId);
    d_constraintsInfo.distribute_slave_to_master(d_rhsVec);
    d_rhsVec.accumulateAddLocallyOwned();

    return d_rhsVec;
  }

  template class MultiVectorPoissonLinearSolverProblem<
    dftfe::utils::MemorySpace::HOST>;

} // namespace dftfe
