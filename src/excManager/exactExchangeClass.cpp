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

#include "exactExchangeClass.h"
#include "vectorUtilities.h"
#include "exactExchangeClassKernel.h"

namespace dftfe
{

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  ExactExchange<ValueType, memorySpace>::ExactExchange(const bool useTucker,
                const double factorOfExactExchange,
                const MPI_Comm &mpi_comm_parent,
                const MPI_Comm &mpi_comm_domain)
    : mpi_communicator_parent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_multiVectorPoissonSolver(mpi_comm_parent,mpi_comm_domain)
    , d_multiVectorCGsolver(mpi_comm_parent,mpi_comm_domain)
  {
    d_useTucker = useTucker;
    d_factorOfExactExchange = factorOfExactExchange;
    d_blockSizeInputVector = 0;
    d_blockSizePoissonSolver = 0;
    d_VFockQuadInputValues.resize(0);
    d_alphaValuePoisson.resize(1);
    d_exchangeOperatorExists = false;
  }

template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void 
  ExactExchange<ValueType, memorySpace>::updateFockExachageFactor(double factorOfExactExchange)
  {
	  d_factorOfExactExchange = factorOfExactExchange;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
    ExactExchange<ValueType, memorySpace>::reinit(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> BLASWrapperPtr,
                                               std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> BLASWrapperHostPtr,
                                               std::shared_ptr<
                                                 dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                                 basisOperationsPtrPsi,
                                               std::shared_ptr<
                                                 dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                                                                         basisOperationsPtrElectroNoDBC,
											 dealii::MatrixFree<3, double> &matrixFreeDataPRefined,
                                               const dealii::AffineConstraints<double> &constraintMatrixPsi,
                                               std::vector<const dealii::AffineConstraints<double> *> & constraintMatrixSingleElectro,
                                               const dealii::AffineConstraints<double> &constraintMatrixSingleElectroHangingPeriodic,
                                               const dealii::AffineConstraints<double> &constraintMatrixSingleElectroHangingPeriodicHomogenous,
                                               const unsigned int                       matrixFreeVectorPsiComponent,
                                               const unsigned int                       matrixFreeQuadratureComponentPsiRhs,
                                               const unsigned int                       matrixFreeVectorElectroComponentNoDBC,
                                                const unsigned int                       matrixFreeVectorElectroComponentDBC,
                                               const unsigned int                       matrixFreeQuadratureComponentElectroRhs,
                                               const unsigned int                       matrixFreeQuadratureComponentElectroAX,
                                               const unsigned int                       numKpoints,
                                               const unsigned int                       numSpins,
                                               const unsigned int                       numWaveFunctions,
					       const dftParameters &                   dftParam)
  {
    d_BasisOperatorMemPtrPsi = basisOperationsPtrPsi;
    d_matrixFreeDataPtrPsi = &(basisOperationsPtrPsi->matrixFreeData());
    d_matrixFreePsiVectorComponent = matrixFreeVectorPsiComponent;
    d_matrixFreePsiQuadratureComponentRhs = matrixFreeQuadratureComponentPsiRhs;


    d_BasisOperatorMemPtrElectroNoDBC = basisOperationsPtrElectroNoDBC;
    d_matrixFreeDataPtrElectro = &(basisOperationsPtrElectroNoDBC->matrixFreeData());
    d_matrixFreeElectroVectorComponentNoDBC = matrixFreeVectorElectroComponentNoDBC;
    d_matrixFreeElectroVectorComponentDBC = matrixFreeVectorElectroComponentDBC;
    d_matrixFreeElectroQuadratureComponentRhs = matrixFreeQuadratureComponentElectroRhs;
    d_matrixFreeElectroQuadratureComponentAx = matrixFreeQuadratureComponentElectroAX;
    d_numKPoints = numKpoints;
    d_numSpins = numSpins;
    d_localVectorSizePsi = d_BasisOperatorMemPtrPsi->nOwnedDofs();

    d_BasisOperatorMemPtrElectroDBC = std::make_shared<
      dftfe::basis::
        FEBasisOperations<double, double, memorySpace>>(
      BLASWrapperPtr);

    std::vector<dftfe::basis::UpdateFlags> updateFlags;
    updateFlags.resize(2);
    updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                     dftfe::basis::update_gradients |
                     dftfe::basis::update_quadpoints |
                     dftfe::basis::update_transpose;

    updateFlags[1] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                     dftfe::basis::update_gradients |
                     dftfe::basis::update_quadpoints |
                     dftfe::basis::update_transpose;

    std::vector<unsigned int> quadVec;
    quadVec.resize(2);
    quadVec[0] = matrixFreeQuadratureComponentElectroRhs;
    quadVec[1] = matrixFreeQuadratureComponentElectroAX;

    // TODO this would increase the memory requirements
    d_BasisOperatorMemPtrElectroDBC->init(matrixFreeDataPRefined,
                                          constraintMatrixSingleElectro,
                      matrixFreeVectorElectroComponentDBC,
                      quadVec,
                      updateFlags);


    d_localVectorSizeElectro = d_BasisOperatorMemPtrElectroNoDBC->nOwnedDofs();
    d_numberWaveFunctions = numWaveFunctions;

    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtrPsi->get_quadrature(
        d_matrixFreePsiQuadratureComponentRhs);

    d_numberQuadraturePointsRhs = quadratureRhs.size();

    d_dftParamsPtr        = &dftParam;

    d_BLASWrapperMemPtr = BLASWrapperPtr;
    d_BLASWrapperHostPtr = BLASWrapperHostPtr;

    d_constraintMatrixPtrPsi = &constraintMatrixPsi;

    d_constraintMatrixSingleElectroHangingPeriodic = &constraintMatrixSingleElectroHangingPeriodic;

    d_constraintMatrixSingleElectroHangingPeriodicHomogenous = &constraintMatrixSingleElectroHangingPeriodicHomogenous;
      // There are 4 different constraint matrices.
      // d_constraintsMatrixOperatorDataInfo  --- stores the homogenous BC + hanging + periodic constraints of the operator.
      // This corresponds to the FEOrder and blockSize = number of wavefunctions
      // d_constraintsMatrixPsiDataInfo      ---  stores the homogenous BC + hanging + periodic constraints of the input.
      // This corresponds to the FEOrder and blockSize = blockSizeInput, which can keep changing.

    d_constraintsMatrixOperatorDataInfo.initialize(
        d_matrixFreeDataPtrPsi->get_vector_partitioner(
          d_matrixFreePsiVectorComponent),
        *d_constraintMatrixPtrPsi);

    d_ConstraintMatrixElectroHangingPeriodicInfo.initialize(
      d_matrixFreeDataPtrElectro->get_vector_partitioner(
        d_matrixFreeElectroVectorComponentNoDBC),
      *d_constraintMatrixSingleElectroHangingPeriodic);

    d_ConstraintMatrixElectroHangingPeriodicHomogenousInfo.initialize(
      d_matrixFreeDataPtrElectro->get_vector_partitioner(
        d_matrixFreeElectroVectorComponentDBC),
      *d_constraintMatrixSingleElectroHangingPeriodicHomogenous);

    // Calculation of the local Id of the boundary nodes and its distance from the center

    distributedCPUVec<double> globalToLocalVec,invDistVec;

    vectorTools::createDealiiVector<double>(
              d_matrixFreeDataPtrElectro->get_vector_partitioner(
                      d_matrixFreeElectroVectorComponentNoDBC),
              1,
              globalToLocalVec);

    
    vectorTools::createDealiiVector<double>(
              d_matrixFreeDataPtrElectro->get_vector_partitioner(
                      d_matrixFreeElectroVectorComponentNoDBC),
              1,
              invDistVec);

	     

    std::cout<<" d_matrixFreeElectroVectorComponentNoDBC = "<<d_matrixFreeElectroVectorComponentNoDBC
	    <<" d_matrixFreeElectroVectorComponentDBC = "<<d_matrixFreeElectroVectorComponentDBC<<"\n";
    //d_matrixFreeDataPtrElectro->initialize_dof_vector(invDistVec, d_matrixFreeElectroVectorComponentNoDBC);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataPtrElectro->get_vector_partitioner(d_matrixFreeElectroVectorComponentNoDBC),
      1,
      d_distVecMemSpace);
    for(unsigned  int localId = 0; localId < d_localVectorSizeElectro ; localId ++)
      {
        globalToLocalVec.local_element(localId) = localId;
        invDistVec.local_element(localId) = 0;
      }

    invDistVec.update_ghost_values();

    /*
    dealii::MappingQGeneric <3,3> mapping(d_matrixFreeDataPtrElectro->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC)
                                            .get_fe().degree);
					    */
    std::map< dealii::types::global_dof_index, dealii::Point<3,double>> dof_coords_electro ;
  
  dof_coords_electro = dealii::DoFTools::map_dofs_to_support_points<3,3>(dealii::MappingQ1<3, 3>(),
                                                       d_matrixFreeDataPtrElectro
                                                       ->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC));  
    
    /*
    dealii::DoFTools::map_dofs_to_support_points<3,3>(mappingQ1(),
                                                       d_matrixFreeDataPtrElectro
                                                       ->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC),
                                                       dof_coords_electro);
*/
    const unsigned int dofs_per_cell  = d_BasisOperatorMemPtrElectroNoDBC->nDofsPerCell();
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const unsigned int dofs_per_face  = d_BasisOperatorMemPtrElectroNoDBC->nDofsPerFace();

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(dofs_per_face);


    std::cout<<" Dofs from mf = "<<(d_matrixFreeDataPtrElectro->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC)
                                            .get_fe().degree)<<" from basis Op Electro = " <<dofs_per_cell<<"\n";


    std::cout<<"Total dofs from hand = " <<d_matrixFreeDataPtrElectro
                                                       ->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC).n_dofs()<<
						       " from basis Op = "<<d_BasisOperatorMemPtrElectroNoDBC->getDofHandler().n_dofs()<<"\n";

    std::cout<<std::flush; 
    d_numCells = d_BasisOperatorMemPtrElectroNoDBC->nCells();

    Assert(
      d_BasisOperatorMemPtrElectroNoDBC->nCells() == d_BasisOperatorMemPtrPsi->nCells(),
      dealii::ExcMessage(
        "DFT-FE Error: Number of cells in electro and psi basis op are not same."));

 
    dealii::DoFHandler<3>::active_cell_iterator cellElectro = d_BasisOperatorMemPtrElectroNoDBC->getDofHandler().begin_active(),
              endElectro = d_BasisOperatorMemPtrElectroNoDBC->getDofHandler().end();

std::vector<bool> dofs_touched(d_matrixFreeDataPtrElectro
                                                       ->get_dof_handler(d_matrixFreeElectroVectorComponentNoDBC).n_dofs(), false);
    for (; cellElectro != endElectro; ++cellElectro)
          if (cellElectro->is_locally_owned() || cellElectro->is_ghost())
          {    
        cellElectro->get_dof_indices(cellGlobalDofIndices);
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
          {
            const unsigned int boundaryId = cellElectro->face(iFace)->boundary_id();
            if (boundaryId == 0)
              {
                cellElectro->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                     ++iFaceDof)
                  {
                    const dealii::types::global_dof_index nodeId =
                      iFaceGlobalDofIndices[iFaceDof];
                    if (dofs_touched[nodeId])
                      continue;
                    dofs_touched[nodeId] = true;
                    if (!d_constraintMatrixSingleElectroHangingPeriodic->is_constrained(nodeId) && (globalToLocalVec.in_local_range(nodeId)))
                      {
			      /*
			      AssertThrow( dof_coords_electro.find(nodeId) != dof_coords_electro.end(),
      dealii::ExcMessage(
        "DFT-FE Error: Node id not found in dof_coords_electro."));
                        */
			      dealii::Point<3> p;
			     
			     try
			     {
				    p  =
                            dof_coords_electro.find(nodeId)->second;
			     } catch(const std::runtime_error& error ){ std::cout << "Error: " << error.what()<< " node id = "<<nodeId<< " end ? "<<(dof_coords_electro.find(nodeId) != dof_coords_electro.end()) << std::endl;}
			double rad = 0.0, rad1 = 0.0, rad2 = 0.0;
                        rad = p[0] * p[0] + p[1]*p[1] + p[2]*p[2];
                        rad = std::sqrt(rad);

                        if( rad < 1e-6)
                          rad = 0.0;
                        else
                          rad = 1.0/rad;

                        invDistVec[nodeId]= rad;
                      }
                  }
              }
          }
      }     // cell locally owned

    invDistVec.update_ghost_values();
    d_constraintMatrixSingleElectroHangingPeriodic->distribute(invDistVec);
    invDistVec.update_ghost_values();


    dftfe::utils::MemoryStorage<ValueType,memorySpace> invDistVecMemStorage;
    invDistVecMemStorage.resize(d_localVectorSizeElectro);

    invDistVecMemStorage.template copyFrom<dftfe::utils::MemorySpace::HOST>(invDistVec.get_values(), d_localVectorSizeElectro, 0, 0);


    invDistVecMemStorage.template copyTo<dftfe::utils::MemorySpace::HOST>(d_distVecMemSpace.data(),
           d_localVectorSizeElectro,
	   0,
	   0);
    
    d_distVecMemSpace.updateGhostValues();

    d_multiVectorPoissonSolver.reinit(
      d_BLASWrapperMemPtr,
      d_BasisOperatorMemPtrElectroDBC,
        *d_constraintMatrixSingleElectroHangingPeriodicHomogenous,
      d_matrixFreeElectroVectorComponentDBC,
        d_matrixFreeElectroQuadratureComponentRhs,
        d_matrixFreeElectroQuadratureComponentAx,
      false );//                                     isComputeMeanValueConstraint)

    quadIds.resize(d_numCells * d_numberQuadraturePointsRhs);
    d_mapQuadIdToProcId.resize(d_numCells * d_numberQuadraturePointsRhs);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExactExchange<ValueType, memorySpace>::applyPoissonExactExchangeOperator( dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                                                                                             & src,
                                                                           dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
                                                                           const unsigned int inputVecSize,
                                                                           const double factor)
  {
    const double alpha = factor*d_factorOfExactExchange;
    const unsigned int inc  = 1;

    if (d_VFockQuadInputValues.size() < d_numCells*inputVecSize*d_numberQuadraturePointsRhs)
      {
        d_VFockQuadInputValues.resize(d_numCells*inputVecSize*d_numberQuadraturePointsRhs);
      }

    computeApplyAtQuadPointsUsingPoisson(src,
                                         d_VFockQuadInputValues,
                                         d_cellLeveQuadValuesInput,
                                         inputVecSize,
                                         0, //kPointIndex,
  					 0); //spinIndex);
  
 
   //scale VFockQuadInputValues.begin with factor and 0.25;
   unsigned int VFockSize = d_numCells*inputVecSize*d_numberQuadraturePointsRhs;
   unsigned int contiguousSize = 1;
   d_alphaValuePoisson.setValue(alpha);
   d_BLASWrapperMemPtr->stridedBlockScaleColumnWise(contiguousSize, VFockSize, d_alphaValuePoisson.data(), d_VFockQuadInputValues.data());
   
   d_BasisOperatorMemPtrPsi->reinit(inputVecSize,
                                     0,
                                     d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                     true,//          isResizeTempStorageForInterpolation,
                                     true);

   d_BasisOperatorMemPtrPsi->integrateWithBasis(d_VFockQuadInputValues.begin(),
                                                 nullptr,
                                                 dst,
                                                 d_mapQuadIdToProcId);
  
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExactExchange<ValueType, memorySpace>::applyTuckerExactExchangeOperator()
  {

  }




  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
    ExactExchange<ValueType, memorySpace>::computeApplyAtQuadPointsUsingPoisson(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> & src,
    dftfe::utils::MemoryStorage<ValueType,memorySpace> &VFockQuadInputValues,
    dftfe::utils::MemoryStorage<ValueType,memorySpace> &cellLeveQuadValuesInput,
    const unsigned int inputPsiSize,
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    VFockQuadInputValues.setValue(0.0);

    const dftfe::utils::MemorySpace
            memorySpaceHostTransfer = (dftfe::utils::MemorySpace::HOST == memorySpace) ? dftfe::utils::MemorySpace::HOST :
            dftfe::utils::MemorySpace::HOST_PINNED;

    dftfe::utils::MemoryStorage<double, memorySpaceHostTransfer> total_charge;

    if (inputPsiSize != d_blockSizeInputVector)
      {
        d_blockSizeInputVector = inputPsiSize;

        for (unsigned int i = 0; i < d_numCells * d_numberQuadraturePointsRhs; i++)
          {
            quadIds.data()[i] = i * d_blockSizeInputVector;
          }
        d_mapQuadIdToProcId.copyFrom(quadIds);
      }




    char   doNotTransposeMat = 'N';
    double alpha = 1.0, beta = 0.0;
    const unsigned int inc  = 1;
    unsigned int one =1;

    unsigned int strideInputLength    = d_blockSizeInputVector ; // dftParameters::blockSizeInputLength;
    unsigned int strideOperatorLength = d_numberWaveFunctions; // dftParameters::blockSizeOperatorLength;

    unsigned int sizeofInputToPoisson = strideInputLength* strideOperatorLength, poissonBlockId = 0;

    if (d_cellLevelQuadValuesInputOperator.size() < d_numCells*d_numberQuadraturePointsRhs*strideOperatorLength*strideInputLength)
      {
        d_cellLevelQuadValuesInputOperator.resize(d_numCells*d_numberQuadraturePointsRhs*strideOperatorLength*strideInputLength);
      }
    if(d_cellLevelQuadValuesOperator.size() <  d_numCells*d_numberQuadraturePointsRhs*strideOperatorLength)
      {
        d_cellLevelQuadValuesOperator.resize(d_numCells*d_numberQuadraturePointsRhs*strideOperatorLength);
      }

    if(cellLeveQuadValuesInput.size() < d_numCells*d_numberQuadraturePointsRhs*strideInputLength)
      {
        cellLeveQuadValuesInput.resize(d_numCells*d_numberQuadraturePointsRhs*strideInputLength);
      }

    d_BasisOperatorMemPtrPsi->reinit(strideOperatorLength,
                                     0,
                                     d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                     true,//          isResizeTempStorageForInterpolation,
                                     false); //         isResizeTempStorageForCellMatrices);

    d_FockOperatorVector[0].updateGhostValues(); //d_numSpins * iKPoint + iSpin].updateGhostValues();
    d_constraintsMatrixOperatorDataInfo.distribute(d_FockOperatorVector[0]); //d_numSpins * iKPoint + iSpin]);

    d_BasisOperatorMemPtrPsi->interpolate(d_FockOperatorVector[d_numSpins * kPointIndex + spinIndex],
                                          d_cellLevelQuadValuesOperator.begin(), // *quadratureValues,
                nullptr); // *quadratureGradients)


    d_BasisOperatorMemPtrPsi->reinit(strideInputLength,
                                     0,
                                     d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                     true,//          isResizeTempStorageForInterpolation,
                                     false); //         isResizeTempStorageForCellMatrices);

    src.updateGhostValues();
    d_constraintsMatrixOperatorDataInfo.distribute(src);
    d_BasisOperatorMemPtrPsi->interpolate(src,
                                          cellLeveQuadValuesInput.begin(), // *quadratureValues,
                                          nullptr); // *quadratureGradients)



    if (d_blockSizePoissonSolver != sizeofInputToPoisson)
      {
        d_blockSizePoissonSolver = sizeofInputToPoisson ;
        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          d_matrixFreeDataPtrElectro->get_vector_partitioner(d_matrixFreeElectroVectorComponentDBC),
          strideOperatorLength*strideInputLength,
          d_multiVectorOutputDBC);

        d_boundaryValuesDBC.reinit(d_multiVectorOutputDBC);
        d_total_charge_memSpace.resize(d_blockSizePoissonSolver);

	//d_multiVectorOutputDBC.setValue(0.0);

	pcout<<" poisson solve size changed and multi vec was set to zero\n";
      }

    d_multiVectorOutputDBC.setValue(0.0);
    //d_multiVectorOutputDBC.updateGhostValues();
    //d_ConstraintMatrixElectroHangingPeriodicHomogenousInfo.distribute(d_multiVectorOutputDBC);
    total_charge.resize(d_blockSizePoissonSolver);
    d_boundaryValuesDBC.setValue(0.0);

    std::fill(total_charge.begin(),total_charge.end(),0.0);


    // set the values in d_cellLevelQuadValuesInputOperator

    dftfe::exchangeInternal::computeHadamardProductKernel(cellLeveQuadValuesInput,
                           d_cellLevelQuadValuesOperator,
                           d_cellLevelQuadValuesInputOperator,
                                                          d_numCells*d_numberQuadraturePointsRhs,
                           strideInputLength,
                           strideOperatorLength);

    d_BLASWrapperMemPtr->xgemm('N',
                               'N',
                               d_blockSizePoissonSolver,
                               one,
                               d_numCells * d_numberQuadraturePointsRhs,
                               &alpha,
                               d_cellLevelQuadValuesInputOperator.data(),
                               d_blockSizePoissonSolver,
                               d_BasisOperatorMemPtrPsi->JxW().data() ,
                               d_numCells * d_numberQuadraturePointsRhs,
                               &beta,
                               d_total_charge_memSpace.data(),
                               d_blockSizePoissonSolver);

    total_charge.copyFrom(d_total_charge_memSpace);
    // Compute the \int \psi_i \psi_j by summing over all the processors
    MPI_Allreduce(MPI_IN_PLACE,
                  &total_charge[0],
                  d_blockSizePoissonSolver,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    /*
    pcout<<" total charge \n";
    for(unsigned int iBlock = 0 ; iBlock < d_blockSizePoissonSolver; iBlock++)
    {
	    pcout<<" iBlock = "<<iBlock<<" cha = "<< total_charge[iBlock]<<"\n";
    }
    */

    d_total_charge_memSpace.copyFrom(total_charge);

    dftfe::exchangeInternal::setBoundaryValues<ValueType>(d_distVecMemSpace,
                      d_total_charge_memSpace,
                                                          d_boundaryValuesDBC,
                                                          d_localVectorSizeElectro,
                      d_blockSizePoissonSolver);


    d_boundaryValuesDBC.updateGhostValues();

    d_multiVectorPoissonSolver.setDataForRhsVec(d_cellLevelQuadValuesInputOperator);

    d_multiVectorCGsolver.solve(
      d_multiVectorPoissonSolver,
      d_BLASWrapperMemPtr,
      d_multiVectorOutputDBC,
      d_boundaryValuesDBC,
      d_localVectorSizeElectro,
      d_blockSizePoissonSolver,
      d_dftParamsPtr->absLinearSolverTolerance,
      d_dftParamsPtr->maxLinearSolverIterations,
      d_dftParamsPtr->verbosity,
      true);


    d_BasisOperatorMemPtrElectroDBC->reinit(d_blockSizePoissonSolver ,0, d_matrixFreeElectroQuadratureComponentRhs, true);

    d_cellLevelQuadValuesInputOperator.setValue(0.0);
    d_BasisOperatorMemPtrElectroDBC->interpolate(d_multiVectorOutputDBC,d_cellLevelQuadValuesInputOperator.data(), nullptr );


    dftfe::exchangeInternal::computeFockExchange(VFockQuadInputValues,
                        d_orbitalOccupancyOperatorMemSpace[0], //kPoint and iSpin set to zero
                        d_cellLevelQuadValuesOperator,
                        d_cellLevelQuadValuesInputOperator,
                                                 d_numCells*d_numberQuadraturePointsRhs,
                        strideOperatorLength,
                        strideInputLength);

//    for (unsigned  int BlockIdInputToPoisson = 0 ; BlockIdInputToPoisson < strideOperatorLength * strideInputLength ; BlockIdInputToPoisson++)
//      {
//        unsigned int OperatorId = BlockIdInputToPoisson % strideOperatorLength;
//        unsigned int inputId    = BlockIdInputToPoisson / strideOperatorLength;
//        for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsRhs;
//             ++q_point)
//          {
//            VFockQuadInputValues[iElem][q_point * d_blockSizeInputVector +
//                                        inputId + outerLoopOverInput] -=
//              d_orbitalOccupancyOperator[OperatorId + outerLoopOverOperator] *
//              d_cellLevelOperator[q_point * strideOperatorLength + OperatorId] *
//              d_cellLevelPoissonOutput[q_point * d_blockSizePoissonSolver +
//                                       BlockIdInputToPoisson];
//          }
//      }


/*
    d_BasisOperatorMemPtrPsi->integrateWithBasis(VFockQuadInputValues.begin(),
                                                 nullptr,
                                                 dst,
                                                 d_mapQuadIdToProcId);
  
 
     

    dftfe::utils::MemoryStorage<ValueType,
                                    dftfe::utils::MemorySpace::HOST> dstL2NormHost;

    dftfe::utils::MemoryStorage<ValueType, memorySpace> dstL2NormMemSpace,onesMemSpace;

    dstL2NormMemSpace.resize(strideInputLength);
    dstL2NormHost.resize(strideInputLength);

    onesMemSpace.resize(dst.locallyOwnedSize());
    onesMemSpace.setValue(1.0);
    dftfe::linearAlgebra::MultiVector<double, memorySpace> tempVec(dst, 0);
       
    d_BLASWrapperMemPtr->MultiVectorXDot(strideInputLength,
                                    dst.locallyOwnedSize(),
                                    dst.data(),
                                    dst.data(),
                                    onesMemSpace.data(),
                                    tempVec.data(),
                                    dstL2NormMemSpace.data(),
                                    mpi_communicator,
                                    dstL2NormHost.data());
  pcout<<" prinint dst after fock \n";

  for(unsigned int iBlock = 0; iBlock < strideInputLength; iBlock++)
  {
	  pcout<<" iBlock = "<<iBlock<< " dst Norm = "<<dstL2NormHost[iBlock]<<"\n";
  }
*/
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExactExchange<ValueType, memorySpace>::updateACEOperator
    (const dftfe::utils::MemoryStorage<ValueType,memorySpace> &eigenVectors,
     const std::vector<std::vector<double>> & partialOccupancies,
     const bool updateOperator)
  {
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> applyExactExchange;
    unsigned int BVec = d_numberWaveFunctions; // TODO for now all the wavefunctions are
                                               // TODO considered simultaneously
    d_blockSizeInputVector = d_numberWaveFunctions;
    for (unsigned int i = 0; i < d_numCells * d_numberQuadraturePointsRhs; i++)
      {
        quadIds.data()[i] = i * d_blockSizeInputVector;
      }
    d_mapQuadIdToProcId.copyFrom(quadIds);

    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST> M_host ;
    M_host.resize(d_numberWaveFunctions*d_numberWaveFunctions);
    M_host.setValue(0.0);

    dftfe::utils::MemoryStorage<ValueType, memorySpace> M_memSpace ;
    M_memSpace.resize(d_numberWaveFunctions*d_numberWaveFunctions);
    M_memSpace.setValue(0.0);

    double startExchangeOperatorQuad  = 0.0, endExchangeOperatorQuad  = 0.0;

    unsigned int inputPsiSize = d_numberWaveFunctions;

    d_psiInputVectorForAce.resize(d_numSpins*d_numKPoints);
    std::vector<dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>> orbitalOccupancyInputHost;
    std::vector<dftfe::utils::MemoryStorage<ValueType,memorySpace>> orbitalOccupancyInputMemSpace;
    orbitalOccupancyInputMemSpace.resize(d_numKPoints);
    orbitalOccupancyInputHost.resize(d_numKPoints);

    for( unsigned int iKPoint = 0 ; iKPoint < d_numKPoints ; iKPoint++)
      {
        orbitalOccupancyInputHost[iKPoint].resize(d_numSpins*d_numberWaveFunctions);
        for( unsigned int iSpin = 0 ; iSpin < d_numSpins ; iSpin++)
          {
            for (unsigned int jvec = 0; jvec < d_numberWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, d_numberWaveFunctions - jvec);
                dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
                  d_matrixFreeDataPtrPsi->get_vector_partitioner(d_matrixFreePsiVectorComponent),
                  currentBlockSize,
                  d_psiInputVectorForAce[d_numSpins * iKPoint + iSpin]);

                if (memorySpace == dftfe::utils::MemorySpace::HOST)
                  for (unsigned int iNode = 0; iNode < d_localVectorSizePsi;
                       ++iNode)
                    std::memcpy(d_psiInputVectorForAce[d_numSpins * iKPoint + iSpin].data() +
                                  iNode * currentBlockSize,
                                eigenVectors.data() +
                                  d_localVectorSizePsi * d_numberWaveFunctions *
                                    (d_numSpins * iKPoint + iSpin) +
                                  iNode * d_numberWaveFunctions + jvec,
                                currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                  d_BLASWrapperMemPtr->stridedCopyToBlockConstantStride(
                    currentBlockSize,
                    d_numberWaveFunctions,
                    d_localVectorSizePsi,
                    jvec,
                    eigenVectors.data() + d_localVectorSizePsi * d_numberWaveFunctions *
                                            (d_numSpins * iKPoint + iSpin),
                    d_psiInputVectorForAce[d_numSpins * iKPoint + iSpin].data());
#endif

              }
            for (unsigned int iWave = 0; iWave < d_numberWaveFunctions ; iWave ++)
              {
                orbitalOccupancyInputHost[iKPoint][d_numberWaveFunctions*iSpin + iWave] = partialOccupancies[iKPoint][d_numberWaveFunctions*iSpin + iWave];
              }

            d_psiInputVectorForAce[d_numSpins * iKPoint + iSpin].updateGhostValues();
            d_constraintsMatrixOperatorDataInfo.distribute(d_psiInputVectorForAce[d_numSpins * iKPoint + iSpin]);

            //This is not required as this is called as part of distribute.
            // d_FockOperatorVector[d_numSpins * iKPoint + iSpin].update_ghost_values();
          }
        orbitalOccupancyInputMemSpace[iKPoint].resize(orbitalOccupancyInputHost[iKPoint].size());
        orbitalOccupancyInputMemSpace[iKPoint].copyFrom(orbitalOccupancyInputHost[iKPoint]);
      }

    char   doNotTransposeMat = 'N' , doTransposeMat = 'T';
    double alpha = 1.0, alpha_minus_one = -1.0, beta = 0.0;
    const unsigned int inc        = 1;
    int rank = 0;
    char upperTriangular = 'U', lowerTriangular = 'L';
    int errorCholesky = 0;
    double tol = 1e-8;

    double startACEDgemm = 0; 
    bool needMixing = false; 
    if ( std::abs(d_dftParamsPtr->mixingParameterForExactExchange - 1.0) > 1e-3)
    {
	    needMixing = true;
    }
    if (d_exchangeOperatorExists && needMixing)
      {
        MPI_Barrier(mpi_communicator);
        startExchangeOperatorQuad = MPI_Wtime();

        dftUtils::printCurrentMemoryUsage(
          mpi_communicator, "start ACE Poisson exchange operator ");

        if ( d_VFockQuadInputValues.size() < d_numCells*d_numberWaveFunctions*d_numberQuadraturePointsRhs)
          {
            d_VFockQuadInputValues.resize(d_numCells*d_numberWaveFunctions*d_numberQuadraturePointsRhs);
          }

        d_VFockQuadInputValues.setValue(0.0);

        d_BasisOperatorMemPtrPsi->createMultiVector(
          d_numberWaveFunctions,
          applyExactExchange);
        applyExactExchange.setValue(0.0);

        computeApplyAtQuadPointsUsingPoisson(d_psiInputVectorForAce[0], //extend to kpoint and ispin
                                             d_VFockQuadInputValues,
                                             d_cellLeveQuadValuesInput,
                                             inputPsiSize,
                                             0, // kPointIndex
                                             0); //spinIndex


        d_BasisOperatorMemPtrPsi->reinit(inputPsiSize,
                                         0,
                                         d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                         true,//          isResizeTempStorageForInterpolation,
                                         true);

        d_BasisOperatorMemPtrPsi->integrateWithBasis(d_VFockQuadInputValues.begin(),
                                                     nullptr,
                                                     applyExactExchange,
                                                     d_mapQuadIdToProcId); // This is set to inputPsiSize
        d_constraintsMatrixOperatorDataInfo.distribute_slave_to_master(applyExactExchange);
        applyExactExchange.accumulateAddLocallyOwned();

        dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> singleParticleExchangeEnergy;
        computeExactE_HFQuad( d_cellLeveQuadValuesInput ,d_VFockQuadInputValues, singleParticleExchangeEnergy);

        pcout<<" exact exchange energy during ACE  = "<<d_fockEnergy<<"\n";
        MPI_Barrier(mpi_communicator);
        endExchangeOperatorQuad = MPI_Wtime();


        MPI_Barrier(mpi_communicator);
        startACEDgemm = MPI_Wtime();

        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST> M_host_old ;
        M_host_old.resize(d_numberWaveFunctions*d_numberWaveFunctions);
        M_host_old.setValue(0.0);

        dftfe::utils::MemoryStorage<ValueType, memorySpace> M_memSpace_old ;
        M_memSpace_old.resize(d_numberWaveFunctions*d_numberWaveFunctions);
        M_memSpace_old.setValue(0.0);

        //TODO check if d_FockOperatorVector needs
        //TODO distribute has to be called
        d_BLASWrapperMemPtr->xgemm(doNotTransposeMat,
                                   doTransposeMat,
                                   d_numberWaveFunctions,
                                   d_numberWaveFunctions,
                                   d_localVectorSizePsi,
                                   &alpha_minus_one,
                                   applyExactExchange.begin(),
                                   d_numberWaveFunctions,
                                   d_psiInputVectorForAce[0].begin(),
                                   d_numberWaveFunctions,
                                   &beta,
                                   M_memSpace_old.data(),
                                   d_numberWaveFunctions);

        M_host_old.copyFrom(M_memSpace_old);

        MPI_Allreduce(MPI_IN_PLACE,
                      &M_host_old[0],
                      d_numberWaveFunctions*d_numberWaveFunctions,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        if (updateOperator)
          {
            updatePoissonExactExchangeOperator(eigenVectors , partialOccupancies);
          }

        d_VFockQuadInputValues.setValue(0.0);

        applyExactExchange.setValue(0.0);
        computeApplyAtQuadPointsUsingPoisson(d_psiInputVectorForAce[0], //extend to kpoint and ispin
                                             d_VFockQuadInputValues,
                                             d_cellLeveQuadValuesInput,
                                             inputPsiSize,
                                             0, // kPointIndex
                                             0); //spinIndex


        d_BasisOperatorMemPtrPsi->reinit(inputPsiSize,
                                         0,
                                         d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                         true,//          isResizeTempStorageForInterpolation,
                                         true);

        d_BasisOperatorMemPtrPsi->integrateWithBasis(d_VFockQuadInputValues.begin(),
                                                     nullptr,
                                                     applyExactExchange,
                                                     d_mapQuadIdToProcId); // This is set to inputPsiSize
        d_constraintsMatrixOperatorDataInfo.distribute_slave_to_master(applyExactExchange);
        applyExactExchange.accumulateAddLocallyOwned();

        computeExactE_HFQuad( d_cellLeveQuadValuesInput ,d_VFockQuadInputValues, singleParticleExchangeEnergy);

        pcout<<" exact exchange energy during ACE  = "<<d_fockEnergy<<"\n";
        MPI_Barrier(mpi_communicator);
        endExchangeOperatorQuad = MPI_Wtime();

        MPI_Barrier(mpi_communicator);
        startACEDgemm = MPI_Wtime();

        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST> M_host_new ;
        M_host_new.resize(d_numberWaveFunctions*d_numberWaveFunctions);
        M_host_new.setValue(0.0);

        dftfe::utils::MemoryStorage<ValueType, memorySpace> M_memSpace_new ;
        M_memSpace_new.resize(d_numberWaveFunctions*d_numberWaveFunctions);
        M_memSpace_new.setValue(0.0);

        //TODO check if d_FockOperatorVector needs
        //TODO distribute has to be called
        d_BLASWrapperMemPtr->xgemm(doNotTransposeMat,
                                   doTransposeMat,
                                   d_numberWaveFunctions,
                                   d_numberWaveFunctions,
                                   d_localVectorSizePsi,
                                   &alpha_minus_one,
                                   applyExactExchange.begin(),
                                   d_numberWaveFunctions,
                                   d_psiInputVectorForAce[0].begin(),
                                   d_numberWaveFunctions,
                                   &beta,
                                   M_memSpace_new.data(),
                                   d_numberWaveFunctions);

        M_host_new.copyFrom(M_memSpace_new);

        MPI_Allreduce(MPI_IN_PLACE,
                      &M_host_new[0],
                      d_numberWaveFunctions*d_numberWaveFunctions,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        for ( unsigned int iWave = 0; iWave < d_numberWaveFunctions*d_numberWaveFunctions;
             iWave++)
          {
            M_host[iWave] = (d_dftParamsPtr->mixingParameterForExactExchange)*M_host_new[iWave] +
                            (1.0 - d_dftParamsPtr->mixingParameterForExactExchange)*M_host_old[iWave];
          }

      }
    else
      {
        if (updateOperator)
          {
            updatePoissonExactExchangeOperator(eigenVectors , partialOccupancies);
          }


        MPI_Barrier(mpi_communicator);
        startExchangeOperatorQuad = MPI_Wtime();

        dftUtils::printCurrentMemoryUsage(
          mpi_communicator, "start ACE Poisson exchange operator ");

        if ( d_VFockQuadInputValues.size() < d_numCells*d_numberWaveFunctions*d_numberQuadraturePointsRhs)
          {
            d_VFockQuadInputValues.resize(d_numCells*d_numberWaveFunctions*d_numberQuadraturePointsRhs);
          }

        d_VFockQuadInputValues.setValue(0.0);

        d_BasisOperatorMemPtrPsi->createMultiVector(
          d_numberWaveFunctions,
          applyExactExchange);
        applyExactExchange.setValue(0.0);

        computeApplyAtQuadPointsUsingPoisson(d_psiInputVectorForAce[0], //extend to kpoint and ispin
                                             d_VFockQuadInputValues,
                                             d_cellLeveQuadValuesInput,
                                             inputPsiSize,
                                             0, // kPointIndex
                                             0); //spinIndex


        d_BasisOperatorMemPtrPsi->reinit(inputPsiSize,
                                         0,
                                         d_matrixFreePsiQuadratureComponentRhs, // this should correspond to the dft class value
                                         true,//          isResizeTempStorageForInterpolation,
                                         true);

        d_BasisOperatorMemPtrPsi->integrateWithBasis(d_VFockQuadInputValues.begin(),
                                                     nullptr,
                                                     applyExactExchange,
                                                     d_mapQuadIdToProcId); // This is set to inputPsiSize
        d_constraintsMatrixOperatorDataInfo.distribute_slave_to_master(applyExactExchange);
        applyExactExchange.accumulateAddLocallyOwned();

        dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> singleParticleExchangeEnergy;
        computeExactE_HFQuad( d_cellLeveQuadValuesInput ,d_VFockQuadInputValues, singleParticleExchangeEnergy);

        pcout<<" exact exchange energy during ACE  = "<<d_fockEnergy<<"\n";
        MPI_Barrier(mpi_communicator);
        endExchangeOperatorQuad = MPI_Wtime();


        MPI_Barrier(mpi_communicator);
        startACEDgemm = MPI_Wtime();

        //TODO check if d_FockOperatorVector needs
        //TODO distribute has to be called
        d_BLASWrapperMemPtr->xgemm(doNotTransposeMat,
                                   doTransposeMat,
                                   d_numberWaveFunctions,
                                   d_numberWaveFunctions,
                                   d_localVectorSizePsi,
                                   &alpha_minus_one,
                                   applyExactExchange.begin(),
                                   d_numberWaveFunctions,
                                   d_psiInputVectorForAce[0].begin(),
                                   d_numberWaveFunctions,
                                   &beta,
                                   M_memSpace.data(),
                                   d_numberWaveFunctions);

        M_host.copyFrom(M_memSpace);

        MPI_Allreduce(MPI_IN_PLACE,
                      &M_host[0],
                      d_numberWaveFunctions*d_numberWaveFunctions,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
      }


    MPI_Barrier(mpi_communicator);
    double endACEDgemm = MPI_Wtime();

    
    for (unsigned int iWave = 0; iWave < d_numberWaveFunctions ; iWave ++)
      {
        pcout<<" iWave = "<< iWave<<"diag val = "<< M_host[(iWave) *(d_numberWaveFunctions+1) ]<<"\n";
      }

    dpotrf_(&lowerTriangular,
            &d_numberWaveFunctions,
            &M_host[0],
            &d_numberWaveFunctions,
            &errorCholesky);

    for (unsigned int iWave = 0; iWave < d_numberWaveFunctions ; iWave ++)
      {
        pcout<<" iWave = "<< iWave<<"Eigen val = "<< M_host[(iWave) *(d_numberWaveFunctions+1) ]<<"\n";
      }


    MPI_Barrier(mpi_communicator);
    double endChol = MPI_Wtime();

    for (unsigned int iWave = 0; iWave < d_numberWaveFunctions ; iWave ++)
      {
        for (unsigned int jWave = iWave + 1; jWave < d_numberWaveFunctions ; jWave ++)
          {
            M_host[iWave + jWave *d_numberWaveFunctions ] = 0.0;
          }
      }
    MPI_Barrier(mpi_communicator);
    double endZero = MPI_Wtime();

    pcout<<" errorCholesky = "<< errorCholesky<<"\n";
    AssertThrow(
      errorCholesky == 0,
      dealii::ExcMessage(
        "DFT-FE Error: Cholesky error in ACE computation \n"));

    // Invert the upper triangular matrix

    int errorInvertion = 0;

    char nonUnitTriangular = 'N';

    MPI_Barrier(mpi_communicator);
    double endInv = MPI_Wtime();

    dtrtri_(&lowerTriangular,
           &nonUnitTriangular,
           &d_numberWaveFunctions,
           &M_host[0],
           &d_numberWaveFunctions,
           &errorInvertion );

    AssertThrow(
      errorInvertion == 0,
      dealii::ExcMessage(
        "DFT-FE Error:  Error during inversion of Matrix in ACE computation \n"));

    // TODO resize VFockQuadInputValues to zero, if running out of memory
    d_BasisOperatorMemPtrPsi->createMultiVector(
      d_numberWaveFunctions,
      d_ACEOperatorVecMemSpace);
    d_ACEOperatorVecMemSpace.setValue(0.0);

    int errorSolve = 0;

    M_memSpace.copyFrom(M_host);

          d_BLASWrapperMemPtr->xgemm(doNotTransposeMat,
                 doNotTransposeMat,
                 d_numberWaveFunctions,
                 d_localVectorSizePsi,
                 d_numberWaveFunctions,
                 &alpha,
                 M_memSpace.data(),
                 d_numberWaveFunctions,
                 applyExactExchange.begin(),
                 d_numberWaveFunctions,
                 &beta,
                 d_ACEOperatorVecMemSpace.begin(),
                 d_numberWaveFunctions);

    d_ACEOperatorVecMemSpace.updateGhostValues();

    d_exchangeOperatorExists = true;
    MPI_Barrier(mpi_communicator);
    double endDgemm2 = MPI_Wtime();

    pcout<<" Time taken for computing discrete exchange = "<<endExchangeOperatorQuad - startExchangeOperatorQuad<<"  dgemm1 = "<< endACEDgemm - startACEDgemm
          <<"Chol = "<<endChol-  endACEDgemm<<"Zeroing  = "<< endZero- endChol <<" Inv = "<<endInv - endChol<<" dgemm2 = "<<endDgemm2 - endInv<<"\n";


  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExactExchange<ValueType, memorySpace>:: applyACEOperator(const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
                                                          dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>      &dst,
                                                          const unsigned int inputPsiSize,
                                                          const double factor)
  {
    d_blockSizeInputVector = inputPsiSize ;

    d_mMatrixMemSpace.resize(d_blockSizeInputVector *d_numberWaveFunctions);
    d_mMatrixMemSpace.setValue(0.0);

    char doNotTransposeMat = 'N';
    char doTransposeMat    = 'T';
    double alpha_minus_one = -1.0*factor*d_factorOfExactExchange, alpha = 1.0, betaZero = 0.0, betaOne = 1.0;

    d_BLASWrapperMemPtr->xgemm(
      'N',
      'T',
      d_blockSizeInputVector,
      d_numberWaveFunctions,
      d_localVectorSizePsi,
      &alpha,
      src.begin(),
      d_blockSizeInputVector,
      d_ACEOperatorVecMemSpace.begin(),
      d_numberWaveFunctions,
      &betaZero,
      d_mMatrixMemSpace.data(),
      d_blockSizeInputVector);

    d_mMatrixHost.resize(d_mMatrixMemSpace.size());
    d_mMatrixHost.copyFrom(d_mMatrixMemSpace);

    MPI_Allreduce(MPI_IN_PLACE,
                  &d_mMatrixHost[0],
                  d_blockSizeInputVector*d_numberWaveFunctions,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    d_mMatrixMemSpace.copyFrom(d_mMatrixHost);

    d_BLASWrapperMemPtr->xgemm(
      'N',
      'N',
      d_blockSizeInputVector,
      d_localVectorSizePsi,
      d_numberWaveFunctions,
      &alpha_minus_one,
      d_mMatrixMemSpace.begin(),
      d_blockSizeInputVector,
      d_ACEOperatorVecMemSpace.begin(),
      d_numberWaveFunctions,
      &betaOne,
      dst.data(),
      d_blockSizeInputVector);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
    ExactExchange<ValueType, memorySpace>::updatePoissonExactExchangeOperator(
    const dftfe::utils::MemoryStorage<ValueType,memorySpace> &eigenVectors,
    const std::vector<std::vector<double>> & partialOccupancies)
  {
    pcout<<"Entering the update operator psi bar \n";

    d_orbitalOccupancyOperatorHost.resize(d_numKPoints);

    d_orbitalOccupancyOperatorMemSpace.resize(d_numKPoints);
    unsigned int BVec = d_numberWaveFunctions; // TODO for now all the wavefunctions are
                                               // TODO considered simultaneously

    d_FockOperatorVector.resize(d_numSpins*d_numKPoints);
    for( unsigned int iKPoint = 0 ; iKPoint < d_numKPoints ; iKPoint++)
      {
        d_orbitalOccupancyOperatorHost[iKPoint].resize(d_numSpins*d_numberWaveFunctions);
        for( unsigned int iSpin = 0 ; iSpin < d_numSpins ; iSpin++)
          {
            for (unsigned int jvec = 0; jvec < d_numberWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, d_numberWaveFunctions - jvec);
                dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
                  d_matrixFreeDataPtrPsi->get_vector_partitioner(d_matrixFreePsiVectorComponent),
                  currentBlockSize,
                  d_FockOperatorVector[d_numSpins * iKPoint + iSpin]);

                if (memorySpace == dftfe::utils::MemorySpace::HOST)
                  for (unsigned int iNode = 0; iNode < d_localVectorSizePsi;
                       ++iNode)
                    std::memcpy(d_FockOperatorVector[d_numSpins * iKPoint + iSpin].data() +
                                  iNode * currentBlockSize,
                                eigenVectors.data() +
                                  d_localVectorSizePsi * d_numberWaveFunctions *
                                    (d_numSpins * iKPoint + iSpin) +
                                  iNode * d_numberWaveFunctions + jvec,
                                currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                  d_BLASWrapperMemPtr->stridedCopyToBlockConstantStride(
                    currentBlockSize,
                    d_numberWaveFunctions,
                    d_localVectorSizePsi,
                    jvec,
                    eigenVectors.data() + d_localVectorSizePsi * d_numberWaveFunctions *
                                  (d_numSpins * iKPoint + iSpin),
                    d_FockOperatorVector[d_numSpins * iKPoint + iSpin].data());
#endif

              }
            for (unsigned int iWave = 0; iWave < d_numberWaveFunctions ; iWave ++)
              {
                d_orbitalOccupancyOperatorHost[iKPoint][d_numberWaveFunctions*iSpin + iWave] = partialOccupancies[iKPoint][d_numberWaveFunctions*iSpin + iWave];
              }

            d_FockOperatorVector[d_numSpins * iKPoint + iSpin].updateGhostValues();
            d_constraintsMatrixOperatorDataInfo.distribute(d_FockOperatorVector[d_numSpins * iKPoint + iSpin]);

            //This is not required as this is called as part of distribute.
            // d_FockOperatorVector[d_numSpins * iKPoint + iSpin].update_ghost_values();
          }
        d_orbitalOccupancyOperatorMemSpace[iKPoint].resize(d_orbitalOccupancyOperatorHost[iKPoint].size());
        d_orbitalOccupancyOperatorMemSpace[iKPoint].copyFrom(d_orbitalOccupancyOperatorHost[iKPoint]);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  ExactExchange<ValueType, memorySpace>::computeExactE_HFQuad(
    const dftfe::utils::MemoryStorage<ValueType,memorySpace> &wavefunctionInpuVec,
    dftfe::utils::MemoryStorage<ValueType,memorySpace> &VFockQuadInputValues,
    dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & singleParticleExchangeEnergy)
  {
    d_BLASWrapperMemPtr->hadamardProduct(
      d_numCells*d_numberQuadraturePointsRhs*d_numberWaveFunctions,
      wavefunctionInpuVec.data(),
      VFockQuadInputValues.data(),
      d_cellLevelQuadValuesOperator.data());

    pcout<<" Size of wavefunctionInpuVec = "<<wavefunctionInpuVec.size()<<" VFOck = "<<VFockQuadInputValues.size()<<" d_cellLevelQuadValuesOperator = "<<d_cellLevelQuadValuesOperator.size()<< " JxW = "<<d_BasisOperatorMemPtrPsi->JxW().size()<<"\n";

    dftfe::utils::MemoryStorage<ValueType,memorySpace> singleParticleExchangeEnergyMemSpace;
    singleParticleExchangeEnergyMemSpace.resize(d_numberWaveFunctions);
    double oneDouble = 1.0;
    unsigned int one = 1;
    double beta = 0.0;
    /*
    d_BLASWrapperMemPtr->xgemm('N',
                          'T',
                          one,
                          d_numberWaveFunctions,
                          d_numCells*d_numberQuadraturePointsRhs,
                          &oneDouble,
                          d_BasisOperatorMemPtrPsi->JxW().data(),
                          one,
                          d_cellLevelQuadValuesOperator.data(),
                          d_numberWaveFunctions,
                          &beta,
                          singleParticleExchangeEnergyMemSpace.data(),
                          one);
*/
    d_BLASWrapperMemPtr->xgemm('N',
                          'N',
                          d_numberWaveFunctions,
                          one,
                          d_numCells*d_numberQuadraturePointsRhs,
                          &oneDouble,
                          d_cellLevelQuadValuesOperator.data(),
                          d_numberWaveFunctions,
                          d_BasisOperatorMemPtrPsi->JxW().data(),
                          d_numCells*d_numberQuadraturePointsRhs,
                          &beta,
                          singleParticleExchangeEnergyMemSpace.data(),
                          d_numberWaveFunctions);

    singleParticleExchangeEnergy.resize(singleParticleExchangeEnergyMemSpace.size());
    singleParticleExchangeEnergy.copyFrom(singleParticleExchangeEnergyMemSpace);

    MPI_Allreduce(MPI_IN_PLACE,
                  singleParticleExchangeEnergy.data(),
                  d_numberWaveFunctions,
                  dataTypes::mpi_type_id(singleParticleExchangeEnergy.data()),
                  MPI_SUM,
                  mpi_communicator);

/*
    pcout<<" single particle energy = \n";
    for( unsigned int iBlock = 0 ;iBlock < d_numberWaveFunctions; iBlock++)
    {
	    pcout<<" iBlock = "<<iBlock<<" single energy = "<<singleParticleExchangeEnergy.data()[iBlock]<<"\n";
    }
    */

    d_fockEnergy = 0;
    for (unsigned int iWave = 0; iWave < d_numberWaveFunctions;
         iWave++)
      {
        d_fockEnergy +=
          d_orbitalOccupancyOperatorHost[0][iWave] * singleParticleExchangeEnergy[iWave];

        //TODO extend to kpoints and spin
      }

    return d_fockEnergy;
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  ExactExchange<ValueType,memorySpace>::getExpectationOfExactExchangeOperator()
  {
    Assert(
      true,
      dealii::ExcMessage(
        "DFT-FE Error: getExpectationOfExactExchangeOperator not defined properly."));
    return 0;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  ExactExchange<ValueType,memorySpace>::getExactExchangeEnergy()
  {
    return d_fockEnergy;
  }

    template class ExactExchange<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class ExactExchange<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif

}
