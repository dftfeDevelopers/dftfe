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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <kohnShamDFTOperator.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <vectorUtilities.h>


namespace dftfe
{
#include "computeLocalAndNonLocalHamiltonianTimesX.cc"
#include "computeNonLocalProjectorKetTimesXTimesV.cc"
#include "computeNonLocalHamiltonianTimesXMemoryOpt.cc"
#include "hamiltonianMatrixCalculator.cc"
#include "matrixVectorProductImplementations.cc"
#include "shapeFunctionDataCalculator.cc"


  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::kohnShamDFTOperatorClass(
    dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_parent,
    const MPI_Comm &                   mpi_comm_domain)
    : dftPtr(_dftPtr)
    , d_kPointIndex(0)
    , d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell(
        dftPtr->d_densityDofHandlerIndex))
    , d_numberCellsLocallyOwned(_dftPtr->matrix_free_data.n_physical_cells())
    , d_isStiffnessMatrixExternalPotCorrComputed(false)
    , d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , operatorDFTClass(mpi_comm_domain,
                       _dftPtr->getMatrixFreeData(),
                       _dftPtr->constraintsNoneDataInfo)
  {}


  //
  // initialize kohnShamDFTOperatorClass object
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::init()
  {
    computing_timer.enter_subsection("kohnShamDFTOperatorClass setup");


    dftPtr->matrix_free_data.initialize_dof_vector(
      d_invSqrtMassVector, dftPtr->d_densityDofHandlerIndex);
    d_sqrtMassVector.reinit(d_invSqrtMassVector);



    //
    // compute mass vector
    //
    computeMassVector(dftPtr->dofHandler,
                      dftPtr->constraintsNone,
                      d_sqrtMassVector,
                      d_invSqrtMassVector);


    operatorDFTClass::setInvSqrtMassVector(d_invSqrtMassVector);

    d_cellHamiltonianMatrix.clear();
    d_cellHamiltonianMatrix.resize(dftPtr->d_kPointWeights.size() *
                                   (1 + dftPtr->d_dftParamsPtr->spinPolarized));


    vectorTools::classifyInteriorSurfaceNodesInCell(
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      d_nodesPerCellClassificationMap);

    vectorTools::classifyInteriorSurfaceNodesInGlobalArray(
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      dftPtr->constraintsNone,
      d_nodesPerCellClassificationMap,
      d_globalArrayClassificationMap);

    computing_timer.leave_subsection("kohnShamDFTOperatorClass setup");
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::resetExtPotHamFlag()
  {
    d_isStiffnessMatrixExternalPotCorrComputed = false;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int                    numberWaveFunctions,
    distributedCPUVec<dataTypes::number> &flattenedArray,
    bool                                  flag)
  {
    if (flag)
      {
        vectorTools::createDealiiVector<dataTypes::number>(
          dftPtr->matrix_free_data.get_vector_partitioner(),
          numberWaveFunctions,
          flattenedArray);
      }
    distributedCPUMultiVec<dataTypes::number> tempFlattenedArray;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      dftPtr->matrix_free_data.get_vector_partitioner(),
      numberWaveFunctions,
      tempFlattenedArray);

    if (dftPtr->d_dftParamsPtr->isPseudopotential)
      {
        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
          numberWaveFunctions,
          dftPtr->d_projectorKetTimesVectorParFlattened);
      }



    vectorTools::computeCellLocalIndexSetMap(
      tempFlattenedArray.getMPIPatternP2P(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_flattenedArrayMacroCellLocalProcIndexIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap);



    vectorTools::computeCellLocalIndexSetMap(
      tempFlattenedArray.getMPIPatternP2P(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_FullflattenedArrayMacroCellLocalProcIndexIdMap,
      d_normalCellIdToMacroCellIdMap,
      d_macroCellIdToNormalCellIdMap,
      d_FullflattenedArrayCellLocalProcIndexIdMap);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int                         numberWaveFunctions,
    distributedCPUMultiVec<dataTypes::number> &flattenedArray,
    bool                                       flag)
  {
    if (flag)
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        dftPtr->matrix_free_data.get_vector_partitioner(),
        numberWaveFunctions,
        flattenedArray);

    if (dftPtr->d_dftParamsPtr->isPseudopotential)
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
        numberWaveFunctions,
        dftPtr->d_projectorKetTimesVectorParFlattened);



    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.getMPIPatternP2P(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_flattenedArrayMacroCellLocalProcIndexIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap);



    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.getMPIPatternP2P(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_FullflattenedArrayMacroCellLocalProcIndexIdMap,
      d_normalCellIdToMacroCellIdMap,
      d_macroCellIdToNormalCellIdMap,
      d_FullflattenedArrayCellLocalProcIndexIdMap);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int numberWaveFunctions)
  {
    if (dftPtr->d_dftParamsPtr->isPseudopotential)
      {
        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
          numberWaveFunctions,
          dftPtr->d_projectorKetTimesVectorParFlattened);
      }
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::initCellWaveFunctionMatrix(
    const unsigned int                         numberWaveFunctions,
    distributedCPUMultiVec<dataTypes::number> &src,
    std::vector<dataTypes::number> &           cellWaveFunctionMatrix)
  {
    cellWaveFunctionMatrix.resize(d_numberCellsLocallyOwned *
                                    d_numberNodesPerElement *
                                    numberWaveFunctions,
                                  0.0);

    const unsigned int inc = 1;
    for (unsigned int iElem = 0; iElem < d_numberCellsLocallyOwned; ++iElem)
      {
        for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
          {
            dealii::types::global_dof_index localNodeId =
              d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
#ifdef USE_COMPLEX
            zcopy_(&numberWaveFunctions,
                   src.data() + localNodeId,
                   &inc,
                   &cellWaveFunctionMatrix[d_numberNodesPerElement *
                                             numberWaveFunctions * iElem +
                                           numberWaveFunctions * iNode],
                   &inc);

#else
            dcopy_(&numberWaveFunctions,
                   src.data() + localNodeId,
                   &inc,
                   &cellWaveFunctionMatrix[d_numberNodesPerElement *
                                             numberWaveFunctions * iElem +
                                           numberWaveFunctions * iNode],
                   &inc);
#endif
          }
      }
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    fillGlobalArrayFromCellWaveFunctionMatrix(
      const unsigned int                         numberWaveFunctions,
      const std::vector<dataTypes::number> &     cellWaveFunctionMatrix,
      distributedCPUMultiVec<dataTypes::number> &glbArray)

  {
    const unsigned int inc = 1;
    for (unsigned int iElem = 0; iElem < d_numberCellsLocallyOwned; ++iElem)
      {
        for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
          {
            if (d_nodesPerCellClassificationMap[iNode] == 0)
              {
                dealii::types::global_dof_index localNodeId =
                  d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
#ifdef USE_COMPLEX
                zcopy_(&numberWaveFunctions,
                       &cellWaveFunctionMatrix[d_numberNodesPerElement *
                                                 numberWaveFunctions * iElem +
                                               numberWaveFunctions * iNode],
                       &inc,
                       glbArray.data() + localNodeId,
                       &inc);
#else
                dcopy_(&numberWaveFunctions,
                       &cellWaveFunctionMatrix[d_numberNodesPerElement *
                                                 numberWaveFunctions * iElem +
                                               numberWaveFunctions * iNode],
                       &inc,
                       glbArray.data() + localNodeId,
                       &inc);
#endif
              }
          }
      }
  }



  // Y = a*X + Y
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::initWithScalar(
    const unsigned int              numberWaveFunctions,
    double                          scalarValue,
    std::vector<dataTypes::number> &cellWaveFunctionMatrix)

  {
    unsigned int numberCells = dftPtr->matrix_free_data.n_physical_cells();
    cellWaveFunctionMatrix.resize(numberCells * d_numberNodesPerElement *
                                    numberWaveFunctions,
                                  scalarValue);
  }



  // Y = a*X + b*Y
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::axpby(
    double                                scalarA,
    double                                scalarB,
    const unsigned int                    numberWaveFunctions,
    const std::vector<dataTypes::number> &cellXWaveFunctionMatrix,
    std::vector<dataTypes::number> &      cellYWaveFunctionMatrix)
  {
    unsigned int iElem = 0;
    unsigned int productNumNodesWaveFunctions =
      d_numberNodesPerElement * numberWaveFunctions;
    for (unsigned int iElem = 0; iElem < d_numberCellsLocallyOwned; ++iElem)
      {
        unsigned int indexTemp = productNumNodesWaveFunctions * iElem;
        for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
          {
            if (d_nodesPerCellClassificationMap[iNode] == 0)
              {
                unsigned int indexVal = indexTemp + numberWaveFunctions * iNode;
                for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                     ++iWave)
                  {
                    cellYWaveFunctionMatrix[indexVal + iWave] =
                      scalarB * cellYWaveFunctionMatrix[indexVal + iWave] +
                      scalarA * cellXWaveFunctionMatrix[indexVal + iWave];
                  }
              }
          }
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getInteriorSurfaceNodesMapFromGlobalArray(
      std::vector<unsigned int> &globalArrayClassificationMap)

  {
    globalArrayClassificationMap = d_globalArrayClassificationMap;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<dealii::types::global_dof_index> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getFlattenedArrayCellLocalProcIndexIdMap() const
  {
    return d_FullflattenedArrayCellLocalProcIndexIdMap;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedCPUMultiVec<dataTypes::number> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getParallelProjectorKetTimesBlockVector()
  {
    return dftPtr->d_projectorKetTimesVectorParFlattened;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesDensityGaussQuad() const
  {
    return d_densityGaussQuadShapeFunctionValues;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradValuesDensityGaussQuad() const
  {
    return d_densityGaussQuadShapeFunctionGradientValues;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesDensityGaussLobattoQuad() const
  {
    return d_densityGlQuadShapeFunctionValues;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesDensityTransposed() const
  {
    return d_shapeFunctionValueDensityTransposed;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesNLPTransposed() const
  {
    return d_shapeFunctionValueNLPTransposed;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesNLPTransposed() const
  {
    return d_shapeFunctionGradientValueNLPTransposed;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const std::vector<double> &
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::getInverseJacobiansNLP()
    const
  {
    return d_inverseJacobiansNLP;
  }



  //
  // compute mass Vector
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeMassVector(
    const dealii::DoFHandler<3> &            dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrix,
    distributedCPUVec<double> &              sqrtMassVec,
    distributedCPUVec<double> &              invSqrtMassVec)
  {
    computing_timer.enter_subsection("kohnShamDFTOperatorClass Mass assembly");
    invSqrtMassVec = 0.0;
    sqrtMassVec    = 0.0;

    dealii::QGaussLobatto<3> quadrature(FEOrder + 1);
    dealii::FEValues<3>      fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int     dofs_per_cell = (dofHandler.get_fe()).dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          constraintMatrix.distribute_local_to_global(massVectorLocal,
                                                      local_dof_indices,
                                                      invSqrtMassVec);
        }

    invSqrtMassVec.compress(dealii::VectorOperation::add);


    for (dealii::types::global_dof_index i = 0; i < invSqrtMassVec.size(); ++i)
      if (invSqrtMassVec.in_local_range(i) &&
          !constraintMatrix.is_constrained(i))
        {
          if (std::abs(invSqrtMassVec(i)) > 1.0e-15)
            {
              sqrtMassVec(i)    = std::sqrt(invSqrtMassVec(i));
              invSqrtMassVec(i) = 1.0 / std::sqrt(invSqrtMassVec(i));
            }
          AssertThrow(
            !std::isnan(invSqrtMassVec(i)),
            dealii::ExcMessage(
              "Value of inverse square root of mass matrix on the unconstrained node is undefined"));
        }

    invSqrtMassVec.compress(dealii::VectorOperation::insert);
    sqrtMassVec.compress(dealii::VectorOperation::insert);
    computing_timer.leave_subsection("kohnShamDFTOperatorClass Mass assembly");
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinitkPointSpinIndex(
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values);
    const int           numberQuadraturePoints = quadrature_formula.size();


    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);

    // allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(numberQuadraturePoints);
    std::vector<double> corrPotentialVal(numberQuadraturePoints);
    std::vector<double> densityValue(numberQuadraturePoints);

    //
    // loop over cell block
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr    = dftPtr->dofHandler.begin_active(),
      endcellPtr = dftPtr->dofHandler.end();

    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              densityValue[q] =
                rhoValues[0][iElemCount * numberQuadraturePoints + q];

            const double *tempPhi =
              phiValues.data() + iElemCount * numberQuadraturePoints;


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  densityValue[q] += temp2[q];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &exchangePotentialVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &corrPotentialVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhi[q] + exchangePotentialVal[q] + corrPotentialVal[q]) *
                  fe_values.JxW(q);
              }

            iElemCount++;
          }
      }


    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradRhoValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_inverse_jacobians |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                                    numberQuadraturePoints * 3,
                                                  0.0);

    // allocate storage for exchange potential
    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> gradDensityValue(3 * numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              densityValue[q] =
                rhoValues[0][iElemCount * numberQuadraturePoints + q];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                gradDensityValue[3 * q + iDim] =
                  rhoValues[0][iElemCount * numberQuadraturePoints * 3 + 3 * q +
                               iDim];


            const double *tempPhi =
              phiValues.data() + iElemCount * numberQuadraturePoints;


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[q] += temp2[q];
                    gradDensityValue[3 * q + 0] += temp3[3 * q + 0];
                    gradDensityValue[3 * q + 1] += temp3[3 * q + 1];
                    gradDensityValue[3 * q + 2] += temp3[3 * q + 2];
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX = gradDensityValue[3 * q + 0];
                const double gradRhoY = gradDensityValue[3 * q + 1];
                const double gradRhoZ = gradDensityValue[3 * q + 2];
                sigmaValue[q] = gradRhoX * gradRhoX + gradRhoY * gradRhoY +
                                gradRhoZ * gradRhoZ;
              }
            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigmaVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigmaVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhi[q] + derExchEnergyWithDensityVal[q] +
                   derCorrEnergyWithDensityVal[q]) *
                  fe_values.JxW(q);
              }


            // Rethink about this
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw      = fe_values.JxW(q);
                const double gradRhoX = gradDensityValue[3 * q + 0];
                const double gradRhoY = gradDensityValue[3 * q + 1];
                const double gradRhoZ = gradDensityValue[3 * q + 2];
                const double term =
                  derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][0][0] * gradRhoX +
                   inverseJacobians[q][0][1] * gradRhoY +
                   inverseJacobians[q][0][2] * gradRhoZ) *
                  term * jxw;
                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][1][0] * gradRhoX +
                   inverseJacobians[q][1][1] * gradRhoY +
                   inverseJacobians[q][1][2] * gradRhoZ) *
                  term * jxw;
                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][2][0] * gradRhoX +
                   inverseJacobians[q][2][1] * gradRhoY +
                   inverseJacobians[q][2][2] * gradRhoZ) *
                  term * jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }


#ifdef USE_COMPLEX
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUMultiVec<std::complex<double>> &src,
    const unsigned int                            numberWaveFunctions,
    const bool                                    scaleFlag,
    const double                                  scalar,
    distributedCPUMultiVec<std::complex<double>> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)


  {
    const unsigned int numberDofs = src.locallyOwnedSize();
    const unsigned int inc        = 1;

    //
    // scale src vector with M^{-1/2}
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff =
          d_invSqrtMassVector.local_element(i) * scalar;
        zdscal_(&numberWaveFunctions,
                &scalingCoeff,
                src.data() + i * numberWaveFunctions,
                &inc);
      }


    if (scaleFlag)
      {
        for (int i = 0; i < numberDofs; ++i)
          {
            const double scalingCoeff = d_sqrtMassVector.local_element(i);
            zdscal_(&numberWaveFunctions,
                    &scalingCoeff,
                    dst.data() + i * numberWaveFunctions,
                    &inc);
          }
      }

    //
    // update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src, numberWaveFunctions);



    //
    // Hloc*M^{-1/2}*X
    //
    computeLocalHamiltonianTimesX(src, numberWaveFunctions, dst);

    //
    // required if its a pseudopotential calculation and number of nonlocal
    // atoms are greater than zero H^{nloc}*M^{-1/2}*X
    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0 &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        computeNonLocalHamiltonianTimesX(src, numberWaveFunctions, dst);
      }


    //
    // update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(
      dst, numberWaveFunctions);



    src.zeroOutGhosts();
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff = d_invSqrtMassVector.local_element(i);
        zdscal_(&numberWaveFunctions,
                &scalingCoeff,
                dst.data() + i * numberWaveFunctions,
                &inc);
      }


    //
    // unscale src M^{1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff =
          d_sqrtMassVector.local_element(i) * (1.0 / scalar);
        zdscal_(&numberWaveFunctions,
                &scalingCoeff,
                src.data() + i * numberWaveFunctions,
                &inc);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUMultiVec<std::complex<double>> &src,
    std::vector<std::complex<double>> &           cellSrcWaveFunctionMatrix,
    const unsigned int                            numberWaveFunctions,
    const bool                                    scaleFlag,
    const double                                  scalar,
    const double                                  scalarA,
    const double                                  scalarB,
    distributedCPUMultiVec<std::complex<double>> &dst,
    std::vector<std::complex<double>> &           cellDstWaveFunctionMatrix)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

#else
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUMultiVec<double> &src,
    const unsigned int numberWaveFunctions,
    const bool scaleFlag,
    const double scalar,
    distributedCPUMultiVec<double> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const unsigned int numberDofs = src.locallyOwnedSize();
    const unsigned int inc = 1;


    //
    // scale src vector with M^{-1/2}
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff = d_invSqrtMassVector.local_element(i);
        dscal_(&numberWaveFunctions,
               &scalingCoeff,
               src.data() + i * numberWaveFunctions,
               &inc);
      }


    if (scaleFlag)
      {
        for (int i = 0; i < numberDofs; ++i)
          {
            const double scalingCoeff = d_sqrtMassVector.local_element(i);
            dscal_(&numberWaveFunctions,
                   &scalingCoeff,
                   dst.data() + i * numberWaveFunctions,
                   &inc);
          }
      }

    //
    // update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src, numberWaveFunctions);


    //
    // Hloc*M^{-1/2}*X
    //
    computeLocalHamiltonianTimesX(src, numberWaveFunctions, dst, scalar);

    //
    // required if its a pseudopotential calculation and number of nonlocal
    // atoms are greater than zero H^{nloc}*M^{-1/2}*X
    if (dftPtr->d_dftParamsPtr->isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0 &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        computeNonLocalHamiltonianTimesX(src, numberWaveFunctions, dst, scalar);
      }



    //
    // update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(
      dst, numberWaveFunctions);

    src.zeroOutGhosts();
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        dscal_(&numberWaveFunctions,
               &d_invSqrtMassVector.local_element(i),
               dst.data() + i * numberWaveFunctions,
               &inc);
      }

    //
    // unscale src M^{1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        double scalingCoeff = d_sqrtMassVector.local_element(i);
        dscal_(&numberWaveFunctions,
               &scalingCoeff,
               src.data() + i * numberWaveFunctions,
               &inc);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUMultiVec<double> &src,
    std::vector<double> &cellSrcWaveFunctionMatrix,
    const unsigned int numberWaveFunctions,
    const bool scaleFlag,
    const double scalar,
    const double scalarA,
    const double scalarB,
    distributedCPUMultiVec<double> &dst,
    std::vector<double> &cellDstWaveFunctionMatrix)
  {
    const unsigned int numberDofs = src.locallyOwnedSize();
    const unsigned int inc = 1;


    for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
      {
        if (d_globalArrayClassificationMap[iDof] == 1)
          {
            const double scalingCoeff = d_invSqrtMassVector.local_element(iDof);
            for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
              {
                src.data()[iDof * numberWaveFunctions + iWave] *= scalingCoeff;
              }
          }
      }

    unsigned int productNumNodesWaveFunctions =
      d_numberNodesPerElement * numberWaveFunctions;
    std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
      d_numberNodesPerElement);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    unsigned int iElem = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            unsigned int indexTemp = productNumNodesWaveFunctions * iElem;

            cellPtr->get_dof_indices(cell_dof_indicesGlobal);
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                if (d_nodesPerCellClassificationMap[iNode] == 0)
                  {
                    dealii::types::global_dof_index localDoFId =
                      dftPtr->matrix_free_data.get_vector_partitioner()
                        ->global_to_local(cell_dof_indicesGlobal[iNode]);
                    const double scalingCoeff =
                      d_invSqrtMassVector.local_element(localDoFId);
                    unsigned int indexVal =
                      indexTemp + numberWaveFunctions * iNode;
                    for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                         ++iWave)
                      {
                        cellSrcWaveFunctionMatrix[indexVal + iWave] *=
                          scalingCoeff;
                      }
                  }
              }
            iElem++;
          }
      }


    if (scaleFlag)
      {
        for (int i = 0; i < numberDofs; ++i)
          {
            if (d_globalArrayClassificationMap[i] == 1)
              {
                const double scalingCoeff = d_sqrtMassVector.local_element(i);
                dscal_(&numberWaveFunctions,
                       &scalingCoeff,
                       dst.data() + i * numberWaveFunctions,
                       &inc);
              }
          }
      }



    //
    // update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src, numberWaveFunctions);



    computeHamiltonianTimesXInternal(src,
                                     cellSrcWaveFunctionMatrix,
                                     numberWaveFunctions,
                                     dst,
                                     cellDstWaveFunctionMatrix,
                                     scalar,
                                     scalarA,
                                     scalarB,
                                     scaleFlag);


    //
    // update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(
      dst, numberWaveFunctions);


    src.zeroOutGhosts();
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();

    // unscale cell level src vector
    for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
      {
        const double scalingCoeff = d_sqrtMassVector.local_element(iDof);
        if (d_globalArrayClassificationMap[iDof] == 1)
          {
            for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
              {
                src.data()[iDof * numberWaveFunctions + iWave] *= scalingCoeff;
              }
          }
      }

    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    endcellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .end();

    iElem = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            cellPtr->get_dof_indices(cell_dof_indicesGlobal);
            unsigned int indexTemp = productNumNodesWaveFunctions * iElem;
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                if (d_nodesPerCellClassificationMap[iNode] == 0)
                  {
                    dealii::types::global_dof_index localDoFId =
                      dftPtr->matrix_free_data.get_vector_partitioner()
                        ->global_to_local(cell_dof_indicesGlobal[iNode]);
                    const double scalingCoeff =
                      d_sqrtMassVector.local_element(localDoFId);
                    unsigned int indexVal =
                      indexTemp + numberWaveFunctions * iNode;
                    for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                         ++iWave)
                      {
                        unsigned int indexVal =
                          indexTemp + numberWaveFunctions * iNode;
                        cellSrcWaveFunctionMatrix[indexVal + iWave] *=
                          scalingCoeff;
                      }
                  }
              }
            iElem++;
          }
      }

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        if (d_globalArrayClassificationMap[i] == 1)
          {
            dscal_(&numberWaveFunctions,
                   &d_invSqrtMassVector.local_element(i),
                   dst.data() + i * numberWaveFunctions,
                   &inc);
          }
      }

    dftPtr->constraintsNoneDataInfo.set_zero(src, numberWaveFunctions);
  }
#endif



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHX(
    const dataTypes::number *       X,
    const unsigned int              numberWaveFunctions,
    const unsigned int              numberDofs,
    std::vector<dataTypes::number> &ProjHam)
  {
    //
    // Get access to number of locally owned nodes on the current processor
    //

    //
    // Resize ProjHam
    //
    ProjHam.clear();
    ProjHam.resize(numberWaveFunctions * numberWaveFunctions, 0.0);

    //
    // create temporary array XTemp
    //
    distributedCPUMultiVec<dataTypes::number> XTemp;
    reinit(numberWaveFunctions, XTemp, true);
    for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
      for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
        XTemp.data()[iNode * numberWaveFunctions + iWave] =
          X[iNode * numberWaveFunctions + iWave];

    //
    // create temporary array Y
    //
    distributedCPUMultiVec<dataTypes::number> Y;
    reinit(numberWaveFunctions, Y, true);

    Y.setValue(dataTypes::number(0));
    //
    // evaluate H times XTemp and store in Y
    //
    bool   scaleFlag = false;
    double scalar    = 1.0;
    HX(XTemp, numberWaveFunctions, scaleFlag, scalar, Y);

#ifdef USE_COMPLEX
    for (unsigned int i = 0; i < Y.locallyOwnedSize(); ++i)
      Y.data()[i] = std::conj(Y.data()[i]);

    char                       transA = 'N';
    char                       transB = 'T';
    const std::complex<double> alpha = 1.0, beta = 0.0;
    zgemm_(&transA,
           &transB,
           &numberWaveFunctions,
           &numberWaveFunctions,
           &numberDofs,
           &alpha,
           Y.begin(),
           &numberWaveFunctions,
           &X[0],
           &numberWaveFunctions,
           &beta,
           &ProjHam[0],
           &numberWaveFunctions);
#else
    char transA = 'N';
    char transB = 'T';
    const double alpha = 1.0, beta = 0.0;

    dgemm_(&transA,
           &transB,
           &numberWaveFunctions,
           &numberWaveFunctions,
           &numberDofs,
           &alpha,
           &X[0],
           &numberWaveFunctions,
           Y.begin(),
           &numberWaveFunctions,
           &beta,
           &ProjHam[0],
           &numberWaveFunctions);
#endif

    reinit(0, Y, true);

    dealii::Utilities::MPI::sum(ProjHam, mpi_communicator, ProjHam);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHX(
    const dataTypes::number *                        X,
    const unsigned int                               numberWaveFunctions,
    const unsigned int                               numberDofs,
    const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
    dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    //
    // Get access to number of locally owned nodes on the current processor
    //

    // create temporary arrays XBlock,Hx
    distributedCPUMultiVec<dataTypes::number> XBlock, HXBlock;

    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);
    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               numberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    /*
     * X^{T}*Hc*Xc is done in a blocked approach for memory optimization:
     * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
     * has a much smaller memory compared to X^{T}*H*Xc.
     * X^{T} (denoted by X in the code with column major format storage)
     * is a matrix with size (N x MLoc).
     * N is denoted by numberWaveFunctions in the code.
     * MLoc, which is number of local dofs is denoted by numberDofs in the code.
     * Xc denotes complex conjugate of X.
     * XcBlock is a matrix of size (MLoc x B). B is the block size.
     * A further optimization is done to reduce floating point operations:
     * As X^{T}*Hc*Xc is a Hermitian matrix, it suffices to compute only the
     * lower triangular part. To exploit this, we do X^{T}*Hc*Xc=Sum_{blocks}
     * XTrunc^{T}*H*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T}
     * with the row indices ranging from the lowest global index of XcBlock
     * (denoted by jvec in the code) to N. D=N-jvec. The parallel ScaLapack
     * matrix projHamPar is directly filled from the XTrunc^{T}*Hc*XcBlock
     * result
     */

    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::number> projHamBlock(numberWaveFunctions *
                                                  vectorsBlockSize,
                                                dataTypes::number(0.0));

    if (dftPtr->d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator,
        "Inside Blocked XtHX with parallel projected Ham matrix");

    for (unsigned int jvec = 0; jvec < numberWaveFunctions;
         jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B =
          std::min(vectorsBlockSize, numberWaveFunctions - jvec);
        if (jvec == 0 || B != vectorsBlockSize)
          {
            reinit(B, XBlock, true);
            HXBlock.reinit(XBlock);
          }

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            XBlock.setValue(dataTypes::number(0));
            // fill XBlock^{T} from X:
            for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
              for (unsigned int iWave = 0; iWave < B; ++iWave)
                XBlock.data()[iNode * B + iWave] =
                  X[iNode * numberWaveFunctions + jvec + iWave];


            MPI_Barrier(getMPICommunicator());
            // evaluate H times XBlock and store in HXBlock^{T}
            HXBlock.setValue(dataTypes::number(0));
            const bool   scaleFlag = false;
            const double scalar    = 1.0;

            HX(XBlock,
               B,
               scaleFlag,
               scalar,
               HXBlock,
               onlyHPrimePartForFirstOrderDensityMatResponse);

            MPI_Barrier(getMPICommunicator());

            const char transA = 'N';
            const char transB =
              std::is_same<dataTypes::number, std::complex<double>>::value ?
                'C' :
                'T';

            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            std::fill(projHamBlock.begin(),
                      projHamBlock.end(),
                      dataTypes::number(0.));

            const unsigned int D = numberWaveFunctions - jvec;

            // Comptute local XTrunc^{T}*HXcBlock.
            xgemm(&transA,
                  &transB,
                  &D,
                  &B,
                  &numberDofs,
                  &alpha,
                  &X[0] + jvec,
                  &numberWaveFunctions,
                  HXBlock.data(),
                  &B,
                  &beta,
                  &projHamBlock[0],
                  &D);

            MPI_Barrier(getMPICommunicator());
            // Sum local XTrunc^{T}*HXcBlock across domain decomposition
            // processors
            MPI_Allreduce(MPI_IN_PLACE,
                          &projHamBlock[0],
                          D * B,
                          dataTypes::mpi_type_id(&projHamBlock[0]),
                          MPI_SUM,
                          getMPICommunicator());

            // Copying only the lower triangular part to the ScaLAPACK projected
            // Hamiltonian matrix
            if (processGrid->is_process_active())
              for (unsigned int j = 0; j < B; ++j)
                if (globalToLocalColumnIdMap.find(j + jvec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const unsigned int localColumnId =
                      globalToLocalColumnIdMap[j + jvec];
                    for (unsigned int i = j + jvec; i < numberWaveFunctions;
                         ++i)
                      {
                        std::unordered_map<unsigned int, unsigned int>::iterator
                          it = globalToLocalRowIdMap.find(i);
                        if (it != globalToLocalRowIdMap.end())
                          projHamPar.local_el(it->second, localColumnId) =
                            projHamBlock[j * D + i - jvec];
                      }
                  }

          } // band parallelization

      } // block loop

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHXMixedPrec(
    const dataTypes::number *                        X,
    const unsigned int                               N,
    const unsigned int                               Ncore,
    const unsigned int                               numberDofs,
    const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
    dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    //
    // Get access to number of locally owned nodes on the current processor
    //

    // create temporary arrays XBlock,Hx
    distributedCPUMultiVec<dataTypes::number> XBlock, HXBlock;

    std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);
    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);

    /*
     * X^{T}*H*Xc is done in a blocked approach for memory optimization:
     * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
     * has a much smaller memory compared to X^{T}*Hc*Xc.
     * X^{T} (denoted by X in the code with column major format storage)
     * is a matrix with size (N x MLoc).
     * MLoc, which is number of local dofs is denoted by numberDofs in the code.
     * Xc denotes complex conjugate of X.
     * XcBlock is a matrix of size (MLoc x B). B is the block size.
     * A further optimization is done to reduce floating point operations:
     * As X^{T}*Hc*Xc is a Hermitian matrix, it suffices to compute only the
     * lower triangular part. To exploit this, we do X^{T}*Hc*Xc=Sum_{blocks}
     * XTrunc^{T}*Hc*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
     * X^{T} with the row indices ranging from the lowest global index of
     * XcBlock (denoted by jvec in the code) to N. D=N-jvec. The parallel
     * ScaLapack matrix projHamPar is directly filled from the
     * XTrunc^{T}*Hc*XcBlock result
     */

    const unsigned int vectorsBlockSize =
      std::min(dftPtr->d_dftParamsPtr->wfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::numberFP32> projHamBlockSinglePrec(
      N * vectorsBlockSize, 0.0);
    std::vector<dataTypes::number> projHamBlock(N * vectorsBlockSize, 0.0);

    std::vector<dataTypes::numberFP32> HXBlockSinglePrec;

    std::vector<dataTypes::numberFP32> XSinglePrec(X, X + numberDofs * N);

    if (dftPtr->d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator,
        "Inside Blocked XtHX with parallel projected Ham matrix");

    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);
        if (jvec == 0 || B != vectorsBlockSize)
          {
            reinit(B, XBlock, true);
            HXBlock.reinit(XBlock);
            HXBlockSinglePrec.resize(B * numberDofs);
          }

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            XBlock.setValue(dataTypes::number(0));
            // fill XBlock^{T} from X:
            for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
              for (unsigned int iWave = 0; iWave < B; ++iWave)
                XBlock.data()[iNode * B + iWave] = X[iNode * N + jvec + iWave];


            MPI_Barrier(getMPICommunicator());
            // evaluate H times XBlock and store in HXBlock^{T}
            HXBlock.setValue(dataTypes::number(0));
            const bool   scaleFlag = false;
            const double scalar    = 1.0;

            HX(XBlock,
               B,
               scaleFlag,
               scalar,
               HXBlock,
               onlyHPrimePartForFirstOrderDensityMatResponse);

            MPI_Barrier(getMPICommunicator());

            const char transA = 'N';
            const char transB =
              std::is_same<dataTypes::number, std::complex<double>>::value ?
                'C' :
                'T';
            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            std::fill(projHamBlock.begin(),
                      projHamBlock.end(),
                      dataTypes::number(0.));

            if (jvec + B > Ncore)
              {
                const unsigned int D = N - jvec;

                // Comptute local XTrunc^{T}*HXcBlock.
                xgemm(&transA,
                      &transB,
                      &D,
                      &B,
                      &numberDofs,
                      &alpha,
                      &X[0] + jvec,
                      &N,
                      HXBlock.data(),
                      &B,
                      &beta,
                      &projHamBlock[0],
                      &D);

                MPI_Barrier(getMPICommunicator());
                // Sum local XTrunc^{T}*HXcBlock across domain decomposition
                // processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &projHamBlock[0],
                              D * B,
                              dataTypes::mpi_type_id(&projHamBlock[0]),
                              MPI_SUM,
                              getMPICommunicator());


                // Copying only the lower triangular part to the ScaLAPACK
                // projected Hamiltonian matrix
                if (processGrid->is_process_active())
                  for (unsigned int j = 0; j < B; ++j)
                    if (globalToLocalColumnIdMap.find(j + jvec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[j + jvec];
                        for (unsigned int i = jvec + j; i < N; ++i)
                          {
                            std::unordered_map<unsigned int,
                                               unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(i);
                            if (it != globalToLocalRowIdMap.end())
                              projHamPar.local_el(it->second, localColumnId) =
                                projHamBlock[j * D + i - jvec];
                          }
                      }
              }
            else
              {
                const dataTypes::numberFP32 alphaSinglePrec =
                                              dataTypes::numberFP32(1.0),
                                            betaSinglePrec =
                                              dataTypes::numberFP32(0.0);

                for (unsigned int i = 0; i < numberDofs * B; ++i)
                  HXBlockSinglePrec[i] = HXBlock.data()[i];

                const unsigned int D = N - jvec;

                // single prec gemm
                xgemm(&transA,
                      &transB,
                      &D,
                      &B,
                      &numberDofs,
                      &alphaSinglePrec,
                      &XSinglePrec[0] + jvec,
                      &N,
                      &HXBlockSinglePrec[0],
                      &B,
                      &betaSinglePrec,
                      &projHamBlockSinglePrec[0],
                      &D);

                MPI_Barrier(getMPICommunicator());
                MPI_Allreduce(MPI_IN_PLACE,
                              &projHamBlockSinglePrec[0],
                              D * B,
                              dataTypes::mpi_type_id(
                                &projHamBlockSinglePrec[0]),
                              MPI_SUM,
                              getMPICommunicator());


                if (processGrid->is_process_active())
                  for (unsigned int j = 0; j < B; ++j)
                    if (globalToLocalColumnIdMap.find(j + jvec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[j + jvec];
                        for (unsigned int i = jvec + j; i < N; ++i)
                          {
                            std::unordered_map<unsigned int,
                                               unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(i);
                            if (it != globalToLocalRowIdMap.end())
                              projHamPar.local_el(it->second, localColumnId) =
                                projHamBlockSinglePrec[j * D + i - jvec];
                          }
                      }
              }


          } // band parallelization

      } // block loop

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffSpinPolarized(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  phiValues,
    const unsigned int                                   spinIndex,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)

  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values);
    const int           numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);


    // allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(2 * numberQuadraturePoints);
    std::vector<double> corrPotentialVal(2 * numberQuadraturePoints);
    std::vector<double> densityValue(2 * numberQuadraturePoints);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();


    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] =
                  (rhoValues[0][iElemCount * numberQuadraturePoints + q] +
                   rhoValues[1][iElemCount * numberQuadraturePoints + q]) /
                  2.0;
                densityValue[2 * q + 1] =
                  (rhoValues[0][iElemCount * numberQuadraturePoints + q] -
                   rhoValues[1][iElemCount * numberQuadraturePoints + q]) /
                  2.0;
              }

            const double *tempPhi =
              phiValues.data() + iElemCount * numberQuadraturePoints;


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &exchangePotentialVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &corrPotentialVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhi[q] + exchangePotentialVal[2 * q + spinIndex] +
                   corrPotentialVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            iElemCount++;
          }
      }



    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffSpinPolarized(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradRhoValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  phiValues,
    const unsigned int                                   spinIndex,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);

    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_inverse_jacobians |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                                    numberQuadraturePoints * 3,
                                                  0.0);

    // allocate storage for exchange potential
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> sigmaValue(3 * numberQuadraturePoints);
    std::vector<double> densityValue(2 * numberQuadraturePoints);
    std::vector<double> gradDensityValue(6 * numberQuadraturePoints);


    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] =
                  (rhoValues[0][iElemCount * numberQuadraturePoints + q] +
                   rhoValues[1][iElemCount * numberQuadraturePoints + q]) /
                  2.0;
                densityValue[2 * q + 1] =
                  (rhoValues[0][iElemCount * numberQuadraturePoints + q] -
                   rhoValues[1][iElemCount * numberQuadraturePoints + q]) /
                  2.0;
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  gradDensityValue[6 * q + iDim] =
                    (gradRhoValues[0][3 * iElemCount * numberQuadraturePoints +
                                      3 * q] +
                     gradRhoValues[1][3 * iElemCount * numberQuadraturePoints +
                                      3 * q + iDim]) /
                    2.0;
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  gradDensityValue[6 * q + 3 + iDim] =
                    (gradRhoValues[0][3 * iElemCount * numberQuadraturePoints +
                                      3 * q] -
                     gradRhoValues[1][3 * iElemCount * numberQuadraturePoints +
                                      3 * q + iDim]) /
                    2.0;
              }

            const double *tempPhi =
              phiValues.data() + iElemCount * numberQuadraturePoints;


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhi[q] + derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw = fe_values.JxW(q);
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][0][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][0][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][0][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][1][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][1][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][1][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] =
                  2.0 *
                  (inverseJacobians[q][2][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][2][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][2][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;
              }
            iElemCount++;
          } // subcell loop

      } // cell loop

    if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
         dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffExternalPotCorr(
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    d_externalPotCorrQuadratureId = externalPotCorrQuadratureId;
    const int numberQuadraturePoints =
      dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId)
        .size();

    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    dealii::FEValues<3> feValues(
      dftPtr->matrix_free_data.get_dof_handler().get_fe(),
      dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId),
      dealii::update_JxW_values);
    d_vEffExternalPotCorrJxW.resize(totalLocallyOwnedCells *
                                      numberQuadraturePoints,
                                    0.0);



    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();


    unsigned int iElem = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            feValues.reinit(cellPtr);
            const std::vector<double> &temp =
              externalPotCorrValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffExternalPotCorrJxW[totalLocallyOwnedCells * q + iElem] =
                  temp[q] * feValues.JxW(q);
              }
            iElem++;
          }
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffPrime(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                                                  rhoValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
    const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);

    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensityVal(numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              densityValue[q] =
                tempDensityTotalValues[iElemCount * numberQuadraturePoints + q];

            std::vector<double> densityPrimeValue =
              (rhoPrimeValues).find(cellPtr->id())->second;

            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[q] += temp2[q];
                  }
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2ExchangeEnergy;
            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2CorrEnergy;


            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2ExchEnergyWithDensityVal;

            outputDer2CorrEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2CorrEnergyWithDensityVal;


            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedFxc(
              numberQuadraturePoints,
              rhoData,
              outputDer2ExchangeEnergy,
              outputDer2CorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhiPrime[q] + (der2ExchEnergyWithDensityVal[q] +
                                      der2CorrEnergyWithDensityVal[q]) *
                                       densityPrimeValue[q]) *
                  fe_values.JxW(q);
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
  }


  // Fourth order stencil finite difference stencil used
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);

    std::vector<double> densityValue(2 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();
    const double lambda = 1e-2;

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);


            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += 2.0 * lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] +=
                  2.0 * lambda * dirperturb1[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  -(derExchEnergyWithDensityVal[2 * q + spinIndex] +
                    derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop

    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;


                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] += lambda * dirperturb1[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }


            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);


            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= 2.0 * lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] -=
                  2.0 * lambda * dirperturb1[2 * q + 1];
              }
            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }


            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempDensityMagValues   = rhoValues[1];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q + 0] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] +
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
                densityValue[2 * q + 1] =
                  0.5 *
                  (tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                          q] -
                   tempDensityMagValues[iElemCount * numberQuadraturePoints +
                                        q]);
              }

            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] -= lambda * dirperturb1[2 * q + 1];
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            rhoData[rhoDataAttributes::values] = &densityValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] -=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);

                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] *=
                  1.0 / 12.0 / lambda;
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  tempPhiPrime[q] * fe_values.JxW(q);
              }

            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffPrime(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                                                  rhoValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                                                  gradRhoValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoPrimeValues,
    const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_inverse_jacobians |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                                    numberQuadraturePoints * 3,
                                                  0.0);

    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> gradDensityValue(3 * numberQuadraturePoints);
    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensitySigmaVal(
      numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensitySigmaVal(
      numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2ExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> der2CorrEnergyWithDensityVal(numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            // std::vector<double> gradDensityValue =
            //  (gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempGradDensityTotalValues =
              gradRhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[q] =
                  tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                         q];
                gradDensityValue[3 * q + 0] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 0];
                gradDensityValue[3 * q + 1] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 1];
                gradDensityValue[3 * q + 2] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 2];
              }

            std::vector<double> densityPrimeValue =
              (rhoPrimeValues).find(cellPtr->id())->second;
            std::vector<double> gradDensityPrimeValue =
              (gradRhoPrimeValues).find(cellPtr->id())->second;

            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[q] += temp2[q];
                    gradDensityValue[3 * q + 0] += temp3[3 * q + 0];
                    gradDensityValue[3 * q + 1] += temp3[3 * q + 1];
                    gradDensityValue[3 * q + 2] += temp3[3 * q + 2];
                  }
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX = gradDensityValue[3 * q + 0];
                const double gradRhoY = gradDensityValue[3 * q + 1];
                const double gradRhoZ = gradDensityValue[3 * q + 2];
                sigmaValue[q] = gradRhoX * gradRhoX + gradRhoY * gradRhoY +
                                gradRhoZ * gradRhoZ;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;

            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2ExchangeEnergy;
            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2CorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigmaVal;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigmaVal;

            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2ExchEnergyWithDensityVal;
            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
                &der2ExchEnergyWithDensitySigmaVal;
            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithSigma] =
                &der2ExchEnergyWithSigmaVal;

            outputDer2CorrEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2CorrEnergyWithDensityVal;
            outputDer2CorrEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
                &der2CorrEnergyWithDensitySigmaVal;
            outputDer2CorrEnergy[fxcOutputDataAttributes::der2EnergyWithSigma] =
              &der2CorrEnergyWithSigmaVal;


            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedFxc(
              numberQuadraturePoints,
              rhoData,
              outputDer2ExchangeEnergy,
              outputDer2CorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX = gradDensityValue[3 * q + 0];
                const double gradRhoY = gradDensityValue[3 * q + 1];
                const double gradRhoZ = gradDensityValue[3 * q + 2];

                const double gradRhoPrimeX = gradDensityPrimeValue[3 * q + 0];
                const double gradRhoPrimeY = gradDensityPrimeValue[3 * q + 1];
                const double gradRhoPrimeZ = gradDensityPrimeValue[3 * q + 2];

                const double gradRhoDotGradRhoPrime = gradRhoX * gradRhoPrimeX +
                                                      gradRhoY * gradRhoPrimeY +
                                                      gradRhoZ * gradRhoPrimeZ;

                // 2.0*del2{exc}/del{sigma}{rho}*\dot{gradrho^{\prime},gradrho}
                const double sigmaDensityMixedDerTerm =
                  2.0 *
                  (der2ExchEnergyWithDensitySigmaVal[q] +
                   der2CorrEnergyWithDensitySigmaVal[q]) *
                  gradRhoDotGradRhoPrime;


                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  (tempPhiPrime[q] +
                   (der2ExchEnergyWithDensityVal[q] +
                    der2CorrEnergyWithDensityVal[q]) *
                     densityPrimeValue[q] +
                   sigmaDensityMixedDerTerm) *
                  fe_values.JxW(q);
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw      = fe_values.JxW(q);
                const double gradRhoX = gradDensityValue[3 * q + 0];
                const double gradRhoY = gradDensityValue[3 * q + 1];
                const double gradRhoZ = gradDensityValue[3 * q + 2];

                const double gradRhoPrimeX = gradDensityPrimeValue[3 * q + 0];
                const double gradRhoPrimeY = gradDensityPrimeValue[3 * q + 1];
                const double gradRhoPrimeZ = gradDensityPrimeValue[3 * q + 2];

                const double gradRhoDotGradRhoPrime = gradRhoX * gradRhoPrimeX +
                                                      gradRhoY * gradRhoPrimeY +
                                                      gradRhoZ * gradRhoPrimeZ;

                const double term1 =
                  derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];
                const double term2 =
                  der2ExchEnergyWithSigmaVal[q] + der2CorrEnergyWithSigmaVal[q];
                const double term3 = der2ExchEnergyWithDensitySigmaVal[q] +
                                     der2CorrEnergyWithDensitySigmaVal[q];

                std::vector<double> tempvec(3, 0.0);

                tempvec[0] = (term1 * gradRhoPrimeX +
                              2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoX +
                              term3 * densityPrimeValue[q] * gradRhoX) *
                             jxw;
                tempvec[1] = (term1 * gradRhoPrimeY +
                              2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoY +
                              term3 * densityPrimeValue[q] * gradRhoY) *
                             jxw;
                tempvec[2] = (term1 * gradRhoPrimeZ +
                              2.0 * term2 * gradRhoDotGradRhoPrime * gradRhoZ +
                              term3 * densityPrimeValue[q] * gradRhoZ) *
                             jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] =
                  (inverseJacobians[q][0][0] * tempvec[0] +
                   inverseJacobians[q][0][1] * tempvec[1] +
                   inverseJacobians[q][0][2] * tempvec[2]);

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] =
                  (inverseJacobians[q][1][0] * tempvec[0] +
                   inverseJacobians[q][1][1] * tempvec[1] +
                   inverseJacobians[q][1][2] * tempvec[2]);

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] =
                  (inverseJacobians[q][2][0] * tempvec[0] +
                   inverseJacobians[q][2][1] * tempvec[1] +
                   inverseJacobians[q][2][2] * tempvec[2]);
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
  }

  // Fourth order stencil finite difference stencil used
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  gradRhoValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues)
  {
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const dealii::Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dftPtr->FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values |
                                    dealii::update_inverse_jacobians |
                                    dealii::update_jacobians);
    const unsigned int  numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                                    numberQuadraturePoints * 3,
                                                  0.0);

    std::vector<double> densityValue(numberQuadraturePoints);
    std::vector<double> gradDensityValue(3 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigma(3 * numberQuadraturePoints);
    std::vector<double> sigmaValue(3 * numberQuadraturePoints);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active(),
      endcellPtr = dftPtr->matrix_free_data
                     .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                     .end();
    const double lambda = 1e-2;

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            // std::vector<double> gradDensityValue =
            //  (gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempGradDensityTotalValues =
              gradRhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[q] =
                  tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                         q];
                gradDensityValue[3 * q + 0] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 0];
                gradDensityValue[3 * q + 1] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 1];
                gradDensityValue[3 * q + 2] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 2];
              }



            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            const std::vector<double> &dirperturb2 =
              gradRhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += 2.0 * lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] +=
                  2.0 * lambda * dirperturb1[2 * q + 1];
                gradDensityValue[6 * q + 0] +=
                  2.0 * lambda * dirperturb2[6 * q + 0];
                gradDensityValue[6 * q + 1] +=
                  2.0 * lambda * dirperturb2[6 * q + 1];
                gradDensityValue[6 * q + 2] +=
                  2.0 * lambda * dirperturb2[6 * q + 2];
                gradDensityValue[6 * q + 3] +=
                  2.0 * lambda * dirperturb2[6 * q + 3];
                gradDensityValue[6 * q + 4] +=
                  2.0 * lambda * dirperturb2[6 * q + 4];
                gradDensityValue[6 * q + 5] +=
                  2.0 * lambda * dirperturb2[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] =
                  -(derExchEnergyWithDensityVal[2 * q + spinIndex] +
                    derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw = fe_values.JxW(q);
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] =
                  -2.0 *
                  (inverseJacobians[q][0][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][0][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][0][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] =
                  -2.0 *
                  (inverseJacobians[q][1][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][1][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][1][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] =
                  -2.0 *
                  (inverseJacobians[q][2][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][2][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][2][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop

    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();


            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            // std::vector<double> gradDensityValue =
            //  (gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempGradDensityTotalValues =
              gradRhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[q] =
                  tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                         q];
                gradDensityValue[3 * q + 0] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 0];
                gradDensityValue[3 * q + 1] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 1];
                gradDensityValue[3 * q + 2] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 2];
              }

            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            const std::vector<double> &dirperturb2 =
              gradRhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] += lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] += lambda * dirperturb1[2 * q + 1];
                gradDensityValue[6 * q + 0] += lambda * dirperturb2[6 * q + 0];
                gradDensityValue[6 * q + 1] += lambda * dirperturb2[6 * q + 1];
                gradDensityValue[6 * q + 2] += lambda * dirperturb2[6 * q + 2];
                gradDensityValue[6 * q + 3] += lambda * dirperturb2[6 * q + 3];
                gradDensityValue[6 * q + 4] += lambda * dirperturb2[6 * q + 4];
                gradDensityValue[6 * q + 5] += lambda * dirperturb2[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw = fe_values.JxW(q);
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] +=
                  8.0 * 2.0 *
                  (inverseJacobians[q][0][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][0][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][0][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] +=
                  8.0 * 2.0 *
                  (inverseJacobians[q][1][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][1][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][1][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] +=
                  8.0 * 2.0 *
                  (inverseJacobians[q][2][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][2][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][2][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();

            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            // std::vector<double> gradDensityValue =
            //  (gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempGradDensityTotalValues =
              gradRhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[q] =
                  tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                         q];
                gradDensityValue[3 * q + 0] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 0];
                gradDensityValue[3 * q + 1] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 1];
                gradDensityValue[3 * q + 2] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 2];
              }


            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            const std::vector<double> &dirperturb2 =
              gradRhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= 2.0 * lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] -=
                  2.0 * lambda * dirperturb1[2 * q + 1];
                gradDensityValue[6 * q + 0] -=
                  2.0 * lambda * dirperturb2[6 * q + 0];
                gradDensityValue[6 * q + 1] -=
                  2.0 * lambda * dirperturb2[6 * q + 1];
                gradDensityValue[6 * q + 2] -=
                  2.0 * lambda * dirperturb2[6 * q + 2];
                gradDensityValue[6 * q + 3] -=
                  2.0 * lambda * dirperturb2[6 * q + 3];
                gradDensityValue[6 * q + 4] -=
                  2.0 * lambda * dirperturb2[6 * q + 4];
                gradDensityValue[6 * q + 5] -=
                  2.0 * lambda * dirperturb2[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw = fe_values.JxW(q);
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] +=
                  2.0 *
                  (inverseJacobians[q][0][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][0][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][0][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] +=
                  2.0 *
                  (inverseJacobians[q][1][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][1][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][1][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] +=
                  2.0 *
                  (inverseJacobians[q][2][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][2][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][2][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop


    cellPtr =
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
        .begin_active();
    iElemCount = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_values.get_inverse_jacobians();


            // std::vector<double> densityValue =
            //  (rhoValues).find(cellPtr->id())->second;
            // std::vector<double> gradDensityValue =
            //  (gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempDensityTotalValues = rhoValues[0];
            const std::vector<double> &tempGradDensityTotalValues =
              gradRhoValues[0];
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[q] =
                  tempDensityTotalValues[iElemCount * numberQuadraturePoints +
                                         q];
                gradDensityValue[3 * q + 0] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 0];
                gradDensityValue[3 * q + 1] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 1];
                gradDensityValue[3 * q + 2] =
                  tempGradDensityTotalValues[3 * iElemCount *
                                               numberQuadraturePoints +
                                             3 * q + 2];
              }


            const std::vector<double> &tempPhiPrime =
              phiPrimeValues.find(cellPtr->id())->second;

            if (dftPtr->d_dftParamsPtr->nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;

                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;

                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                    gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }


            const std::vector<double> &dirperturb1 =
              rhoPrimeValues.find(cellPtr->id())->second;

            const std::vector<double> &dirperturb2 =
              gradRhoPrimeValues.find(cellPtr->id())->second;

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                densityValue[2 * q] -= lambda * dirperturb1[2 * q];
                densityValue[2 * q + 1] -= lambda * dirperturb1[2 * q + 1];
                gradDensityValue[6 * q + 0] -= lambda * dirperturb2[6 * q + 0];
                gradDensityValue[6 * q + 1] -= lambda * dirperturb2[6 * q + 1];
                gradDensityValue[6 * q + 2] -= lambda * dirperturb2[6 * q + 2];
                gradDensityValue[6 * q + 3] -= lambda * dirperturb2[6 * q + 3];
                gradDensityValue[6 * q + 4] -= lambda * dirperturb2[6 * q + 4];
                gradDensityValue[6 * q + 5] -= lambda * dirperturb2[6 * q + 5];
              }


            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double gradRhoX1 = gradDensityValue[6 * q + 0];
                const double gradRhoY1 = gradDensityValue[6 * q + 1];
                const double gradRhoZ1 = gradDensityValue[6 * q + 2];
                const double gradRhoX2 = gradDensityValue[6 * q + 3];
                const double gradRhoY2 = gradDensityValue[6 * q + 4];
                const double gradRhoZ2 = gradDensityValue[6 * q + 5];

                sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }
            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;


            rhoData[rhoDataAttributes::values]         = &densityValue;
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derExchEnergyWithDensityVal;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigma;

            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithDensity] =
                &derCorrEnergyWithDensityVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigma;

            dftPtr->d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
              numberQuadraturePoints,
              rhoData,
              outputDerExchangeEnergy,
              outputDerCorrEnergy);



            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] -=
                  8.0 *
                  (derExchEnergyWithDensityVal[2 * q + spinIndex] +
                   derCorrEnergyWithDensityVal[2 * q + spinIndex]) *
                  fe_values.JxW(q);

                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] *=
                  1.0 / 12.0 / lambda;
                d_vEffJxW[totalLocallyOwnedCells * q + iElemCount] +=
                  tempPhiPrime[q] * fe_values.JxW(q);
              }

            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              {
                const double jxw = fe_values.JxW(q);
                const double gradRhoX =
                  gradDensityValue[6 * q + 0 + 3 * spinIndex];
                const double gradRhoY =
                  gradDensityValue[6 * q + 1 + 3 * spinIndex];
                const double gradRhoZ =
                  gradDensityValue[6 * q + 2 + 3 * spinIndex];
                const double gradRhoOtherX =
                  gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
                const double gradRhoOtherY =
                  gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
                const double gradRhoOtherZ =
                  gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
                const double term =
                  derExchEnergyWithSigma[3 * q + 2 * spinIndex] +
                  derCorrEnergyWithSigma[3 * q + 2 * spinIndex];
                const double termOff = derExchEnergyWithSigma[3 * q + 1] +
                                       derCorrEnergyWithSigma[3 * q + 1];

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] -=
                  8.0 * 2.0 *
                  (inverseJacobians[q][0][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][0][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][0][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] -=
                  8.0 * 2.0 *
                  (inverseJacobians[q][1][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][1][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][1][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] -=
                  8.0 * 2.0 *
                  (inverseJacobians[q][2][0] *
                     (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) +
                   inverseJacobians[q][2][1] *
                     (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) +
                   inverseJacobians[q][2][2] *
                     (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ)) *
                  jxw;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         3 * q +
                                                       iElemCount] *=
                  1.0 / 12.0 / lambda;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 1) +
                                                       iElemCount] *=
                  1.0 / 12.0 / lambda;

                d_invJacderExcWithSigmaTimesGradRhoJxW[totalLocallyOwnedCells *
                                                         (3 * q + 2) +
                                                       iElemCount] *=
                  1.0 / 12.0 / lambda;
              }
            iElemCount++;
          } // if cellPtr->is_locally_owned() loop

      } // cell loop
  }
#include "inst.cc"
} // namespace dftfe
