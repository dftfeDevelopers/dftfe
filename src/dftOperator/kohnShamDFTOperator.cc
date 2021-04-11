// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
    const MPI_Comm &                   mpi_comm_replica)
    : dftPtr(_dftPtr)
    , d_kPointIndex(0)
    , d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell(
        dftPtr->d_densityDofHandlerIndex))
    , d_numberMacroCells(_dftPtr->matrix_free_data.n_macro_cells())
    , d_isStiffnessMatrixExternalPotCorrComputed(false)
    , mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(mpi_comm_replica,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , operatorDFTClass(mpi_comm_replica,
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
    computing_timer.enter_section("kohnShamDFTOperatorClass setup");


    dftPtr->matrix_free_data.initialize_dof_vector(
      d_invSqrtMassVector, dftPtr->d_densityDofHandlerIndex);
    d_sqrtMassVector.reinit(d_invSqrtMassVector);


    //
    // create macro cell map to subcells
    //
    d_macroCellSubCellMap.resize(d_numberMacroCells);
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(iMacroCell);
        d_macroCellSubCellMap[iMacroCell] = n_sub_cells;
      }

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
                                   (1 + dftParameters::spinPolarized));


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

    computing_timer.exit_section("kohnShamDFTOperatorClass setup");
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int                    numberWaveFunctions,
    distributedCPUVec<dataTypes::number> &flattenedArray,
    bool                                  flag)
  {
    if (flag)
      vectorTools::createDealiiVector<dataTypes::number>(
        dftPtr->matrix_free_data.get_vector_partitioner(),
        numberWaveFunctions,
        flattenedArray);

    if (dftParameters::isPseudopotential)
      {
        vectorTools::createDealiiVector<dataTypes::number>(
          dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
          numberWaveFunctions,
          dftPtr->d_projectorKetTimesVectorParFlattened);
      }



    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.get_partitioner(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_flattenedArrayMacroCellLocalProcIndexIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap);



    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.get_partitioner(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_FullflattenedArrayMacroCellLocalProcIndexIdMap,
      d_normalCellIdToMacroCellIdMap,
      d_macroCellIdToNormalCellIdMap,
      d_FullflattenedArrayCellLocalProcIndexIdMap);

    getOverloadedConstraintMatrix()->precomputeMaps(
      dftPtr->matrix_free_data.get_vector_partitioner(),
      flattenedArray.get_partitioner(),
      numberWaveFunctions);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int numberWaveFunctions)
  {
    if (dftParameters::isPseudopotential)
      {
        vectorTools::createDealiiVector<dataTypes::number>(
          dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
          numberWaveFunctions,
          dftPtr->d_projectorKetTimesVectorParFlattened);
      }
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::initCellWaveFunctionMatrix(
    const unsigned int                    numberWaveFunctions,
    distributedCPUVec<dataTypes::number> &src,
    std::vector<dataTypes::number> &      cellWaveFunctionMatrix)
  {
    unsigned int numberCells = dftPtr->matrix_free_data.n_physical_cells();
    cellWaveFunctionMatrix.resize(numberCells * d_numberNodesPerElement *
                                    numberWaveFunctions,
                                  0.0);
    unsigned int       iElem = 0;
    const unsigned int inc   = 1;
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        for (unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell];
             ++iCell)
          {
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                dealii::types::global_dof_index localNodeId =
                  d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
#ifdef USE_COMPLEX
                zcopy_(
                  &numberWaveFunctions,
                  src.begin() + localNodeId,
                  &inc,
                  &cellWaveFunctionMatrix
                    [d_numberNodesPerElement * numberWaveFunctions * iElem +
                     numberWaveFunctions *
                       iNode], //&cellWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],
                  &inc);

#else
                dcopy_(
                  &numberWaveFunctions,
                  src.begin() + localNodeId,
                  &inc,
                  &cellWaveFunctionMatrix
                    [d_numberNodesPerElement * numberWaveFunctions * iElem +
                     numberWaveFunctions *
                       iNode], //&cellWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],
                  &inc);
#endif
              }
            ++iElem;
          }
      }
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
    fillGlobalArrayFromCellWaveFunctionMatrix(
      const unsigned int                    numberWaveFunctions,
      const std::vector<dataTypes::number> &cellWaveFunctionMatrix,
      distributedCPUVec<dataTypes::number> &glbArray)

  {
    unsigned int       iElem = 0;
    const unsigned int inc   = 1;
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        for (unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell];
             ++iCell)
          {
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                if (d_nodesPerCellClassificationMap[iNode] == 0)
                  {
                    dealii::types::global_dof_index localNodeId =
                      d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem]
                                                                  [iNode];
#ifdef USE_COMPLEX
                    zcopy_(
                      &numberWaveFunctions,
                      &cellWaveFunctionMatrix
                        [d_numberNodesPerElement * numberWaveFunctions * iElem +
                         numberWaveFunctions *
                           iNode], //&cellWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],//src.begin()+localNodeId,
                      &inc,
                      glbArray.begin() + localNodeId,
                      &inc);
#else
                    dcopy_(
                      &numberWaveFunctions,
                      &cellWaveFunctionMatrix
                        [d_numberNodesPerElement * numberWaveFunctions * iElem +
                         numberWaveFunctions *
                           iNode], //&cellWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],//src.begin()+localNodeId,
                      &inc,
                      glbArray.begin() + localNodeId,
                      &inc);
#endif
                  }
              }
            ++iElem;
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
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        for (unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell];
             ++iCell)
          {
            unsigned int indexTemp = productNumNodesWaveFunctions * iElem;
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                if (d_nodesPerCellClassificationMap[iNode] == 0)
                  {
                    unsigned int indexVal =
                      indexTemp + numberWaveFunctions * iNode;
                    for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                         ++iWave)
                      {
                        // cellYWaveFunctionMatrix[iElem][numberWaveFunctions*iNode
                        // + iWave] =
                        // scalarA*cellXWaveFunctionMatrix[iElem][numberWaveFunctions*iNode
                        // +
                        // iWave]+scalarB*cellYWaveFunctionMatrix[iElem][numberWaveFunctions*iNode
                        // + iWave];
                        cellYWaveFunctionMatrix[indexVal + iWave] =
                          scalarB * cellYWaveFunctionMatrix[indexVal + iWave] +
                          scalarA * cellXWaveFunctionMatrix[indexVal + iWave];
                      }
                  }
              }
            ++iElem;
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
    computing_timer.enter_section("kohnShamDFTOperatorClass Mass assembly");
    invSqrtMassVec = 0.0;
    sqrtMassVec    = 0.0;

    QGaussLobatto<3>   quadrature(FEOrder + 1);
    FEValues<3>        fe_values(dofHandler.get_fe(),
                          quadrature,
                          update_values | update_JxW_values);
    const unsigned int dofs_per_cell   = (dofHandler.get_fe()).dofs_per_cell;
    const unsigned int num_quad_points = quadrature.size();
    Vector<double>     massVectorLocal(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dofHandler.begin_active(),
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

    invSqrtMassVec.compress(VectorOperation::add);


    for (types::global_dof_index i = 0; i < invSqrtMassVec.size(); ++i)
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
            ExcMessage(
              "Value of inverse square root of mass matrix on the unconstrained node is undefined"));
        }

    invSqrtMassVec.compress(VectorOperation::insert);
    sqrtMassVec.compress(VectorOperation::insert);
    computing_timer.exit_section("kohnShamDFTOperatorClass Mass assembly");
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
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const int numberQuadraturePoints = quadrature_formula.size();
      

    d_vEffJxW.resize(totalLocallyOwnedCells*numberQuadraturePoints,0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(numberQuadraturePoints);
    std::vector<double> corrPotentialVal(numberQuadraturePoints);
    
    //
    // loop over cell block
    //
     unsigned int iElemCount = 0;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
	const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
	for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	    fe_values.reinit(cellPtr);
	    
            std::vector<double> densityValue  = (*rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempPhi =
              phiValues.find(cellPtr->id())->second;
           

            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  densityValue[q] += temp2[q];
              }

	    xc_lda_vxc(&(dftPtr->funcX),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &exchangePotentialVal[0]);
	    
	    xc_lda_vxc(&(dftPtr->funcC),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &corrPotentialVal[0]);
       
	    for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
		d_vEffJxW[iElemCount*numberQuadraturePoints + q] = (tempPhi[q] + exchangePotentialVal[q] + corrPotentialVal[q])*fe_values.JxW(q);
	      }
	    
	    iElemCount++;
	  }

      }
    
    /*vEff.reinit(n_cells, numberQuadraturePoints);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // loop over cell block
    //
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        std::vector<dealii::VectorizedArray<double>> tempPhi(
          numberQuadraturePoints, dealii::make_vectorized_array(0.0));
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<std::vector<double>> tempRho(n_sub_cells);
        std::vector<std::vector<double>> tempPseudo(n_sub_cells);
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
            tempRho[v] = (*rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &temp =
              phiValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              tempPhi[q][v] = temp[q];

            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  tempRho[v][q] += temp2[q];
              }
          }

        for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
          {
            //
            // loop over each cell
            //
            std::vector<double> densityValue(n_sub_cells),
              exchangePotentialVal(n_sub_cells), corrPotentialVal(n_sub_cells);
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                densityValue[v] = tempRho[v][q];
              }

            xc_lda_vxc(&(dftPtr->funcX),
                       n_sub_cells,
                       &densityValue[0],
                       &exchangePotentialVal[0]);
            xc_lda_vxc(&(dftPtr->funcC),
                       n_sub_cells,
                       &densityValue[0],
                       &corrPotentialVal[0]);

            VectorizedArray<double> exchangePotential, corrPotential;
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                exchangePotential[v] = exchangePotentialVal[v];
                corrPotential[v]     = corrPotentialVal[v];
              }

            //
            // sum all to vEffective
            //
            vEff(cell, q) = tempPhi[q] + exchangePotential + corrPotential;
          }
	  }*/

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEff(
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values|update_inverse_jacobians|update_jacobians);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                             numberQuadraturePoints * 3,
                                             0.0);

    //allocate storage for exchange potential
    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);
    
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    //loop over cell block
    //
    unsigned int iElemCount = 0;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
	const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<std::vector<double>> tempRho(n_sub_cells);
        std::vector<std::vector<double>> tempGradRho(n_sub_cells);
        std::vector<std::vector<double>> tempPseudo(n_sub_cells);
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	    fe_values.reinit(cellPtr);

	    const std::vector<DerivativeForm<1, 3, 3>> &inverseJacobians =
	      fe_values.get_inverse_jacobians();

            std::vector<double> densityValue =
            (*rhoValues).find(cellPtr->id())->second;
            std::vector<double> gradDensityValue =
            (*gradRhoValues).find(cellPtr->id())->second;
	    
            //tempRho[v] = (*rhoValues).find(cellPtr->id())->second;
            //tempGradRho[v] = (*gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempPhi =
              phiValues.find(cellPtr->id())->second;
	    
            
            if (dftParameters::nonLinearCoreCorrection)
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

	    for(unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
                const double gradRhoX = gradDensityValue[3 * q + 0];
		const double gradRhoY = gradDensityValue[3 * q + 1];
		const double gradRhoZ = gradDensityValue[3 * q + 2];
		sigmaValue[q] =
		  gradRhoX * gradRhoX + gradRhoY * gradRhoY + gradRhoZ * gradRhoZ;
	      }

	    xc_gga_vxc(&(dftPtr->funcX),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &sigmaValue[0],
		       &derExchEnergyWithDensityVal[0],
		       &derExchEnergyWithSigmaVal[0]);
	    
	    xc_gga_vxc(&(dftPtr->funcC),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &sigmaValue[0],
		       &derCorrEnergyWithDensityVal[0],
		       &derCorrEnergyWithSigmaVal[0]);

	    
	    for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
		d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
		  (tempPhi[q] + derExchEnergyWithDensityVal[q] +
		   derCorrEnergyWithDensityVal[q])*fe_values.JxW(q);
	      }


	    //Rethink about this
	    for(unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
		const double jxw      = fe_values.JxW(q);
		const double gradRhoX = gradDensityValue[3 * q + 0];
		const double gradRhoY = gradDensityValue[3 * q + 1];
		const double gradRhoZ = gradDensityValue[3 * q + 2];
		const double term =
		  derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];
		

		d_invJacderExcWithSigmaTimesGradRhoJxW[iElemCount*numberQuadraturePoints*3 + 3*q] = 2.0*(inverseJacobians[q][0][0]*gradRhoX + inverseJacobians[q][0][1]*gradRhoY + inverseJacobians[q][0][2]*gradRhoZ)*term*jxw;
		d_invJacderExcWithSigmaTimesGradRhoJxW[iElemCount*numberQuadraturePoints*3 + 3*q + 1] = 2.0*(inverseJacobians[q][1][0]*gradRhoX + inverseJacobians[q][1][1]*gradRhoY + inverseJacobians[q][1][2]*gradRhoZ)*term*jxw;
		d_invJacderExcWithSigmaTimesGradRhoJxW[iElemCount*numberQuadraturePoints*3 + 3*q + 2] = 2.0*(inverseJacobians[q][2][0]*gradRhoX + inverseJacobians[q][2][1]*gradRhoY + inverseJacobians[q][2][2]*gradRhoZ)*term*jxw;
		
	      }

	    iElemCount++;
	    
          }
      
      }

    /* vEff.reinit(n_cells, numberQuadraturePoints);
    derExcWithSigmaTimesGradRho.reinit(
      TableIndices<2>(n_cells, numberQuadraturePoints));
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // loop over cell block
    //
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        std::vector<dealii::VectorizedArray<double>> tempPhi(
          numberQuadraturePoints, dealii::make_vectorized_array(0.0));
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<std::vector<double>> tempRho(n_sub_cells);
        std::vector<std::vector<double>> tempGradRho(n_sub_cells);
        std::vector<std::vector<double>> tempPseudo(n_sub_cells);
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
            tempRho[v] = (*rhoValues).find(cellPtr->id())->second;
            tempGradRho[v] = (*gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &temp =
              phiValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              tempPhi[q][v] = temp[q];

            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    tempRho[v][q] += temp2[q];
                    tempGradRho[v][3 * q + 0] += temp3[3 * q + 0];
                    tempGradRho[v][3 * q + 1] += temp3[3 * q + 1];
                    tempGradRho[v][3 * q + 2] += temp3[3 * q + 2];
                  }
              }
          }
        for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
          {
            //
            // loop over each cell
            //
            std::vector<double> densityValue(n_sub_cells),
              derExchEnergyWithDensityVal(n_sub_cells),
              derCorrEnergyWithDensityVal(n_sub_cells),
              derExchEnergyWithSigma(n_sub_cells),
              derCorrEnergyWithSigma(n_sub_cells), sigmaValue(n_sub_cells);
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                densityValue[v] = tempRho[v][q];
                double gradRhoX = tempGradRho[v][3 * q + 0];
                double gradRhoY = tempGradRho[v][3 * q + 1];
                double gradRhoZ = tempGradRho[v][3 * q + 2];
                sigmaValue[v]   = gradRhoX * gradRhoX + gradRhoY * gradRhoY +
                                gradRhoZ * gradRhoZ;
              }

            xc_gga_vxc(&(dftPtr->funcX),
                       n_sub_cells,
                       &densityValue[0],
                       &sigmaValue[0],
                       &derExchEnergyWithDensityVal[0],
                       &derExchEnergyWithSigma[0]);
            xc_gga_vxc(&(dftPtr->funcC),
                       n_sub_cells,
                       &densityValue[0],
                       &sigmaValue[0],
                       &derCorrEnergyWithDensityVal[0],
                       &derCorrEnergyWithSigma[0]);


            VectorizedArray<double> derExchEnergyWithDensity,
              derCorrEnergyWithDensity, derExcWithSigmaTimesGradRhoX,
              derExcWithSigmaTimesGradRhoY, derExcWithSigmaTimesGradRhoZ;
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                derExchEnergyWithDensity[v] = derExchEnergyWithDensityVal[v];
                derCorrEnergyWithDensity[v] = derCorrEnergyWithDensityVal[v];
                double gradRhoX             = tempGradRho[v][3 * q + 0];
                double gradRhoY             = tempGradRho[v][3 * q + 1];
                double gradRhoZ             = tempGradRho[v][3 * q + 2];
                double term =
                  derExchEnergyWithSigma[v] + derCorrEnergyWithSigma[v];
                derExcWithSigmaTimesGradRhoX[v] = term * gradRhoX;
                derExcWithSigmaTimesGradRhoY[v] = term * gradRhoY;
                derExcWithSigmaTimesGradRhoZ[v] = term * gradRhoZ;
              }

            //
            // sum all to vEffective
            //
            vEff(cell, q) =
              tempPhi[q] + derExchEnergyWithDensity + derCorrEnergyWithDensity;
            derExcWithSigmaTimesGradRho(cell, q)[0] =
              derExcWithSigmaTimesGradRhoX;
            derExcWithSigmaTimesGradRho(cell, q)[1] =
              derExcWithSigmaTimesGradRhoY;
            derExcWithSigmaTimesGradRho(cell, q)[2] =
              derExcWithSigmaTimesGradRhoZ;
          }
	  }*/

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }


#ifdef USE_COMPLEX
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUVec<std::complex<double>> &src,
    const unsigned int                       numberWaveFunctions,
    const bool                               scaleFlag,
    const double                             scalar,
    distributedCPUVec<std::complex<double>> &dst)


  {
    const unsigned int numberDofs = src.local_size() / numberWaveFunctions;
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
                src.begin() + i * numberWaveFunctions,
                &inc);
      }

    // const std::complex<double> zeroValue = 0.0;
    // dst = zeroValue;

    if (scaleFlag)
      {
        for (int i = 0; i < numberDofs; ++i)
          {
            const double scalingCoeff = d_sqrtMassVector.local_element(i);
            zdscal_(&numberWaveFunctions,
                    &scalingCoeff,
                    dst.begin() + i * numberWaveFunctions,
                    &inc);
          }
      }

    //
    // update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src, numberWaveFunctions);


    // src.update_ghost_values();


    //
    // Hloc*M^{-1/2}*X
    //
    computeLocalHamiltonianTimesX(src, numberWaveFunctions, dst);

    //
    // required if its a pseudopotential calculation and number of nonlocal
    // atoms are greater than zero H^{nloc}*M^{-1/2}*X
    if (dftParameters::isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      {
        computeNonLocalHamiltonianTimesX(src, numberWaveFunctions, dst);
      }


    //
    // update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(
      dst, numberWaveFunctions);



    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff = d_invSqrtMassVector.local_element(i);
        zdscal_(&numberWaveFunctions,
                &scalingCoeff,
                dst.begin() + i * numberWaveFunctions,
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
                src.begin() + i * numberWaveFunctions,
                &inc);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUVec<std::complex<double>> &src,
    std::vector<std::complex<double>> &      cellSrcWaveFunctionMatrix,
    const unsigned int                       numberWaveFunctions,
    const bool                               scaleFlag,
    const double                             scalar,
    const double                             scalarA,
    const double                             scalarB,
    distributedCPUVec<std::complex<double>> &dst,
    std::vector<std::complex<double>> &      cellDstWaveFunctionMatrix)

  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

#else
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUVec<double> &src,
    const unsigned int numberWaveFunctions,
    const bool scaleFlag,
    const double scalar,
    distributedCPUVec<double> &dst)


  {
    const unsigned int numberDofs = src.local_size() / numberWaveFunctions;
    const unsigned int inc = 1;


    //
    // scale src vector with M^{-1/2}
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        const double scalingCoeff =
          d_invSqrtMassVector.local_element(i); 
        dscal_(&numberWaveFunctions,
               &scalingCoeff,
               src.begin() + i * numberWaveFunctions,
               &inc);
      }


    if (scaleFlag)
      {
        for (int i = 0; i < numberDofs; ++i)
          {
            const double scalingCoeff = d_sqrtMassVector.local_element(i);
            dscal_(&numberWaveFunctions,
                   &scalingCoeff,
                   dst.begin() + i * numberWaveFunctions,
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
    if (dftParameters::isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      {
        computeNonLocalHamiltonianTimesX(src, numberWaveFunctions, dst, scalar);
      }



    //
    // update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(
      dst, numberWaveFunctions);


    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        dscal_(&numberWaveFunctions,
               &d_invSqrtMassVector.local_element(i),
               dst.begin() + i * numberWaveFunctions,
               &inc);
      }

    //
    // unscale src M^{1/2}*X
    //
    for (unsigned int i = 0; i < numberDofs; ++i)
      {
        double scalingCoeff =
          d_sqrtMassVector.local_element(i); 
        dscal_(&numberWaveFunctions,
               &scalingCoeff,
               src.begin() + i * numberWaveFunctions,
               &inc);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::HX(
    distributedCPUVec<double> &src,
    std::vector<double> &cellSrcWaveFunctionMatrix,
    const unsigned int numberWaveFunctions,
    const bool scaleFlag,
    const double scalar,
    const double scalarA,
    const double scalarB,
    distributedCPUVec<double> &dst,
    std::vector<double> &cellDstWaveFunctionMatrix)



  {
    const unsigned int numberDofs = src.local_size() / numberWaveFunctions;
    const unsigned int inc = 1;


    for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
      {
        if (d_globalArrayClassificationMap[iDof] == 1)
          {
            const double scalingCoeff = d_invSqrtMassVector.local_element(iDof);
            for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
              {
                src.local_element(iDof * numberWaveFunctions + iWave) *=
                  scalingCoeff;
              }
          }
      }

    unsigned int iElem = 0;
    unsigned int productNumNodesWaveFunctions =
      d_numberNodesPerElement * numberWaveFunctions;
    std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
      d_numberNodesPerElement);
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        for (unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell];
             ++iCell)
          {
            unsigned int indexTemp = productNumNodesWaveFunctions * iElem;
            dftPtr->matrix_free_data.get_cell_iterator(iMacroCell, iCell)
              ->get_dof_indices(cell_dof_indicesGlobal);
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
            ++iElem;
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
                       dst.begin() + i * numberWaveFunctions,
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


    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    // unscale cell level src vector
    for (unsigned int iDof = 0; iDof < numberDofs; ++iDof)
      {
        const double scalingCoeff = d_sqrtMassVector.local_element(iDof);
        if (d_globalArrayClassificationMap[iDof] == 1)
          {
            for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
              {
                src.local_element(iDof * numberWaveFunctions + iWave) *=
                  scalingCoeff;
              }
          }
      }

    iElem = 0;
    for (unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells;
         ++iMacroCell)
      {
        for (unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell];
             ++iCell)
          {
            dftPtr->matrix_free_data.get_cell_iterator(iMacroCell, iCell)
              ->get_dof_indices(cell_dof_indicesGlobal);
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
            ++iElem;
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
                   dst.begin() + i * numberWaveFunctions,
                   &inc);
          }
      }

    dftPtr->constraintsNoneDataInfo.set_zero(src, numberWaveFunctions);
  }
#endif



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHX(
    const std::vector<dataTypes::number> &X,
    const unsigned int                    numberWaveFunctions,
    std::vector<dataTypes::number> &      ProjHam)
  {
    //
    // Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size() / numberWaveFunctions;

    //
    // Resize ProjHam
    //
    ProjHam.clear();
    ProjHam.resize(numberWaveFunctions * numberWaveFunctions, 0.0);

    //
    // create temporary array XTemp
    //
    distributedCPUVec<dataTypes::number> XTemp;
    reinit(numberWaveFunctions, XTemp, true);
    for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
      for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
        XTemp.local_element(iNode * numberWaveFunctions + iWave) =
          X[iNode * numberWaveFunctions + iWave];

    //
    // create temporary array Y
    //
    distributedCPUVec<dataTypes::number> Y;
    reinit(numberWaveFunctions, Y, true);

    Y = dataTypes::number(0);
    //
    // evaluate H times XTemp and store in Y
    //
    bool   scaleFlag = false;
    double scalar    = 1.0;
    HX(XTemp, numberWaveFunctions, scaleFlag, scalar, Y);

#ifdef USE_COMPLEX
    for (unsigned int i = 0; i < Y.local_size(); ++i)
      Y.local_element(i) = std::conj(Y.local_element(i));

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

    Y.reinit(0);

    Utilities::MPI::sum(ProjHam, mpi_communicator, ProjHam);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHX(
    const std::vector<dataTypes::number> &X,
    const unsigned int                    numberWaveFunctions,
    const std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
      &                                         processGrid,
    dealii::ScaLAPACKMatrix<dataTypes::number> &projHamPar)
  {
#ifdef USE_COMPLEX
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
    //
    // Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size() / numberWaveFunctions;

    // create temporary arrays XBlock,Hx
    distributedCPUVec<dataTypes::number> XBlock, HXBlock;

    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
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
     * X^{T}*H*Xc is done in a blocked approach for memory optimization:
     * Sum_{blocks} X^{T}*H*XcBlock. The result of each X^{T}*H*XcBlock
     * has a much smaller memory compared to X^{T}*H*Xc.
     * X^{T} (denoted by X in the code with column major format storage)
     * is a matrix with size (N x MLoc).
     * N is denoted by numberWaveFunctions in the code.
     * MLoc, which is number of local dofs is denoted by numberDofs in the code.
     * Xc denotes complex conjugate of X.
     * XcBlock is a matrix of size (MLoc x B). B is the block size.
     * A further optimization is done to reduce floating point operations:
     * As X^{T}*H*Xc is a Hermitian matrix, it suffices to compute only the
     * lower triangular part. To exploit this, we do X^{T}*H*Xc=Sum_{blocks}
     * XTrunc^{T}*H*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T}
     * with the row indices ranging from the lowest global index of XcBlock
     * (denoted by jvec in the code) to N. D=N-jvec. The parallel ScaLapack
     * matrix projHamPar is directly filled from the XTrunc^{T}*H*XcBlock result
     */

    const unsigned int vectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::number> projHamBlock(numberWaveFunctions *
                                                  vectorsBlockSize,
                                                0.0);

    if (dftParameters::verbosity >= 4)
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
            XBlock = 0;
            // fill XBlock^{T} from X:
            for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
              for (unsigned int iWave = 0; iWave < B; ++iWave)
                XBlock.local_element(iNode * B + iWave) =
                  X[iNode * numberWaveFunctions + jvec + iWave];


            MPI_Barrier(getMPICommunicator());
            // evaluate H times XBlock^{T} and store in HXBlock^{T}
            HXBlock = 0;
            const bool scaleFlag = false;
            const dataTypes::number scalar = 1.0;

            HX(XBlock, B, scaleFlag, scalar, HXBlock);

            MPI_Barrier(getMPICommunicator());

            const char transA = 'N';
            const char transB = 'T';

            const dataTypes::number alpha = 1.0, beta = 0.0;
            std::fill(projHamBlock.begin(), projHamBlock.end(), 0.);

            const unsigned int D = numberWaveFunctions - jvec;

            // Comptute local XTrunc^{T}*HXcBlock.
            dgemm_(&transA,
                   &transB,
                   &D,
                   &B,
                   &numberDofs,
                   &alpha,
                   &X[0] + jvec,
                   &numberWaveFunctions,
                   HXBlock.begin(),
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
                        std::map<unsigned int, unsigned int>::iterator it =
                          globalToLocalRowIdMap.find(i);
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
#endif
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::XtHXMixedPrec(
    const std::vector<dataTypes::number> &X,
    const unsigned int                    N,
    const unsigned int                    Ncore,
    const std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
      &                                         processGrid,
    dealii::ScaLAPACKMatrix<dataTypes::number> &projHamPar)
  {
#ifdef USE_COMPLEX
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
    //
    // Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size() / N;

    // create temporary arrays XBlock,Hx
    distributedCPUVec<dataTypes::number> XBlock, HXBlock;

    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
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
     * Sum_{blocks} X^{T}*H*XcBlock. The result of each X^{T}*H*XcBlock
     * has a much smaller memory compared to X^{T}*H*Xc.
     * X^{T} (denoted by X in the code with column major format storage)
     * is a matrix with size (N x MLoc).
     * MLoc, which is number of local dofs is denoted by numberDofs in the code.
     * Xc denotes complex conjugate of X.
     * XcBlock is a matrix of size (MLoc x B). B is the block size.
     * A further optimization is done to reduce floating point operations:
     * As X^{T}*H*Xc is a Hermitian matrix, it suffices to compute only the
     * lower triangular part. To exploit this, we do X^{T}*H*Xc=Sum_{blocks}
     * XTrunc^{T}*H*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T}
     * with the row indices ranging from the lowest global index of XcBlock
     * (denoted by jvec in the code) to N. D=N-jvec. The parallel ScaLapack
     * matrix projHamPar is directly filled from the XTrunc^{T}*H*XcBlock result
     */

    const unsigned int vectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::numberLowPrec> projHamBlockSinglePrec(
      N * vectorsBlockSize, 0.0);
    std::vector<dataTypes::number> projHamBlock(N * vectorsBlockSize, 0.0);

    std::vector<dataTypes::numberLowPrec> HXBlockSinglePrec;

    std::vector<dataTypes::numberLowPrec> XSinglePrec(&X[0], &X[0] + X.size());

    if (dftParameters::verbosity >= 4)
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
            XBlock = 0;
            // fill XBlock^{T} from X:
            for (unsigned int iNode = 0; iNode < numberDofs; ++iNode)
              for (unsigned int iWave = 0; iWave < B; ++iWave)
                XBlock.local_element(iNode * B + iWave) =
                  X[iNode * N + jvec + iWave];


            MPI_Barrier(getMPICommunicator());
            // evaluate H times XBlock^{T} and store in HXBlock^{T}
            HXBlock = 0;
            const bool scaleFlag = false;
            const dataTypes::number scalar = 1.0;

            HX(XBlock, B, scaleFlag, scalar, HXBlock);

            MPI_Barrier(getMPICommunicator());

            const char transA = 'N';
#  ifdef USE_COMPLEX
            const char transB = 'C';
#  else
            const char transB = 'T';
#  endif
            const dataTypes::number alpha = 1.0, beta = 0.0;
            std::fill(projHamBlock.begin(), projHamBlock.end(), 0.);

            if (jvec + B > Ncore)
              {
                const unsigned int D = N - jvec;

                // Comptute local XTrunc^{T}*HXcBlock.
                dgemm_(&transA,
                       &transB,
                       &D,
                       &B,
                       &numberDofs,
                       &alpha,
                       &X[0] + jvec,
                       &N,
                       HXBlock.begin(),
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
                            std::map<unsigned int, unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(i);
                            if (it != globalToLocalRowIdMap.end())
                              projHamPar.local_el(it->second, localColumnId) =
                                projHamBlock[j * D + i - jvec];
                          }
                      }
              }
            else
              {
                const dataTypes::numberLowPrec alphaSinglePrec = 1.0,
                                               betaSinglePrec = 0.0;

                for (unsigned int i = 0; i < numberDofs * B; ++i)
                  HXBlockSinglePrec[i] = HXBlock.local_element(i);

                const unsigned int D = N - jvec;

                sgemm_(&transA,
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
                            std::map<unsigned int, unsigned int>::iterator it =
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
#endif
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffSpinPolarized(
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const unsigned int                                   spinIndex,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)

  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const int numberQuadraturePoints = quadrature_formula.size();

    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(2 * numberQuadraturePoints);
    std::vector<double> corrPotentialVal(2 * numberQuadraturePoints);

    //
    // loop over cell block
    //
    unsigned int iElemCount = 0;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
	const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
	for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	    fe_values.reinit(cellPtr);
	    
	    std::vector<double> densityValue = (*rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &tempPhi =
              phiValues.find(cellPtr->id())->second;
           

            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    densityValue[2 * q] += temp2[q] / 2.0;
                    densityValue[2 * q + 1] += temp2[q] / 2.0;
                  }
              }

            xc_lda_vxc(&(dftPtr->funcX),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &exchangePotentialVal[0]);
	    
	    xc_lda_vxc(&(dftPtr->funcC),
		       numberQuadraturePoints,
		       &densityValue[0],
		       &corrPotentialVal[0]);

	    for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
		d_vEffJxW[iElemCount*numberQuadraturePoints + q] = (tempPhi[q] + exchangePotentialVal[2*q + spinIndex] + corrPotentialVal[2*q + spinIndex])*fe_values.JxW(q);
	      }

	    iElemCount++;
	    
          }
      }


    
    /*vEff.reinit(n_cells, numberQuadraturePoints);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // loop over cell block
    //
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        std::vector<dealii::VectorizedArray<double>> tempPhi(
          numberQuadraturePoints, dealii::make_vectorized_array(0.0));
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<std::vector<double>> tempRho(n_sub_cells);
        std::vector<std::vector<double>> tempPseudo(n_sub_cells);
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
            tempRho[v] = (*rhoValues).find(cellPtr->id())->second;

            const std::vector<double> &temp =
              phiValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              tempPhi[q][v] = temp[q];


            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    tempRho[v][2 * q] += temp2[q] / 2.0;
                    tempRho[v][2 * q + 1] += temp2[q] / 2.0;
                  }
              }
          }

        for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
          {
            //
            // loop over each cell
            //
            std::vector<double> densityValue(2 * n_sub_cells),
              exchangePotentialVal(2 * n_sub_cells),
              corrPotentialVal(2 * n_sub_cells);
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                densityValue[2 * v + 1] = tempRho[v][2 * q + 1];
                densityValue[2 * v]     = tempRho[v][2 * q];
              }

            xc_lda_vxc(&(dftPtr->funcX),
                       n_sub_cells,
                       &densityValue[0],
                       &exchangePotentialVal[0]);
            xc_lda_vxc(&(dftPtr->funcC),
                       n_sub_cells,
                       &densityValue[0],
                       &corrPotentialVal[0]);

            VectorizedArray<double> exchangePotential, corrPotential;
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                exchangePotential[v] = exchangePotentialVal[2 * v + spinIndex];
                corrPotential[v]     = corrPotentialVal[2 * v + spinIndex];
              }

            //
            // sum all to vEffective
            //
            vEff(cell, q) = tempPhi[q] + exchangePotential + corrPotential;
          }
      }*/

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeVEffSpinPolarized(
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const unsigned int                                   spinIndex,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int n_array_elements =
      VectorizedArray<double>::n_array_elements;
    const int numberQuadraturePoints =
      dftPtr->matrix_free_data.get_quadrature(0).size();
    vEff.reinit(n_cells, numberQuadraturePoints);
    derExcWithSigmaTimesGradRho.reinit(
      TableIndices<2>(n_cells, numberQuadraturePoints));
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // loop over cell block
    //
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        std::vector<dealii::VectorizedArray<double>> tempPhi(
          numberQuadraturePoints, dealii::make_vectorized_array(0.0));
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<std::vector<double>> tempRho(n_sub_cells);
        std::vector<std::vector<double>> tempGradRho(n_sub_cells);
        std::vector<std::vector<double>> tempPseudo(n_sub_cells);
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr    = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
            tempRho[v] = (*rhoValues).find(cellPtr->id())->second;
            tempGradRho[v] = (*gradRhoValues).find(cellPtr->id())->second;

            const std::vector<double> &temp =
              phiValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              tempPhi[q][v] = temp[q];

            if (dftParameters::nonLinearCoreCorrection)
              {
                const std::vector<double> &temp2 =
                  rhoCoreValues.find(cellPtr->id())->second;
                const std::vector<double> &temp3 =
                  gradRhoCoreValues.find(cellPtr->id())->second;
                for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                  {
                    tempRho[v][2 * q] += temp2[q] / 2.0;
                    tempRho[v][2 * q + 1] += temp2[q] / 2.0;
                    tempGradRho[v][6 * q + 0] += temp3[3 * q + 0] / 2.0;
                    tempGradRho[v][6 * q + 1] += temp3[3 * q + 1] / 2.0;
                    tempGradRho[v][6 * q + 2] += temp3[3 * q + 2] / 2.0;
                    tempGradRho[v][6 * q + 3] += temp3[3 * q + 0] / 2.0;
                    tempGradRho[v][6 * q + 4] += temp3[3 * q + 1] / 2.0;
                    tempGradRho[v][6 * q + 5] += temp3[3 * q + 2] / 2.0;
                  }
              }
          }

        for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
          {
            //
            // loop over each cell
            //
            std::vector<double> densityValue(2 * n_sub_cells),
              derExchEnergyWithDensityVal(2 * n_sub_cells),
              derCorrEnergyWithDensityVal(2 * n_sub_cells),
              derExchEnergyWithSigma(3 * n_sub_cells),
              derCorrEnergyWithSigma(3 * n_sub_cells),
              sigmaValue(3 * n_sub_cells);
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                densityValue[2 * v + 1] = tempRho[v][2 * q + 1];
                densityValue[2 * v]     = tempRho[v][2 * q];
                double gradRhoX1        = tempGradRho[v][6 * q + 0];
                double gradRhoY1        = tempGradRho[v][6 * q + 1];
                double gradRhoZ1        = tempGradRho[v][6 * q + 2];
                double gradRhoX2        = tempGradRho[v][6 * q + 3];
                double gradRhoY2        = tempGradRho[v][6 * q + 4];
                double gradRhoZ2        = tempGradRho[v][6 * q + 5];
                //
                sigmaValue[3 * v + 0] = gradRhoX1 * gradRhoX1 +
                                        gradRhoY1 * gradRhoY1 +
                                        gradRhoZ1 * gradRhoZ1;
                sigmaValue[3 * v + 1] = gradRhoX1 * gradRhoX2 +
                                        gradRhoY1 * gradRhoY2 +
                                        gradRhoZ1 * gradRhoZ2;
                sigmaValue[3 * v + 2] = gradRhoX2 * gradRhoX2 +
                                        gradRhoY2 * gradRhoY2 +
                                        gradRhoZ2 * gradRhoZ2;
              }

            xc_gga_vxc(&(dftPtr->funcX),
                       n_sub_cells,
                       &densityValue[0],
                       &sigmaValue[0],
                       &derExchEnergyWithDensityVal[0],
                       &derExchEnergyWithSigma[0]);
            xc_gga_vxc(&(dftPtr->funcC),
                       n_sub_cells,
                       &densityValue[0],
                       &sigmaValue[0],
                       &derCorrEnergyWithDensityVal[0],
                       &derCorrEnergyWithSigma[0]);


            VectorizedArray<double> derExchEnergyWithDensity,
              derCorrEnergyWithDensity, derExcWithSigmaTimesGradRhoX,
              derExcWithSigmaTimesGradRhoY, derExcWithSigmaTimesGradRhoZ;
            for (unsigned int v = 0; v < n_sub_cells; ++v)
              {
                derExchEnergyWithDensity[v] =
                  derExchEnergyWithDensityVal[2 * v + spinIndex];
                derCorrEnergyWithDensity[v] =
                  derCorrEnergyWithDensityVal[2 * v + spinIndex];
                double gradRhoX = tempGradRho[v][6 * q + 0 + 3 * spinIndex];
                double gradRhoY = tempGradRho[v][6 * q + 1 + 3 * spinIndex];
                double gradRhoZ = tempGradRho[v][6 * q + 2 + 3 * spinIndex];
                double gradRhoOtherX =
                  tempGradRho[v][6 * q + 0 + 3 * (1 - spinIndex)];
                double gradRhoOtherY =
                  tempGradRho[v][6 * q + 1 + 3 * (1 - spinIndex)];
                double gradRhoOtherZ =
                  tempGradRho[v][6 * q + 2 + 3 * (1 - spinIndex)];
                double term = derExchEnergyWithSigma[3 * v + 2 * spinIndex] +
                              derCorrEnergyWithSigma[3 * v + 2 * spinIndex];
                double termOff = derExchEnergyWithSigma[3 * v + 1] +
                                 derCorrEnergyWithSigma[3 * v + 1];
                derExcWithSigmaTimesGradRhoX[v] =
                  term * gradRhoX + 0.5 * termOff * gradRhoOtherX;
                derExcWithSigmaTimesGradRhoY[v] =
                  term * gradRhoY + 0.5 * termOff * gradRhoOtherY;
                derExcWithSigmaTimesGradRhoZ[v] =
                  term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ;
              }

            //
            // sum all to vEffective
            //
            vEff(cell, q) =
              tempPhi[q] + derExchEnergyWithDensity + derCorrEnergyWithDensity;
            derExcWithSigmaTimesGradRho(cell, q)[0] =
              derExcWithSigmaTimesGradRhoX;
            derExcWithSigmaTimesGradRho(cell, q)[1] =
              derExcWithSigmaTimesGradRhoY;
            derExcWithSigmaTimesGradRho(cell, q)[2] =
              derExcWithSigmaTimesGradRhoZ;
          }
      }

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
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
    const unsigned int n_cells    = dftPtr->matrix_free_data.n_macro_cells();
    const int numberQuadraturePoints = dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId).size();

    const unsigned int totalLocallyOwnedCells = dftPtr->matrix_free_data.n_physical_cells();
    FEValues<3> feValues(dftPtr->matrix_free_data.get_dof_handler().get_fe(), dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId), update_JxW_values);
    d_vEffExternalPotCorrJxW.resize(totalLocallyOwnedCells*numberQuadraturePoints,0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
    unsigned int iElem = 0;
    for(unsigned int cell = 0; cell < n_cells; ++cell)
      {
	const unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(cell);
	for(unsigned int v = 0; v < n_sub_cells; ++v)
	  {
	    cellPtr = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	    feValues.reinit(cellPtr);
	    const std::vector<double> & temp = externalPotCorrValues.find(cellPtr->id())->second;
	    for(unsigned int q = 0; q < numberQuadraturePoints; ++q)
	      {
		d_vEffExternalPotCorrJxW[numberQuadraturePoints*iElem + q] = temp[q]*feValues.JxW(q);
	      }
	    iElem++;
	  }
      }

    
    
    /*d_vEffExternalPotCorr.reinit(n_cells, numberQuadraturePoints);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

    //
    // loop over cell block
    //
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        const unsigned int n_sub_cells =
          dftPtr->matrix_free_data.n_components_filled(cell);
        std::vector<VectorizedArray<double>> tempVec(
          numberQuadraturePoints, make_vectorized_array(0.0));
        for (unsigned int v = 0; v < n_sub_cells; ++v)
          {
            cellPtr = dftPtr->matrix_free_data.get_cell_iterator(cell, v);
            const std::vector<double> &temp =
              externalPotCorrValues.find(cellPtr->id())->second;
            for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
              tempVec[q][v] = temp[q];
          }

        for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
          {
            d_vEffExternalPotCorr(cell, q) = tempVec[q];
          }
	  }*/
  }

#include "inst.cc"
} // namespace dftfe
