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
// @author Sambit Das (2017)
//
#include <force.h>
#include <dft.h>
#include <dftUtils.h>
#include <eshelbyTensor.h>

namespace dftfe
{
  // compute configurational force contribution from nuclear self energy on the
  // mesh nodes using linear shape function generators
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::computeConfigurationalForceEselfLinFE(
    const dealii::DoFHandler<3>         &dofHandlerElectro,
    const vselfBinsManager              &vselfBinsManagerElectro,
    const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
    const dftfe::uInt                    smearedChargeQuadratureId)
  {
    const std::vector<std::vector<double>> &atomLocations =
      dftPtr->atomLocations;
    const std::vector<std::vector<double>> &imagePositionsTrunc =
      dftPtr->d_imagePositionsTrunc;
    const std::vector<double> &imageChargesTrunc = dftPtr->d_imageChargesTrunc;
    //
    // First add configurational force contribution from the volume integral
    //
    dealii::QGauss<3>   quadrature(d_dftParams.densityQuadratureRule);
    dealii::FEValues<3> feForceValues(FEForce,
                                      quadrature,
                                      dealii::update_gradients |
                                        dealii::update_JxW_values);
    dealii::FEValues<3> feVselfValues(dofHandlerElectro.get_fe(),
                                      quadrature,
                                      dealii::update_gradients);
    const dftfe::uInt   forceDofsPerCell = FEForce.dofs_per_cell;
    const dftfe::uInt   forceBaseIndicesPerCell =
      forceDofsPerCell / FEForce.components;
    dealii::Vector<double> elementalForce(forceDofsPerCell);
    const dftfe::uInt      numQuadPoints = quadrature.size();
    std::vector<dealii::types::global_dof_index> forceLocalDofIndices(
      forceDofsPerCell);
    const dftfe::uInt numberBins =
      vselfBinsManagerElectro.getAtomIdsBins().size();
    std::vector<dealii::Tensor<1, 3, double>> gradVselfQuad(numQuadPoints);
    std::vector<dftfe::uInt>     baseIndexDofsVec(forceBaseIndicesPerCell * 3);
    dealii::Tensor<1, 3, double> baseIndexForceVec;

    // kpoint group parallelization data structures
    const dftfe::uInt numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
    std::vector<dftfe::Int> kptGroupLowHighPlusOneIndices;

    if (numberBins > 0)
      dftUtils::createKpointParallelizationIndices(
        dftPtr->interpoolcomm, numberBins, kptGroupLowHighPlusOneIndices);

    if (!d_dftParams.floatingNuclearCharges)
      {
        for (dftfe::uInt ibase = 0; ibase < forceBaseIndicesPerCell; ++ibase)
          {
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              baseIndexDofsVec[3 * ibase + idim] =
                FEForce.component_to_system_index(idim, ibase);
          }

        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                const std::vector<dealii::DoFHandler<3>::active_cell_iterator>
                  &cellsVselfBallDofHandler =
                    d_cellsVselfBallsDofHandlerElectro[iBin];
                const std::vector<dealii::DoFHandler<3>::active_cell_iterator>
                  &cellsVselfBallDofHandlerForce =
                    d_cellsVselfBallsDofHandlerForceElectro[iBin];
                const distributedCPUVec<double> &iBinVselfField =
                  vselfBinsManagerElectro.getVselfFieldBins()[iBin];
                std::vector<dealii::DoFHandler<3>::active_cell_iterator>::
                  const_iterator iter1;
                std::vector<dealii::DoFHandler<3>::active_cell_iterator>::
                  const_iterator iter2;
                iter2 = cellsVselfBallDofHandlerForce.begin();
                for (iter1 = cellsVselfBallDofHandler.begin();
                     iter1 != cellsVselfBallDofHandler.end();
                     ++iter1, ++iter2)
                  {
                    dealii::DoFHandler<3>::active_cell_iterator cell = *iter1;
                    dealii::DoFHandler<3>::active_cell_iterator cellForce =
                      *iter2;
                    feVselfValues.reinit(cell);
                    feVselfValues.get_function_gradients(iBinVselfField,
                                                         gradVselfQuad);

                    feForceValues.reinit(cellForce);
                    cellForce->get_dof_indices(forceLocalDofIndices);
                    elementalForce = 0.0;
                    for (dftfe::uInt ibase = 0; ibase < forceBaseIndicesPerCell;
                         ++ibase)
                      {
                        baseIndexForceVec = 0;
                        for (dftfe::uInt qPoint = 0; qPoint < numQuadPoints;
                             ++qPoint)
                          {
                            baseIndexForceVec +=
                              eshelbyTensor::getVselfBallEshelbyTensor(
                                gradVselfQuad[qPoint]) *
                              feForceValues.shape_grad(
                                baseIndexDofsVec[3 * ibase], qPoint) *
                              feForceValues.JxW(qPoint);
                          } // q point loop
                        for (dftfe::uInt idim = 0; idim < 3; idim++)
                          elementalForce[baseIndexDofsVec[3 * ibase + idim]] =
                            baseIndexForceVec[idim];
                      } // base index loop

                    d_constraintsNoneForceElectro.distribute_local_to_global(
                      elementalForce,
                      forceLocalDofIndices,
                      d_configForceVectorLinFEElectro);
                  } // cell loop
              }     // kpt paral
          }         // bin loop
      }

    //
    // Add configurational force due to smeared charges
    //
    if (d_dftParams.smearedNuclearCharges)
      {
        const std::map<dftfe::Int, std::set<dftfe::Int>> &atomIdsBins =
          vselfBinsManagerElectro.getAtomIdsBins();

        FEEvaluationWrapperClass<3> forceEvalSmearedCharge(
          -1,
          1,
          matrixFreeDataElectro,
          d_forceDofHandlerIndexElectro,
          smearedChargeQuadratureId);

        dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
        const dftfe::uInt                           numQuadPointsSmearedb =
          forceEvalSmearedCharge.n_q_points;

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            zeroTensor[idim] = dealii::make_vectorized_array(0.0);
          }

        dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
            {
              zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
            }

        dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
          numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradVselfSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);

        std::map<dftfe::uInt, std::vector<double>>
          forceContributionSmearedChargesGammaAtoms;

        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                FEEvaluationWrapperClass<1> vselfEvalSmearedCharge(
                  -1,
                  1,
                  matrixFreeDataElectro,
                  dftPtr->d_binsStartDofHandlerIndexElectro + 4 * iBin,
                  smearedChargeQuadratureId);

                const std::set<dftfe::Int> &atomIdsInBin =
                  atomIdsBins.find(iBin)->second;
                forceContributionSmearedChargesGammaAtoms.clear();
                for (dftfe::uInt cell = 0;
                     cell < matrixFreeDataElectro.n_cell_batches();
                     ++cell)
                  {
                    std::set<dftfe::uInt>
                                      nonTrivialSmearedChargeAtomIdsMacroCell;
                    const dftfe::uInt numSubCells =
                      matrixFreeDataElectro.n_active_entries_per_cell_batch(
                        cell);
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();
                        const std::vector<dftfe::uInt> &temp =
                          dftPtr->d_bCellNonTrivialAtomIdsBins[iBin]
                            .find(subCellId)
                            ->second;
                        for (dftfe::Int i = 0; i < temp.size(); i++)
                          nonTrivialSmearedChargeAtomIdsMacroCell.insert(
                            temp[i]);
                      }

                    if (nonTrivialSmearedChargeAtomIdsMacroCell.size() == 0)
                      continue;

                    forceEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.read_dof_values_plain(
                      vselfBinsManagerElectro.getVselfFieldBins()[iBin]);
                    vselfEvalSmearedCharge.evaluate(
                      dealii::EvaluationFlags::gradients);

                    std::fill(smearedbQuads.begin(),
                              smearedbQuads.end(),
                              dealii::make_vectorized_array(0.0));
                    std::fill(gradVselfSmearedChargeQuads.begin(),
                              gradVselfSmearedChargeQuads.end(),
                              zeroTensor);

                    bool isCellNonTrivial = false;
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();

                        const std::vector<dftfe::Int> &bQuadAtomIdsCell =
                          dftPtr->d_bQuadAtomIdsAllAtoms.find(subCellId)
                            ->second;
                        const std::vector<double> &bQuadValuesCell =
                          dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;

                        for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                          {
                            if (atomIdsInBin.find(bQuadAtomIdsCell[q]) !=
                                atomIdsInBin.end())
                              {
                                isCellNonTrivial           = true;
                                smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                              }
                          } // quad loop
                      }     // subcell loop

                    if (!isCellNonTrivial)
                      continue;

                    for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                      {
                        gradVselfSmearedChargeQuads[q] =
                          vselfEvalSmearedCharge.get_gradient(q);

                        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                          F = zeroTensor;
                        F   = gradVselfSmearedChargeQuads[q] * smearedbQuads[q];

                        if (!d_dftParams.floatingNuclearCharges)
                          forceEvalSmearedCharge.submit_value(F, q);
                      } // quadloop

                    if (!d_dftParams.floatingNuclearCharges)
                      {
                        forceEvalSmearedCharge.integrate(
                          dealii::EvaluationFlags::values);
                        forceEvalSmearedCharge.distribute_local_to_global(
                          d_configForceVectorLinFEElectro);
                      }

                    FVselfSmearedChargesGammaAtomsElementalContribution(
                      forceContributionSmearedChargesGammaAtoms,
                      forceEvalSmearedCharge,
                      matrixFreeDataElectro,
                      cell,
                      gradVselfSmearedChargeQuads,
                      std::vector<dftfe::uInt>(
                        nonTrivialSmearedChargeAtomIdsMacroCell.begin(),
                        nonTrivialSmearedChargeAtomIdsMacroCell.end()),
                      dftPtr->d_bQuadAtomIdsAllAtoms,
                      smearedbQuads);
                  } // macrocell loop

                if (d_dftParams.floatingNuclearCharges)
                  {
                    accumulateForceContributionGammaAtomsFloating(
                      forceContributionSmearedChargesGammaAtoms,
                      d_forceAtomsFloating);
                  }
                else
                  distributeForceContributionFPSPLocalGammaAtoms(
                    forceContributionSmearedChargesGammaAtoms,
                    d_atomsForceDofsElectro,
                    d_constraintsNoneForceElectro,
                    d_configForceVectorLinFEElectro);
              } // kpt paral
          }     // bin loop
      }

    //
    // Second add configurational force contribution from the surface integral.
    // FIXME: The surface integral is incorrect incase of hanging nodes. The
    // temporary fix is to use a narrow Gaussian generator
    // (d_gaussianConstant=4.0 or 5.0) and self potential ball radius>1.5 Bohr
    // which is anyway required to solve the vself accurately- these parameters
    // assure that the contribution of the surface integral to the
    // configurational force is negligible (< 1e-6 Hartree/Bohr)
    //

    if (!d_dftParams.floatingNuclearCharges)
      {
        dealii::QGauss<3 - 1> faceQuadrature(d_dftParams.densityQuadratureRule);
        dealii::FEFaceValues<3> feForceFaceValues(
          FEForce,
          faceQuadrature,
          dealii::update_values | dealii::update_JxW_values |
            dealii::update_normal_vectors | dealii::update_quadrature_points);
        const dftfe::uInt faces_per_cell =
          dealii::GeometryInfo<3>::faces_per_cell;
        const dftfe::uInt numFaceQuadPoints = faceQuadrature.size();
        const dftfe::uInt forceDofsPerFace  = FEForce.dofs_per_face;
        const dftfe::uInt forceBaseIndicesPerFace =
          forceDofsPerFace / FEForce.components;
        dealii::Vector<double> elementalFaceForce(forceDofsPerFace);
        std::vector<dealii::types::global_dof_index> forceFaceLocalDofIndices(
          forceDofsPerFace);
        std::vector<dftfe::uInt> baseIndexFaceDofsForceVec(
          forceBaseIndicesPerFace * 3);
        dealii::Tensor<1, 3, double> baseIndexFaceForceVec;
        const dftfe::uInt            numberGlobalAtoms = atomLocations.size();

        for (dftfe::uInt iFaceDof = 0; iFaceDof < forceDofsPerFace; ++iFaceDof)
          {
            std::pair<dftfe::uInt, dftfe::uInt> baseComponentIndexPair =
              FEForce.face_system_to_component_index(iFaceDof);
            baseIndexFaceDofsForceVec[3 * baseComponentIndexPair.second +
                                      baseComponentIndexPair.first] = iFaceDof;
          }
        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                               std::vector<dftfe::uInt>>
                  &cellsVselfBallSurfacesDofHandler =
                    d_cellFacesVselfBallSurfacesDofHandlerElectro[iBin];
                const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                               std::vector<dftfe::uInt>>
                  &cellsVselfBallSurfacesDofHandlerForce =
                    d_cellFacesVselfBallSurfacesDofHandlerForceElectro[iBin];
                const distributedCPUVec<double> &iBinVselfField =
                  vselfBinsManagerElectro.getVselfFieldBins()[iBin];
                std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>::const_iterator iter1;
                std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>::const_iterator iter2;
                iter2 = cellsVselfBallSurfacesDofHandlerForce.begin();
                for (iter1 = cellsVselfBallSurfacesDofHandler.begin();
                     iter1 != cellsVselfBallSurfacesDofHandler.end();
                     ++iter1, ++iter2)
                  {
                    dealii::DoFHandler<3>::active_cell_iterator cell =
                      iter1->first;
                    const dftfe::Int closestAtomId =
                      d_cellsVselfBallsClosestAtomIdDofHandlerElectro
                        [iBin][cell->id()];
                    double           closestAtomCharge;
                    dealii::Point<3> closestAtomLocation;
                    if (closestAtomId < numberGlobalAtoms)
                      {
                        closestAtomLocation[0] =
                          atomLocations[closestAtomId][2];
                        closestAtomLocation[1] =
                          atomLocations[closestAtomId][3];
                        closestAtomLocation[2] =
                          atomLocations[closestAtomId][4];
                        if (d_dftParams.isPseudopotential)
                          closestAtomCharge = atomLocations[closestAtomId][1];
                        else
                          closestAtomCharge = atomLocations[closestAtomId][0];
                      }
                    else
                      {
                        const dftfe::Int imageId =
                          closestAtomId - numberGlobalAtoms;
                        closestAtomCharge = imageChargesTrunc[imageId];
                        closestAtomLocation[0] =
                          imagePositionsTrunc[imageId][0];
                        closestAtomLocation[1] =
                          imagePositionsTrunc[imageId][1];
                        closestAtomLocation[2] =
                          imagePositionsTrunc[imageId][2];
                      }

                    dealii::DoFHandler<3>::active_cell_iterator cellForce =
                      iter2->first;

                    const std::vector<dftfe::uInt> &dirichletFaceIds =
                      iter2->second;
                    for (dftfe::uInt index = 0; index < dirichletFaceIds.size();
                         index++)
                      {
                        const dftfe::uInt faceId = dirichletFaceIds[index];

                        feForceFaceValues.reinit(cellForce, faceId);
                        cellForce->face(faceId)->get_dof_indices(
                          forceFaceLocalDofIndices);
                        elementalFaceForce = 0;

                        for (dftfe::uInt ibase = 0;
                             ibase < forceBaseIndicesPerFace;
                             ++ibase)
                          {
                            baseIndexFaceForceVec = 0;
                            for (dftfe::uInt qPoint = 0;
                                 qPoint < numFaceQuadPoints;
                                 ++qPoint)
                              {
                                const dealii::Point<3> quadPoint =
                                  feForceFaceValues.quadrature_point(qPoint);
                                const dealii::Tensor<1, 3, double>
                                  dispClosestAtom =
                                    quadPoint - closestAtomLocation;
                                const double dist = dispClosestAtom.norm();
                                const dealii::Tensor<1, 3, double>
                                  gradVselfFaceQuadExact =
                                    closestAtomCharge * dispClosestAtom / dist /
                                    dist / dist;

                                baseIndexFaceForceVec -=
                                  eshelbyTensor::getVselfBallEshelbyTensor(
                                    gradVselfFaceQuadExact) *
                                  feForceFaceValues.normal_vector(qPoint) *
                                  feForceFaceValues.JxW(qPoint) *
                                  feForceFaceValues.shape_value(
                                    FEForce.face_to_cell_index(
                                      baseIndexFaceDofsForceVec[3 * ibase],
                                      faceId,
                                      cellForce->combined_face_orientation(
                                        faceId)),
                                    qPoint);

                              } // q point loop
                            for (dftfe::uInt idim = 0; idim < 3; idim++)
                              {
                                elementalFaceForce[baseIndexFaceDofsForceVec
                                                     [3 * ibase + idim]] =
                                  baseIndexFaceForceVec[idim];
                              }
                          } // base index loop
                        d_constraintsNoneForceElectro
                          .distribute_local_to_global(
                            elementalFaceForce,
                            forceFaceLocalDofIndices,
                            d_configForceVectorLinFEElectro);
                      } // face loop
                  }     // cell loop
              }         // kpt paral
          }             // bin loop
      }
  }



  // compute configurational force on the mesh nodes using linear shape function
  // generators
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::computeConfigurationalForcePhiExtLinFE()
  {
    FEEvaluationWrapperClass<3> forceEval(1,
                                          d_dftParams.densityQuadratureRule,
                                          dftPtr->matrix_free_data,
                                          d_forceDofHandlerIndex,
                                          0);

    FEEvaluationWrapperClass<1> eshelbyEval(
      d_dftParams.finiteElementPolynomialOrderElectrostatics,
      d_dftParams.densityQuadratureRule,
      dftPtr->d_matrixFreeDataPRefined,
      dftPtr->d_phiExtDofHandlerIndexElectro,
      0); // no constraints


    for (dftfe::uInt cell = 0; cell < dftPtr->matrix_free_data.n_cell_batches();
         ++cell)
      {
        forceEval.reinit(cell);
        eshelbyEval.reinit(cell);
        eshelbyEval.read_dof_values_plain(dftPtr->d_phiExt);
        eshelbyEval.evaluate(dealii::EvaluationFlags::values |
                             dealii::EvaluationFlags::gradients);
        for (dftfe::uInt q = 0; q < forceEval.n_q_points; ++q)
          {
            dealii::VectorizedArray<double> phiExt_q = eshelbyEval.get_value(q);
            dealii::Tensor<1, 3, dealii::VectorizedArray<double>> gradPhiExt_q =
              eshelbyEval.get_gradient(q);
            forceEval.submit_gradient(
              eshelbyTensor::getPhiExtEshelbyTensor(phiExt_q, gradPhiExt_q), q);
          }
        forceEval.integrate(dealii::EvaluationFlags::gradients);
        forceEval.distribute_local_to_global(
          d_configForceVectorLinFE); // also takes care of constraints
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::computeConfigurationalForceEselfNoSurfaceLinFE()
  {
    FEEvaluationWrapperClass<3> forceEval(1,
                                          d_dftParams.densityQuadratureRule,
                                          dftPtr->matrix_free_data,
                                          d_forceDofHandlerIndex,
                                          0);

    FEEvaluationWrapperClass<1> eshelbyEval(
      d_dftParams.finiteElementPolynomialOrderElectrostatics,
      d_dftParams.densityQuadratureRule,
      dftPtr->d_matrixFreeDataPRefined,
      dftPtr->d_phiExtDofHandlerIndexElectro,
      0); // no constraints

    for (dftfe::uInt iBin = 0;
         iBin < dftPtr->d_vselfBinsManager.getVselfFieldBins().size();
         iBin++)
      {
        for (dftfe::uInt cell = 0;
             cell < dftPtr->matrix_free_data.n_cell_batches();
             ++cell)
          {
            forceEval.reinit(cell);
            eshelbyEval.reinit(cell);
            eshelbyEval.read_dof_values_plain(
              dftPtr->d_vselfBinsManager.getVselfFieldBins()[iBin]);
            eshelbyEval.evaluate(dealii::EvaluationFlags::gradients);
            for (dftfe::uInt q = 0; q < forceEval.n_q_points; ++q)
              {
                dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                  gradVself_q = eshelbyEval.get_gradient(q);

                forceEval.submit_gradient(
                  eshelbyTensor::getVselfBallEshelbyTensor(gradVself_q), q);
              }
            forceEval.integrate(dealii::EvaluationFlags::gradients);
            forceEval.distribute_local_to_global(d_configForceVectorLinFE);
          }
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
