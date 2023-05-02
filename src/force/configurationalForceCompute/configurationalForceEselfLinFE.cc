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
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::computeConfigurationalForceEselfLinFE(
    const dealii::DoFHandler<3> &                    dofHandlerElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
    const dealii::MatrixFree<3, double> &            matrixFreeDataElectro,
    const unsigned int                               smearedChargeQuadratureId)
  {
    const std::vector<std::vector<double>> &atomLocations =
      dftPtr->atomLocations;
    const std::vector<std::vector<double>> &imagePositionsTrunc =
      dftPtr->d_imagePositionsTrunc;
    const std::vector<double> &imageChargesTrunc = dftPtr->d_imageChargesTrunc;
    //
    // First add configurational force contribution from the volume integral
    //
    dealii::QGauss<3> quadrature(
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>());
    dealii::FEValues<3> feForceValues(FEForce,
                                      quadrature,
                                      dealii::update_gradients |
                                        dealii::update_JxW_values);
    dealii::FEValues<3> feVselfValues(dofHandlerElectro.get_fe(),
                                      quadrature,
                                      dealii::update_gradients);
    const unsigned int  forceDofsPerCell = FEForce.dofs_per_cell;
    const unsigned int  forceBaseIndicesPerCell =
      forceDofsPerCell / FEForce.components;
    dealii::Vector<double> elementalForce(forceDofsPerCell);
    const unsigned int     numQuadPoints = quadrature.size();
    std::vector<dealii::types::global_dof_index> forceLocalDofIndices(
      forceDofsPerCell);
    const unsigned int numberBins =
      vselfBinsManagerElectro.getAtomIdsBins().size();
    std::vector<dealii::Tensor<1, 3, double>> gradVselfQuad(numQuadPoints);
    std::vector<unsigned int>    baseIndexDofsVec(forceBaseIndicesPerCell * 3);
    dealii::Tensor<1, 3, double> baseIndexForceVec;

    // kpoint group parallelization data structures
    const unsigned int numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

    const unsigned int kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
    std::vector<int> kptGroupLowHighPlusOneIndices;

    if (numberBins > 0)
      dftUtils::createKpointParallelizationIndices(
        dftPtr->interpoolcomm, numberBins, kptGroupLowHighPlusOneIndices);

    if (!d_dftParams.floatingNuclearCharges)
      {
        for (unsigned int ibase = 0; ibase < forceBaseIndicesPerCell; ++ibase)
          {
            for (unsigned int idim = 0; idim < 3; idim++)
              baseIndexDofsVec[3 * ibase + idim] =
                FEForce.component_to_system_index(idim, ibase);
          }

        for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
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
                    for (unsigned int ibase = 0;
                         ibase < forceBaseIndicesPerCell;
                         ++ibase)
                      {
                        baseIndexForceVec = 0;
                        for (unsigned int qPoint = 0; qPoint < numQuadPoints;
                             ++qPoint)
                          {
                            baseIndexForceVec +=
                              eshelbyTensor::getVselfBallEshelbyTensor(
                                gradVselfQuad[qPoint]) *
                              feForceValues.shape_grad(
                                baseIndexDofsVec[3 * ibase], qPoint) *
                              feForceValues.JxW(qPoint);
                          } // q point loop
                        for (unsigned int idim = 0; idim < 3; idim++)
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
        const std::map<int, std::set<int>> &atomIdsBins =
          vselfBinsManagerElectro.getAtomIdsBins();

        dealii::FEEvaluation<3, -1, 1, 3> forceEvalSmearedCharge(
          matrixFreeDataElectro,
          d_forceDofHandlerIndexElectro,
          smearedChargeQuadratureId);

        dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
        const unsigned int                          numQuadPointsSmearedb =
          forceEvalSmearedCharge.n_q_points;

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            zeroTensor[idim] = dealii::make_vectorized_array(0.0);
          }

        dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
            }

        dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
          numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradVselfSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);

        std::map<unsigned int, std::vector<double>>
          forceContributionSmearedChargesGammaAtoms;

        for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                dealii::FEEvaluation<3, -1> vselfEvalSmearedCharge(
                  matrixFreeDataElectro,
                  dftPtr->d_binsStartDofHandlerIndexElectro + 4 * iBin,
                  smearedChargeQuadratureId);

                const std::set<int> &atomIdsInBin =
                  atomIdsBins.find(iBin)->second;
                forceContributionSmearedChargesGammaAtoms.clear();
                for (unsigned int cell = 0;
                     cell < matrixFreeDataElectro.n_macro_cells();
                     ++cell)
                  {
                    std::set<unsigned int>
                                       nonTrivialSmearedChargeAtomIdsMacroCell;
                    const unsigned int numSubCells =
                      matrixFreeDataElectro.n_components_filled(cell);
                    for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();
                        const std::vector<unsigned int> &temp =
                          dftPtr->d_bCellNonTrivialAtomIdsBins[iBin]
                            .find(subCellId)
                            ->second;
                        for (int i = 0; i < temp.size(); i++)
                          nonTrivialSmearedChargeAtomIdsMacroCell.insert(
                            temp[i]);
                      }

                    if (nonTrivialSmearedChargeAtomIdsMacroCell.size() == 0)
                      continue;

                    forceEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.read_dof_values_plain(
                      vselfBinsManagerElectro.getVselfFieldBins()[iBin]);
                    vselfEvalSmearedCharge.evaluate(false, true);

                    std::fill(smearedbQuads.begin(),
                              smearedbQuads.end(),
                              dealii::make_vectorized_array(0.0));
                    std::fill(gradVselfSmearedChargeQuads.begin(),
                              gradVselfSmearedChargeQuads.end(),
                              zeroTensor);

                    bool isCellNonTrivial = false;
                    for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();

                        const std::vector<int> &bQuadAtomIdsCell =
                          dftPtr->d_bQuadAtomIdsAllAtoms.find(subCellId)
                            ->second;
                        const std::vector<double> &bQuadValuesCell =
                          dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;

                        for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
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

                    for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
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
                        forceEvalSmearedCharge.integrate(true, false);
                        forceEvalSmearedCharge.distribute_local_to_global(
                          d_configForceVectorLinFEElectro);
                      }

                    FVselfSmearedChargesGammaAtomsElementalContribution(
                      forceContributionSmearedChargesGammaAtoms,
                      forceEvalSmearedCharge,
                      matrixFreeDataElectro,
                      cell,
                      gradVselfSmearedChargeQuads,
                      std::vector<unsigned int>(
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
        dealii::QGauss<3 - 1> faceQuadrature(
          C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>());
        dealii::FEFaceValues<3> feForceFaceValues(
          FEForce,
          faceQuadrature,
          dealii::update_values | dealii::update_JxW_values |
            dealii::update_normal_vectors | dealii::update_quadrature_points);
        const unsigned int faces_per_cell =
          dealii::GeometryInfo<3>::faces_per_cell;
        const unsigned int numFaceQuadPoints = faceQuadrature.size();
        const unsigned int forceDofsPerFace  = FEForce.dofs_per_face;
        const unsigned int forceBaseIndicesPerFace =
          forceDofsPerFace / FEForce.components;
        dealii::Vector<double> elementalFaceForce(forceDofsPerFace);
        std::vector<dealii::types::global_dof_index> forceFaceLocalDofIndices(
          forceDofsPerFace);
        std::vector<unsigned int> baseIndexFaceDofsForceVec(
          forceBaseIndicesPerFace * 3);
        dealii::Tensor<1, 3, double> baseIndexFaceForceVec;
        const unsigned int           numberGlobalAtoms = atomLocations.size();

        for (unsigned int iFaceDof = 0; iFaceDof < forceDofsPerFace; ++iFaceDof)
          {
            std::pair<unsigned int, unsigned int> baseComponentIndexPair =
              FEForce.face_system_to_component_index(iFaceDof);
            baseIndexFaceDofsForceVec[3 * baseComponentIndexPair.second +
                                      baseComponentIndexPair.first] = iFaceDof;
          }
        for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                               std::vector<unsigned int>>
                  &cellsVselfBallSurfacesDofHandler =
                    d_cellFacesVselfBallSurfacesDofHandlerElectro[iBin];
                const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                               std::vector<unsigned int>>
                  &cellsVselfBallSurfacesDofHandlerForce =
                    d_cellFacesVselfBallSurfacesDofHandlerForceElectro[iBin];
                const distributedCPUVec<double> &iBinVselfField =
                  vselfBinsManagerElectro.getVselfFieldBins()[iBin];
                std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>::const_iterator iter1;
                std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<unsigned int>>::const_iterator iter2;
                iter2 = cellsVselfBallSurfacesDofHandlerForce.begin();
                for (iter1 = cellsVselfBallSurfacesDofHandler.begin();
                     iter1 != cellsVselfBallSurfacesDofHandler.end();
                     ++iter1, ++iter2)
                  {
                    dealii::DoFHandler<3>::active_cell_iterator cell =
                      iter1->first;
                    const int closestAtomId =
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
                        const int imageId = closestAtomId - numberGlobalAtoms;
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

                    const std::vector<unsigned int> &dirichletFaceIds =
                      iter2->second;
                    for (unsigned int index = 0;
                         index < dirichletFaceIds.size();
                         index++)
                      {
                        const unsigned int faceId = dirichletFaceIds[index];

                        feForceFaceValues.reinit(cellForce, faceId);
                        cellForce->face(faceId)->get_dof_indices(
                          forceFaceLocalDofIndices);
                        elementalFaceForce = 0;

                        for (unsigned int ibase = 0;
                             ibase < forceBaseIndicesPerFace;
                             ++ibase)
                          {
                            baseIndexFaceForceVec = 0;
                            for (unsigned int qPoint = 0;
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
                                      cellForce->face_orientation(faceId),
                                      cellForce->face_flip(faceId),
                                      cellForce->face_rotation(faceId)),
                                    qPoint);

                              } // q point loop
                            for (unsigned int idim = 0; idim < 3; idim++)
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
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::computeConfigurationalForcePhiExtLinFE()
  {
    dealii::FEEvaluation<
      3,
      1,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      3>
      forceEval(dftPtr->matrix_free_data, d_forceDofHandlerIndex, 0);

    dealii::FEEvaluation<
      3,
      FEOrderElectro,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1>
      eshelbyEval(dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_phiExtDofHandlerIndexElectro,
                  0); // no constraints


    for (unsigned int cell = 0; cell < dftPtr->matrix_free_data.n_macro_cells();
         ++cell)
      {
        forceEval.reinit(cell);
        eshelbyEval.reinit(cell);
        eshelbyEval.read_dof_values_plain(dftPtr->d_phiExt);
        eshelbyEval.evaluate(true, true);
        for (unsigned int q = 0; q < forceEval.n_q_points; ++q)
          {
            dealii::VectorizedArray<double> phiExt_q = eshelbyEval.get_value(q);
            dealii::Tensor<1, 3, dealii::VectorizedArray<double>> gradPhiExt_q =
              eshelbyEval.get_gradient(q);
            forceEval.submit_gradient(
              eshelbyTensor::getPhiExtEshelbyTensor(phiExt_q, gradPhiExt_q), q);
          }
        forceEval.integrate(false, true);
        forceEval.distribute_local_to_global(
          d_configForceVectorLinFE); // also takes care of constraints
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder,
             FEOrderElectro>::computeConfigurationalForceEselfNoSurfaceLinFE()
  {
    dealii::FEEvaluation<
      3,
      1,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      3>
      forceEval(dftPtr->matrix_free_data, d_forceDofHandlerIndex, 0);

    dealii::FEEvaluation<
      3,
      FEOrderElectro,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1>
      eshelbyEval(dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_phiExtDofHandlerIndexElectro,
                  0); // no constraints

    for (unsigned int iBin = 0;
         iBin < dftPtr->d_vselfBinsManager.getVselfFieldBins().size();
         iBin++)
      {
        for (unsigned int cell = 0;
             cell < dftPtr->matrix_free_data.n_macro_cells();
             ++cell)
          {
            forceEval.reinit(cell);
            eshelbyEval.reinit(cell);
            eshelbyEval.read_dof_values_plain(
              dftPtr->d_vselfBinsManager.getVselfFieldBins()[iBin]);
            eshelbyEval.evaluate(false, true);
            for (unsigned int q = 0; q < forceEval.n_q_points; ++q)
              {
                dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                  gradVself_q = eshelbyEval.get_gradient(q);

                forceEval.submit_gradient(
                  eshelbyTensor::getVselfBallEshelbyTensor(gradVself_q), q);
              }
            forceEval.integrate(false, true);
            forceEval.distribute_local_to_global(d_configForceVectorLinFE);
          }
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
