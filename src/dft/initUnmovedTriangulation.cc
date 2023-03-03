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
//
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//



#ifdef USE_COMPLEX
#  include "initkPointData.cc"
#endif
//
// source file for dft class initializations
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void dftClass<FEOrder, FEOrderElectro>::initUnmovedTriangulation(
  parallel::distributed::Triangulation<3> &triangulation)
{
  computing_timer.enter_subsection("unmoved setup");

  // initialize affine transformation object (must be done on unmoved
  // triangulation)
  if (d_dftParamsPtr->isCellStress)
    d_affineTransformMesh.init(triangulation,
                               d_mesh.getSerialMeshUnmoved(),
                               d_domainBoundingVectors);

  // initialize meshMovementGaussianClass object (must be done on unmoved
  // triangulation) when not using floating charges
  if (!d_dftParamsPtr->floatingNuclearCharges)
    d_gaussianMovePar.init(triangulation,
                           d_mesh.getSerialMeshUnmoved(),
                           d_domainBoundingVectors);

  if (d_dftParamsPtr->verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(
      mpi_communicator,
      "Inititialization of meshmovement class objects completed");
  //
  // initialize FE objects
  //
  dofHandler.clear();
  dofHandlerEigen.clear();
  dofHandler.initialize(triangulation, FE);
  dofHandlerEigen.initialize(triangulation, FEEigen);
  dofHandler.distribute_dofs(FE);
  dofHandlerEigen.distribute_dofs(FEEigen);

  if (d_dftParamsPtr->verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator, "Distributed dofs");
  //
  // extract locally owned dofs
  //
  locally_owned_dofs = dofHandler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandler, locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       dofHandler,
                                       d_supportPoints);


  locally_owned_dofsEigen = dofHandlerEigen.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandlerEigen,
                                          locally_relevant_dofsEigen);
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       dofHandlerEigen,
                                       d_supportPointsEigen);


  //
  // Extract real and imag DOF indices from the global vector - this will be
  // needed in XHX operation, etc.
  //
#ifdef USE_COMPLEX
  FEValuesExtractors::Scalar real(0); // For Eigen
  ComponentMask    componentMaskForRealDOF = FEEigen.component_mask(real);
  dealii::IndexSet selectedDofsReal =
    DoFTools::extract_dofs(dofHandlerEigen, componentMaskForRealDOF);
  std::vector<IndexSet::size_type> local_dof_indices(
    locally_owned_dofsEigen.n_elements());
  locally_owned_dofsEigen.fill_index_vector(local_dof_indices);
  local_dof_indicesReal.clear();
  localProc_dof_indicesReal.clear();
  local_dof_indicesImag.clear();
  localProc_dof_indicesImag.clear();
  for (unsigned int i = 0; i < locally_owned_dofsEigen.n_elements(); i++)
    {
      if (selectedDofsReal.is_element(
            locally_owned_dofsEigen.nth_index_in_set(i)))
        {
          local_dof_indicesReal.push_back(local_dof_indices[i]);
          localProc_dof_indicesReal.push_back(i);
        }
      else
        {
          local_dof_indicesImag.push_back(local_dof_indices[i]);
          localProc_dof_indicesImag.push_back(i);
        }
    }
#endif

  if (d_dftParamsPtr->verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator, "Extracted indices");

  // std::cout<< " procId: "<< this_mpi_process << " ,locallly_owned_dofs:
  // "<<dofHandler.n_locally_owned_dofs()<<std::endl;

  //
  // constraints
  //

  //
  // hanging node constraints
  //
  constraintsNone.clear();
  constraintsNoneEigen.clear();
  constraintsNone.reinit(locally_relevant_dofs);
  constraintsNoneEigen.reinit(locally_relevant_dofsEigen);
  DoFTools::make_hanging_node_constraints(dofHandler, constraintsNone);
  DoFTools::make_hanging_node_constraints(dofHandlerEigen,
                                          constraintsNoneEigen);

  // create unitVectorsXYZ
  std::vector<std::vector<double>> unitVectorsXYZ;
  unitVectorsXYZ.resize(3);

  for (int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3, 0.0);
      unitVectorsXYZ[i][i] = 0.0;
    }

  std::vector<Tensor<1, 3>> offsetVectors;
  // resize offset vectors
  offsetVectors.resize(3);

  for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
        {
          offsetVectors[i][j] =
            unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];
        }
    }

  std::vector<
    GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator>>
    periodicity_vector2, periodicity_vector2Eigen;

  std::vector<int>         periodicDirectionVector;
  const std::array<int, 3> periodic = {d_dftParamsPtr->periodicX,
                                       d_dftParamsPtr->periodicY,
                                       d_dftParamsPtr->periodicZ};
  for (unsigned int d = 0; d < 3; ++d)
    {
      if (periodic[d] == 1)
        {
          periodicDirectionVector.push_back(d);
        }
    }


  for (int i = 0; i < std::accumulate(periodic.begin(), periodic.end(), 0); ++i)
    {
      GridTools::collect_periodic_faces(
        dofHandler,
        /*b_id1*/ 2 * i + 1,
        /*b_id2*/ 2 * i + 2,
        /*direction*/ periodicDirectionVector[i],
        periodicity_vector2,
        offsetVectors[periodicDirectionVector[i]]);
      GridTools::collect_periodic_faces(
        dofHandlerEigen,
        /*b_id1*/ 2 * i + 1,
        /*b_id2*/ 2 * i + 2,
        /*direction*/ periodicDirectionVector[i],
        periodicity_vector2Eigen,
        offsetVectors[periodicDirectionVector[i]]);
    }

  DoFTools::make_periodicity_constraints<DoFHandler<3>>(periodicity_vector2,
                                                        constraintsNone);
  DoFTools::make_periodicity_constraints<DoFHandler<3>>(
    periodicity_vector2Eigen, constraintsNoneEigen);



  constraintsNone.close();
  constraintsNoneEigen.close();

  //
  // create a constraint matrix without only hanging node constraints
  //
  d_noConstraints.clear();
  dealii::AffineConstraints<double> noConstraintsEigen;
  d_noConstraints.reinit(locally_relevant_dofs);
  noConstraintsEigen.reinit(locally_relevant_dofsEigen);
  DoFTools::make_hanging_node_constraints(dofHandler, d_noConstraints);
  DoFTools::make_hanging_node_constraints(dofHandlerEigen, noConstraintsEigen);
  d_noConstraints.close();
  noConstraintsEigen.close();

  if (d_dftParamsPtr->createConstraintsFromSerialDofhandler)
    {
#ifdef USE_COMPLEX
      vectorTools::createParallelConstraintMatrixFromSerial(
        d_mesh.getSerialMeshUnmoved(),
        dofHandler,
        d_mpiCommParent,
        mpi_communicator,
        d_domainBoundingVectors,
        constraintsNone,
        d_noConstraints,
        d_dftParamsPtr->verbosity,
        d_dftParamsPtr->periodicX,
        d_dftParamsPtr->periodicY,
        d_dftParamsPtr->periodicZ);

      vectorTools::createParallelConstraintMatrixFromSerial(
        d_mesh.getSerialMeshUnmoved(),
        dofHandlerEigen,
        d_mpiCommParent,
        mpi_communicator,
        d_domainBoundingVectors,
        constraintsNoneEigen,
        noConstraintsEigen,
        d_dftParamsPtr->verbosity,
        d_dftParamsPtr->periodicX,
        d_dftParamsPtr->periodicY,
        d_dftParamsPtr->periodicZ);
#else
      vectorTools::createParallelConstraintMatrixFromSerial(
        d_mesh.getSerialMeshUnmoved(),
        dofHandler,
        d_mpiCommParent,
        mpi_communicator,
        d_domainBoundingVectors,
        constraintsNone,
        d_noConstraints,
        d_dftParamsPtr->verbosity,
        d_dftParamsPtr->periodicX,
        d_dftParamsPtr->periodicY,
        d_dftParamsPtr->periodicZ);
      constraintsNoneEigen.clear();
      constraintsNoneEigen.reinit(locally_relevant_dofs);
      constraintsNoneEigen.merge(constraintsNone,
                                 dealii::AffineConstraints<double>::
                                   MergeConflictBehavior::right_object_wins);
      constraintsNoneEigen.close();

      noConstraintsEigen.clear();
      noConstraintsEigen.reinit(locally_relevant_dofs);
      noConstraintsEigen.merge(d_noConstraints,
                               dealii::AffineConstraints<double>::
                                 MergeConflictBehavior::right_object_wins);
      noConstraintsEigen.close();
#endif
    }

  //
  // create 2p DoFHandler if Kerker density mixing is on
  //
  createpRefinedDofHandler(triangulation);

  if (d_dftParamsPtr->verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                      "Created the basic constraint matrices");

  forcePtr->initUnmoved(triangulation,
                        d_mesh.getSerialMeshUnmoved(),
                        d_domainBoundingVectors,
                        false);
  forcePtr->initUnmoved(triangulation,
                        d_mesh.getSerialMeshUnmoved(),
                        d_domainBoundingVectors,
                        true);

  if (d_dftParamsPtr->verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator, "Force initUnmoved");


  excManager::createExcClassObj(d_dftParamsPtr->xc_id,
                                (d_dftParamsPtr->spinPolarized == 1) ? true :
                                                                       false,
                                0.0,   // exx factor
                                false, // scale exchange
                                1.0,   // scale exchange factor
                                true,  // computeCorrelation
                                &funcX,
                                &funcC,
                                excFunctionalPtr);
  /*
  exceptParamX = xc_func_init(&funcX, XC_LDA_X, isSpinPolarized);
      exceptParamC = xc_func_init(&funcC, XC_LDA_C_PZ, isSpinPolarized);
  excFunctionalPtr = new excWavefunctionNoneClass (densityFamilyType::LDA,
                                                     funcX,
                                                     funcC,
                 0.0,
                                                     false,
                                                     true,
                                                     1.0);
                 */

  computing_timer.leave_subsection("unmoved setup");
}
