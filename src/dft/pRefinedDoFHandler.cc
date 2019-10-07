// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri
//

//source file for all charge calculations

//
//compute total charge using quad point values
//
template <unsigned int FEOrder>
void dftClass<FEOrder>::createpRefinedDofHandler(dealii::parallel::distributed::Triangulation<3> & triaObject)
{
  d_dofHandlerPRefined.initialize(triaObject,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(2*FEOrder+1)));
  d_dofHandlerPRefined.distribute_dofs(d_dofHandlerPRefined.get_fe());

  dealii::IndexSet locallyRelevantDofs;
  dealii::DoFTools::extract_locally_relevant_dofs(d_dofHandlerPRefined, locallyRelevantDofs);

  dealii::ConstraintMatrix onlyHangingNodeConstraints;
  onlyHangingNodeConstraints.reinit(locallyRelevantDofs);
  dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerPRefined, onlyHangingNodeConstraints);
  onlyHangingNodeConstraints.close();

  d_constraintsPRefined.reinit(locallyRelevantDofs);
  dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerPRefined, d_constraintsPRefined);

  std::vector<std::vector<double> > unitVectorsXYZ;
  unitVectorsXYZ.resize(3);

  for(unsigned int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3,0.0);
      unitVectorsXYZ[i][i] = 0.0;
    }

  std::vector<Tensor<1,3> > offsetVectors;
  //resize offset vectors
  offsetVectors.resize(3);

  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int j = 0; j < 3; ++j)
      offsetVectors[i][j] = unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];

  std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<3>::cell_iterator> > periodicity_vector2;
  const std::array<unsigned int,3> periodic = {dftParameters::periodicX, dftParameters::periodicY, dftParameters::periodicZ};

  std::vector<int> periodicDirectionVector;
  for (unsigned int  d= 0; d < 3; ++d)
    {
      if (periodic[d]==1)
	{
	  periodicDirectionVector.push_back(d);
	}
    }

  for (unsigned int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
    GridTools::collect_periodic_faces(d_dofHandlerPRefined, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ periodicDirectionVector[i], periodicity_vector2,offsetVectors[periodicDirectionVector[i]]);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3> >(periodicity_vector2, d_constraintsPRefined);
  d_constraintsPRefined.close();

  

}


template <unsigned int FEOrder>
void dftClass<FEOrder>::initpRefinedObjects()
{
  d_dofHandlerPRefined.distribute_dofs(d_dofHandlerPRefined.get_fe());

  //matrix free data structure
  typename dealii::MatrixFree<3>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;

  //clear existing constraints matrix vector
  std::vector<const dealii::ConstraintMatrix*> matrixFreeConstraintsInputVector;
  matrixFreeConstraintsInputVector.push_back(&d_constraintsPRefined);

  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
  for(unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
    matrixFreeDofHandlerVectorInput.push_back(&d_dofHandlerPRefined);

  std::vector<Quadrature<1> > quadratureVector;
  quadratureVector.push_back(QGauss<1>(C_num1DQuad<2*FEOrder>()));
  quadratureVector.push_back(QGauss<1>(C_num1DQuad<FEOrder>()));

  d_matrixFreeDataPRefined.reinit(matrixFreeDofHandlerVectorInput,
				  matrixFreeConstraintsInputVector,
				  quadratureVector,
				  additional_data);

  
}
