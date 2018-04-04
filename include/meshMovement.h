// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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

#ifndef meshMovement_H_
#define meshMovement_H_
#include "headers.h"
#include "constants.h"

namespace dftfe {

    using namespace dealii;

    class meshMovementClass
    {

    public:
      meshMovementClass(const MPI_Comm &mpi_comm_replica);
      meshMovementClass();
      virtual ~meshMovementClass() {}
      void init(const Triangulation<3,3> & triangulation, const std::vector<std::vector<double> > & domainBoundingVectors);
      void initMoved(const std::vector<std::vector<double> > & domainBoundingVectors);
      void findClosestVerticesToDestinationPoints(const std::vector<Point<3>> & destinationPoints,
						  std::vector<Point<3>> & closestTriaVertexToDestPointsLocation,
						  std::vector<Tensor<1,3,double>> & dispClosestTriaVerticesToDestPoints);
      void writeMesh(std::string meshFileName);
    protected:
      void initIncrementField();
      void finalizeIncrementField();
      void updateTriangulationVertices();
      //periodic matching sanity check and returns the pair<if negative jacobian, maximum inverse jacobian magnitude>
      std::pair<bool,double> movedMeshCheck();
      virtual std::pair<bool,double> moveMesh(const std::vector<Point<C_DIM> > & controlPointLocations,
					      const std::vector<Tensor<1,C_DIM,double> > & controlPointDisplacements,
					      double controllingParameter)=0;
      virtual void computeIncrement()=0;
      dealii::parallel::distributed::Vector<double>  d_incrementalDisplacementParallel;
      Vector<double> d_incrementalDisplacementSerial;
      bool d_isParallelMesh;

      //dealii based FE data structres
      FESystem<C_DIM>  FEMoveMesh;
      DoFHandler<C_DIM> d_dofHandlerMoveMesh;
      IndexSet   d_locally_owned_dofs;
      IndexSet   d_locally_relevant_dofs;
      ConstraintMatrix d_constraintsMoveMesh;
      ConstraintMatrix d_constraintsHangingNodes;
      std::vector<GridTools::PeriodicFacePair<typename DoFHandler<C_DIM>::cell_iterator> > d_periodicity_vector;
      std::vector<std::vector<double> >  d_domainBoundingVectors;

      //parallel objects
      MPI_Comm mpi_communicator;
      const unsigned int this_mpi_process;
      dealii::ConditionalOStream   pcout;
    };
}
#endif
