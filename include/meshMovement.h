// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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


#ifndef meshMovement_H_
#define meshMovement_H_
#include "headers.h"
#include "constants.h"

namespace dftfe {
    using namespace dealii;

    /**
     * @brief Base class to move triangulation vertices
     *
     * @author Sambit Das
     */
    class meshMovementClass
    {

    public:
      /** @brief Constructor
       *
       *  @param[in] mpi_comm_replica mpi communicator for domain decomposition
       */
      meshMovementClass(const MPI_Comm &mpi_comm_replica);

      virtual ~meshMovementClass() {}

      /** @brief Initializes the required data-structures for a given triangulation
       *
       *  @param[in] triangulation triangulation object whose nodes are to be moved
       *  @param[in] serial triangulation to create constraints from serial dofHandler (temporary fix)
       *  @param[in] domainBoundingVectors domain vectors of the domain corresponding to
       *  the triangulation object.
       */
      void init(Triangulation<3,3> & triangulation,
		Triangulation<3,3> & serialTriangulation,      
	        const std::vector<std::vector<double> > & domainBoundingVectors);

      /** @brief Re-initializes the required data-structures for a given triangulation
       *
       *  @param[in] domainBoundingVectors current domain vectors of the domain corresponding to
       *  the triangulation object.
       */
      void initMoved(const std::vector<std::vector<double> > & domainBoundingVectors);

      /** @brief Finds the closest triangulation vertices to a given vector of position coordinates
       *
       *  @param[in] destinationPoints vector of points in cartesian coordinates (origin at center of
       *  the domain) to which closest triangulation vertices are desired.
       *  @param[out] closestTriaVertexToDestPointsLocation vector of positions of the closest triangulation v
       *  vertices.
       *  @param[out] dispClosestTriaVerticesToDestPoints vector of displacements of the destinationPoints
       *  from the closest triangulation vertices.
       */
      void findClosestVerticesToDestinationPoints(const std::vector<Point<3>> & destinationPoints,
						  std::vector<Point<3>> & closestTriaVertexToDestPointsLocation,
						  std::vector<Tensor<1,3,double>> & dispClosestTriaVerticesToDestPoints);

    protected:
      /// Initializes the parallel layout of d_incrementalDisplacementParallel
      void initIncrementField();

      /// Takes care of communicating the movement of triangulation vertices on processor boundaries,
      /// and also takes care of hanging nodes and periodic constraints
      void finalizeIncrementField();

      /// Function which updates the locally relevant triangulation vertices
      void updateTriangulationVertices();

      /// Function which moves subdivided mesh
      void moveSubdividedMesh();

      /// Performs periodic matching sanity check and returns the pair<if negative jacobian, maximum inverse jacobian magnitude>
      std::pair<bool,double> movedMeshCheck();

      virtual std::pair<bool,double> moveMesh(const std::vector<Point<C_DIM> > & controlPointLocations,
					      const std::vector<Tensor<1,C_DIM,double> > & controlPointDisplacements,
					      const double controllingParameter,
					      const bool moveSubdivided = false)=0;
      virtual void computeIncrement()=0;

      /// vector of displacements of the triangulation vertices
      //Vector<double> d_incrementalDisplacement;
      vectorType d_incrementalDisplacement;

      bool d_isParallelMesh;

      //dealii based FE data structres
      FESystem<C_DIM>  FEMoveMesh;
      DoFHandler<C_DIM> d_dofHandlerMoveMesh;
      parallel::distributed::Triangulation<3> * d_triaPtr;
      Triangulation<3,3>  * d_triaPtrSerial;
      IndexSet   d_locally_owned_dofs;
      IndexSet   d_locally_relevant_dofs;
      ConstraintMatrix d_constraintsMoveMesh;
      std::vector<GridTools::PeriodicFacePair<typename DoFHandler<C_DIM>::cell_iterator> > d_periodicity_vector;
      std::vector<std::vector<double> >  d_domainBoundingVectors;

      //parallel objects
      MPI_Comm mpi_communicator;
      const unsigned int this_mpi_process;
      dealii::ConditionalOStream   pcout;
    };
}
#endif
