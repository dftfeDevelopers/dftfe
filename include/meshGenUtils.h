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
// @author Sambit Das, Phani Motamarri
//

#ifndef meshGenUtils_H_
#define meshGenUtils_H_

#include <headers.h>

namespace dftfe
{
  using namespace dealii;

  namespace meshGenUtils
  {
    inline void
    cross_product(std::vector<double> &a,
                  std::vector<double> &b,
                  std::vector<double> &crossProduct)
    {
      crossProduct.resize(a.size(), 0.0);

      crossProduct[0] = a[1] * b[2] - a[2] * b[1];
      crossProduct[1] = a[2] * b[0] - a[0] * b[2];
      crossProduct[2] = a[0] * b[1] - a[1] * b[0];
    }

    inline void
    computePeriodicFaceNormals(
      std::vector<std::vector<double>> &latticeVectors,
      std::vector<std::vector<double>> &periodicFaceNormals)
    {
      // resize periodic face normal
      periodicFaceNormals.resize(3);


      // evaluate cross product between first two lattice vectors
      cross_product(latticeVectors[0],
                    latticeVectors[1],
                    periodicFaceNormals[2]);

      // evaluate cross product between second two lattice vectors
      cross_product(latticeVectors[1],
                    latticeVectors[2],
                    periodicFaceNormals[0]);


      // evaluate cross product between third and first two lattice vectors
      cross_product(latticeVectors[2],
                    latticeVectors[0],
                    periodicFaceNormals[1]);
    }

    inline void
    computeOffsetVectors(std::vector<std::vector<double>> &latticeVectors,
                         std::vector<Tensor<1, 3>> &       offsetVectors)
    {
      // create unitVectorsXYZ
      std::vector<std::vector<double>> unitVectorsXYZ;
      unitVectorsXYZ.resize(3);

      for (int i = 0; i < 3; ++i)
        {
          unitVectorsXYZ[i].resize(3, 0.0);
          unitVectorsXYZ[i][i] = 0.0;
        }


      // resize offset vectors
      offsetVectors.resize(3);

      for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
            {
              offsetVectors[i][j] = unitVectorsXYZ[i][j] - latticeVectors[i][j];
            }
        }
    }


    inline double getCosineAngle(Tensor<1, 3> &       Vector1,
                                 std::vector<double> &Vector2)
    {
      double dotProduct = Vector1[0] * Vector2[0] + Vector1[1] * Vector2[1] +
                          Vector1[2] * Vector2[2];
      double lengthVector1Sq =
        (Vector1[0] * Vector1[0] + Vector1[1] * Vector1[1] +
         Vector1[2] * Vector1[2]);
      double lengthVector2Sq =
        (Vector2[0] * Vector2[0] + Vector2[1] * Vector2[1] +
         Vector2[2] * Vector2[2]);

      double angle = dotProduct / sqrt(lengthVector1Sq * lengthVector2Sq);

      return angle;
    }


    inline double
    getCosineAngle(std::vector<double> &Vector1, std::vector<double> &Vector2)
    {
      double dotProduct = Vector1[0] * Vector2[0] + Vector1[1] * Vector2[1] +
                          Vector1[2] * Vector2[2];
      double lengthVector1Sq =
        (Vector1[0] * Vector1[0] + Vector1[1] * Vector1[1] +
         Vector1[2] * Vector1[2]);
      double lengthVector2Sq =
        (Vector2[0] * Vector2[0] + Vector2[1] * Vector2[1] +
         Vector2[2] * Vector2[2]);

      double angle = dotProduct / sqrt(lengthVector1Sq * lengthVector2Sq);

      return angle;
    }


    inline void markPeriodicFacesNonOrthogonal(
      Triangulation<3, 3> &             triangulation,
      std::vector<std::vector<double>> &latticeVectors,
      const MPI_Comm &                  mpiCommParent)
    {
      dealii::ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(mpiCommParent) == 0));
      std::vector<std::vector<double>> periodicFaceNormals;
      std::vector<Tensor<1, 3>>        offsetVectors;
      // bool periodicX = dftParameters::periodicX, periodicY =
      // dftParameters::periodicY, periodicZ=dftParameters::periodicZ;

      // compute periodic face normals from lattice vector information
      computePeriodicFaceNormals(latticeVectors, periodicFaceNormals);

      // compute offset vectors such that lattice vectors plus offset vectors
      // are alligned along spatial x,y,z directions
      computeOffsetVectors(latticeVectors, offsetVectors);

      if (dftParameters::verbosity >= 4)
        {
          pcout << "Periodic Face Normals 1: " << periodicFaceNormals[0][0]
                << " " << periodicFaceNormals[0][1] << " "
                << periodicFaceNormals[0][2] << std::endl;
          pcout << "Periodic Face Normals 2: " << periodicFaceNormals[1][0]
                << " " << periodicFaceNormals[1][1] << " "
                << periodicFaceNormals[1][2] << std::endl;
          pcout << "Periodic Face Normals 3: " << periodicFaceNormals[2][0]
                << " " << periodicFaceNormals[2][1] << " "
                << periodicFaceNormals[2][2] << std::endl;
        }

      QGauss<2>       quadratureFace_formula(2);
      FESystem<3>     FE(FE_Q<3>(QGaussLobatto<1>(2)), 1);
      FEFaceValues<3> feFace_values(FE,
                                    quadratureFace_formula,
                                    update_normal_vectors);

      typename Triangulation<3, 3>::active_cell_iterator cell, endc;
      //
      // mark faces
      //
      // const unsigned int px=dftParameters::periodicX,
      // py=dftParameters::periodicX, pz=dftParameters::periodicX;
      //
      cell = triangulation.begin_active(), endc = triangulation.end();
      const std::array<int, 3> periodic = {dftParameters::periodicX,
                                           dftParameters::periodicY,
                                           dftParameters::periodicZ};
      for (; cell != endc; ++cell)
        {
          for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
            {
              const Point<3> face_center = cell->face(f)->center();
              if (cell->face(f)->at_boundary())
                {
                  feFace_values.reinit(cell, f);
                  Tensor<1, 3> faceNormalVector =
                    feFace_values.normal_vector(0);

                  // std::cout<<"Face normal vector: "<<faceNormalVector[0]<<"
                  // "<<faceNormalVector[1]<<"
                  // "<<faceNormalVector[2]<<std::endl; pcout<<"Angle :
                  // "<<getCosineAngle(faceNormalVector,periodicFaceNormals[0])<<"
                  // "<<getCosineAngle(faceNormalVector,periodicFaceNormals[1])<<"
                  // "<<getCosineAngle(faceNormalVector,periodicFaceNormals[2])<<std::endl;

                  unsigned int i = 1;

                  for (unsigned int d = 0; d < 3; ++d)
                    {
                      if (periodic[d] == 1)
                        {
                          if (std::abs(getCosineAngle(faceNormalVector,
                                                      periodicFaceNormals[d]) -
                                       1.0) < 1.0e-05)
                            cell->face(f)->set_boundary_id(i);
                          else if (std::abs(
                                     getCosineAngle(faceNormalVector,
                                                    periodicFaceNormals[d]) +
                                     1.0) < 1.0e-05)
                            cell->face(f)->set_boundary_id(i + 1);
                          i = i + 2;
                        }
                    }
                }
            }
        }

      std::vector<int> periodicDirectionVector;

      for (unsigned int d = 0; d < 3; ++d)
        {
          if (periodic[d] == 1)
            {
              periodicDirectionVector.push_back(d);
            }
        }

      // pcout << "Done with Boundary Flags\n";
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<3, 3>::cell_iterator>>
        periodicity_vector;
      for (int i = 0; i < std::accumulate(periodic.begin(), periodic.end(), 0);
           ++i)
        {
          GridTools::collect_periodic_faces(
            triangulation,
            /*b_id1*/ 2 * i + 1,
            /*b_id2*/ 2 * i + 2,
            /*direction*/ periodicDirectionVector[i],
            periodicity_vector,
            offsetVectors[periodicDirectionVector[i]]);
          // GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1,
          // /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
        }
      triangulation.add_periodicity(periodicity_vector);

      if (dftParameters::verbosity >= 4)
        pcout << "Periodic Facepairs size: " << periodicity_vector.size()
              << std::endl;
      /*
         for(unsigned int i=0; i< periodicity_vector.size(); ++i)
         {
         if (!periodicity_vector[i].cell[0]->active() ||
         !periodicity_vector[i].cell[1]->active()) continue; if
         (periodicity_vector[i].cell[0]->is_artificial() ||
         periodicity_vector[i].cell[1]->is_artificial()) continue;

         std::cout << "matched face pairs: "<<
         periodicity_vector[i].cell[0]->face(periodicity_vector[i].face_idx[0])->boundary_id()
         << " "<<
         periodicity_vector[i].cell[1]->face(periodicity_vector[i].face_idx[1])->boundary_id()<<std::endl;
         }
       */
    }

  } // namespace meshGenUtils
} // namespace dftfe
#endif
