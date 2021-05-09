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
// @author Sambit Das
//

#include <dftParameters.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsInternalCUDA.h>

/** @file linearAlgebraOperationsInternalCUDA.cu
 *  @brief Contains small internal functions used in linearAlgebraOperationsCUDA
 *
 */
namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    namespace internal
    {
      void
      createProcessGridSquareMatrix(
        const MPI_Comm &                           mpi_communicator,
        const unsigned                             size,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const bool                                 useOnlyThumbRule)
      {
        const unsigned int numberProcs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        unsigned int rowProcs =
          (dftParameters::scalapackParalProcs == 0 || useOnlyThumbRule) ?
            std::min(std::floor(std::sqrt(numberProcs)),
                     std::ceil((double)size / (double)(1000))) :
            std::min((unsigned int)std::floor(std::sqrt(numberProcs)),
                     dftParameters::scalapackParalProcs);

#ifdef DFTFE_WITH_ELPA
        rowProcs =
          ((dftParameters::scalapackParalProcs == 0 || useOnlyThumbRule) &&
           dftParameters::useELPA) ?
            std::min((unsigned int)std::floor(std::sqrt(numberProcs)),
                     (unsigned int)std::floor(rowProcs * 3.0)) :
            rowProcs;
#endif

        if (!dftParameters::reproducible_output)
          rowProcs =
            std::min(rowProcs,
                     (unsigned int)std::ceil((double)size / (double)(100)));

        if (dftParameters::verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
            pcout << "Scalapack Matrix created, row procs: " << rowProcs
                  << std::endl;
          }

        processGrid =
          std::make_shared<const dftfe::ProcessGrid>(mpi_communicator,
                                                     rowProcs,
                                                     rowProcs);
      }


      void
      createProcessGridRectangularMatrix(
        const MPI_Comm &                           mpi_communicator,
        const unsigned                             sizeRows,
        const unsigned                             sizeColumns,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
      {
        const unsigned int numberProcs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        const unsigned int rowProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeRows / (double)(1000)));
        const unsigned int columnProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeColumns / (double)(1000)));

        if (dftParameters::verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
            pcout << "Scalapack Matrix created, row procs x column procs: "
                  << rowProcs << " x " << columnProcs << std::endl;
          }

        processGrid =
          std::make_shared<const dftfe::ProcessGrid>(mpi_communicator,
                                                     rowProcs,
                                                     columnProcs);
      }


      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<T> &                mat,
        std::map<unsigned int, unsigned int> &           globalToLocalRowIdMap,
        std::map<unsigned int, unsigned int> &globalToLocalColumnIdMap)
      {
#ifdef USE_COMPLEX
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
        globalToLocalRowIdMap.clear();
        globalToLocalColumnIdMap.clear();
        if (processGrid->is_process_active())
          {
            for (unsigned int i = 0; i < mat.local_m(); ++i)
              globalToLocalRowIdMap[mat.global_row(i)] = i;

            for (unsigned int j = 0; j < mat.local_n(); ++j)
              globalToLocalColumnIdMap[mat.global_column(j)] = j;
          }
#endif
      }


      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T> &                      mat,
        const MPI_Comm &                                 interComm)
      {
#ifdef USE_COMPLEX
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Allreduce(MPI_IN_PLACE,
                          &mat.local_el(0, 0),
                          mat.local_m() * mat.local_n(),
                          MPI_DOUBLE,
                          MPI_SUM,
                          interComm);
          }
#endif
      }



      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T> &                      mat,
        const MPI_Comm &                                 interComm,
        const unsigned int                               broadcastRoot)
      {
#ifdef USE_COMPLEX
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Bcast(&mat.local_el(0, 0),
                      mat.local_m() * mat.local_n(),
                      MPI_DOUBLE,
                      broadcastRoot,
                      interComm);
          }
#endif
      }



      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<dataTypes::number> &mat,
        std::map<unsigned int, unsigned int> &           globalToLocalRowIdMap,
        std::map<unsigned int, unsigned int> &globalToLocalColumnIdMap);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<dataTypes::number> &      mat,
        const MPI_Comm &                                 interComm);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<dataTypes::number> &      mat,
        const MPI_Comm &                                 interComm,
        const unsigned int                               broadcastRoot);
    } // namespace internal
  }   // namespace linearAlgebraOperationsCUDA
} // namespace dftfe
