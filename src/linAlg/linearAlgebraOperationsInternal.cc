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
// @author Sambit Das
//


#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>

/** @file linearAlgebraOperationsInternal.cc
 *  @brief Contains small internal functions used in linearAlgebraOperations
 *
 */
namespace dftfe
{
  namespace linearAlgebraOperations
  {
    namespace internal
    {
      void
      setupELPAHandleParameters(
        const MPI_Comm &mpi_communicator,
        MPI_Comm &      processGridCommunicatorActive,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const unsigned int                               na,
        const unsigned int                               nev,
        const unsigned int                               blockSize,
        elpa_t &                                         elpaHandle,
        const dftParameters &                            dftParams)
      {
        int error;

        if (processGrid->is_process_active())
          {
            int error;
            elpaHandle = elpa_allocate(&error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));
          }

        // Get the group of processes in mpi_communicator
        int       ierr = 0;
        MPI_Group all_group;
        ierr = MPI_Comm_group(mpi_communicator, &all_group);
        AssertThrowMPI(ierr);

        // Construct the group containing all ranks we need:
        const unsigned int n_active_mpi_processes =
          processGrid->get_process_grid_rows() *
          processGrid->get_process_grid_columns();
        std::vector<int> active_ranks;
        for (unsigned int i = 0; i < n_active_mpi_processes; ++i)
          active_ranks.push_back(i);

        MPI_Group active_group;
        const int n = active_ranks.size();
        ierr = MPI_Group_incl(all_group, n, active_ranks.data(), &active_group);
        AssertThrowMPI(ierr);

        // Create the communicator based on active_group.
        // Note that on all the inactive processs the resulting MPI_Comm
        // processGridCommunicatorActive will be MPI_COMM_NULL.
        // MPI_Comm processGridCommunicatorActive;
        ierr = dealii::Utilities::MPI::create_group(
          mpi_communicator, active_group, 50, &processGridCommunicatorActive);
        AssertThrowMPI(ierr);

        ierr = MPI_Group_free(&all_group);
        AssertThrowMPI(ierr);
        ierr = MPI_Group_free(&active_group);
        AssertThrowMPI(ierr);


        dftfe::ScaLAPACKMatrix<double> tempMat(na, processGrid, blockSize);
        if (processGrid->is_process_active())
          {
            /* Set parameters the matrix and it's MPI distribution */
            elpa_set_integer(elpaHandle, "na", na, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));


            elpa_set_integer(elpaHandle, "nev", nev, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle, "nblk", blockSize, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "mpi_comm_parent",
                             MPI_Comm_c2f(processGridCommunicatorActive),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));


            // std::cout<<"local_nrows: "<<tempMat.local_m() <<std::endl;
            // std::cout<<"local_ncols: "<<tempMat.local_n() <<std::endl;
            // std::cout<<"process_row:
            // "<<processGrid->get_this_process_row()<<std::endl;
            // std::cout<<"process_col:
            // "<<processGrid->get_this_process_column()<<std::endl;

            elpa_set_integer(elpaHandle,
                             "local_nrows",
                             tempMat.local_m(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "local_ncols",
                             tempMat.local_n(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_row",
                             processGrid->get_this_process_row(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_col",
                             processGrid->get_this_process_column(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));


            /* Setup */
            AssertThrow(elpa_setup(elpaHandle) == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle, "solver", ELPA_SOLVER_2STAGE, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

#ifdef DFTFE_WITH_DEVICE

            if (dftParams.useELPADeviceKernel)
              {
#  ifdef DFTFE_WITH_DEVICE_NVIDIA
                elpa_set_integer(elpaHandle, "nvidia-gpu", 1, &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                elpa_set_integer(elpaHandle,
                                 "real_kernel",
                                 ELPA_2STAGE_REAL_NVIDIA_GPU,
                                 &error);

                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                elpa_set_integer(elpaHandle,
                                 "complex_kernel",
                                 ELPA_2STAGE_COMPLEX_NVIDIA_GPU,
                                 &error);

                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#  elif DFTFE_WITH_DEVICE_AMD
                elpa_set_integer(elpaHandle, "amd-gpu", 1, &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                elpa_set_integer(elpaHandle,
                                 "real_kernel",
                                 ELPA_2STAGE_REAL_AMD_GPU,
                                 &error);

                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                elpa_set_integer(elpaHandle,
                                 "complex_kernel",
                                 ELPA_2STAGE_COMPLEX_AMD_GPU,
                                 &error);

                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#  endif
              }
#endif

              // elpa_set_integer(elpaHandle,
              // "real_kernel",ELPA_2STAGE_REAL_AVX512_BLOCK6, &error);
              // AssertThrow(error==ELPA_OK,
              //   dealii::ExcMessage("DFT-FE Error: ELPA Error."));

#ifdef DEBUG
            elpa_set_integer(elpaHandle, "debug", 1, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
          }

        // d_elpaAutoTuneHandle = elpa_autotune_setup(d_elpaHandle,
        // ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);   // create
        // autotune object
      }

      void
      createProcessGridSquareMatrix(
        const MPI_Comm &                           mpi_communicator,
        const unsigned                             size,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters &                      dftParams,
        const bool                                 useOnlyThumbRule)
      {
        const unsigned int numberProcs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        unsigned int rowProcs =
          (dftParams.scalapackParalProcs == 0 || useOnlyThumbRule) ?
            std::min(std::floor(std::sqrt(numberProcs)),
                     std::ceil((double)size / (double)(1000))) :
            std::min((unsigned int)std::floor(std::sqrt(numberProcs)),
                     dftParams.scalapackParalProcs);

        rowProcs = ((dftParams.scalapackParalProcs == 0 || useOnlyThumbRule) &&
                    dftParams.useELPA) ?
                     std::min((unsigned int)std::floor(std::sqrt(numberProcs)),
                              (unsigned int)std::floor(rowProcs * 3.0)) :
                     rowProcs;
        if (!dftParams.reproducible_output)
          rowProcs =
            std::min(rowProcs,
                     (unsigned int)std::ceil((double)size / (double)(100)));

        else
          rowProcs =
            std::min(rowProcs,
                     (unsigned int)std::ceil((double)size / (double)(10)));


        if (dftParams.verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
               0));
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
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters &                      dftParams)
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

        if (dftParams.verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
               0));
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
        std::unordered_map<unsigned int, unsigned int> &           globalToLocalRowIdMap,
        std::unordered_map<unsigned int, unsigned int> &globalToLocalColumnIdMap)
      {
        globalToLocalRowIdMap.clear();
        globalToLocalColumnIdMap.clear();
        if (processGrid->is_process_active())
          {
            for (unsigned int i = 0; i < mat.local_m(); ++i)
              globalToLocalRowIdMap[mat.global_row(i)] = i;

            for (unsigned int j = 0; j < mat.local_n(); ++j)
              globalToLocalColumnIdMap[mat.global_column(j)] = j;
          }
      }


      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T> &                      mat,
        const MPI_Comm &                                 interComm)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Allreduce(MPI_IN_PLACE,
                          &mat.local_el(0, 0),
                          mat.local_m() * mat.local_n(),
                          dataTypes::mpi_type_id(&mat.local_el(0, 0)),
                          MPI_SUM,
                          interComm);
          }
      }

      template <typename T>
      void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T> &                      mat,
        const T                                          scalar)
      {
        if (processGrid->is_process_active())
          {
            const unsigned int numberComponents = mat.local_m() * mat.local_n();
            const unsigned int inc              = 1;
            xscal(&numberComponents, &scalar, &mat.local_el(0, 0), &inc);
          }
      }



      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T> &                      mat,
        const MPI_Comm &                                 interComm,
        const unsigned int                               broadcastRoot)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Bcast(&mat.local_el(0, 0),
                      mat.local_m() * mat.local_n(),
                      dataTypes::mpi_type_id(&mat.local_el(0, 0)),
                      broadcastRoot,
                      interComm);
          }
      }

      template <typename T, typename TLowPrec>
      void
      fillParallelOverlapMatrixMixedPrec(
        const T *          subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<T> &                      overlapMatPar,
        const dftParameters &                            dftParams)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          overlapMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
         * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
         * has a much smaller memory compared to X^{T}*Xc.
         * X^{T} is a matrix with size number of wavefunctions times
         * number of local degrees of freedom (N x MLoc).
         * MLoc is denoted by numLocalDofs.
         * Xc denotes complex conjugate of X.
         * XcBlock is a matrix with size (MLoc x B). B is the block size.
         * A further optimization is done to reduce floating point operations:
         * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the
         * lower triangular part. To exploit this, we do X^{T}*Xc=Sum_{blocks}
         * XTrunc^{T}*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
         * X^{T} with the row indices ranging fromt the lowest global index of
         * XcBlock (denoted by ivec in the code) to N. D=N-ivec. The parallel
         * ScaLapack overlap matrix is directly filled from the
         * XTrunc^{T}*XcBlock result
         */
        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T>        overlapMatrixBlock(N * vectorsBlockSize, T(0.0));
        std::vector<TLowPrec> overlapMatrixBlockLowPrec(N * vectorsBlockSize,
                                                        TLowPrec(0.0));
        std::vector<T>        overlapMatrixBlockDoublePrec(vectorsBlockSize *
                                                      vectorsBlockSize,
                                                    T(0.0));

        std::vector<TLowPrec> subspaceVectorsArrayLowPrec(
          subspaceVectorsArray,
          subspaceVectorsArray + subspaceVectorsArrayLocalSize);
        for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const unsigned int B = std::min(vectorsBlockSize, N - ivec);

            // If one plus the ending index of a block lies within a band
            // parallelization group do computations for that block within the
            // band group, otherwise skip that block. This is only activated if
            // NPBAND>1
            if ((ivec + B) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (ivec + B) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                const char transA = 'N',
                           transB =
                             std::is_same<T, std::complex<double>>::value ?
                               'C' :
                               'T';
                const T        scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
                const TLowPrec scalarCoeffAlphaLowPrec = 1.0,
                               scalarCoeffBetaLowPrec  = 0.0;

                std::fill(overlapMatrixBlock.begin(),
                          overlapMatrixBlock.end(),
                          0.);
                std::fill(overlapMatrixBlockLowPrec.begin(),
                          overlapMatrixBlockLowPrec.end(),
                          0.);

                const unsigned int D = N - ivec;

                xgemm(&transA,
                      &transB,
                      &B,
                      &B,
                      &numLocalDofs,
                      &scalarCoeffAlpha,
                      subspaceVectorsArray + ivec,
                      &N,
                      subspaceVectorsArray + ivec,
                      &N,
                      &scalarCoeffBeta,
                      &overlapMatrixBlockDoublePrec[0],
                      &B);

                const unsigned int DRem = D - B;
                if (DRem != 0)
                  {
                    xgemm(&transA,
                          &transB,
                          &DRem,
                          &B,
                          &numLocalDofs,
                          &scalarCoeffAlphaLowPrec,
                          &subspaceVectorsArrayLowPrec[0] + ivec + B,
                          &N,
                          &subspaceVectorsArrayLowPrec[0] + ivec,
                          &N,
                          &scalarCoeffBetaLowPrec,
                          &overlapMatrixBlockLowPrec[0],
                          &DRem);
                  }

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock for double precision across
                // domain decomposition processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlockDoublePrec[0],
                              B * B,
                              dataTypes::mpi_type_id(
                                &overlapMatrixBlockDoublePrec[0]),
                              MPI_SUM,
                              mpiComm);

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock for single precision across
                // domain decomposition processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlockLowPrec[0],
                              DRem * B,
                              dataTypes::mpi_type_id(
                                &overlapMatrixBlockLowPrec[0]),
                              MPI_SUM,
                              mpiComm);

                for (unsigned int i = 0; i < B; ++i)
                  {
                    for (unsigned int j = 0; j < B; ++j)
                      overlapMatrixBlock[i * D + j] =
                        overlapMatrixBlockDoublePrec[i * B + j];

                    for (unsigned int j = 0; j < DRem; ++j)
                      overlapMatrixBlock[i * D + j + B] =
                        overlapMatrixBlockLowPrec[i * DRem + j];
                  }

                // Copying only the lower triangular part to the ScaLAPACK
                // overlap matrix
                if (processGrid->is_process_active())
                  for (unsigned int i = 0; i < B; ++i)
                    if (globalToLocalColumnIdMap.find(i + ivec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[i + ivec];
                        for (unsigned int j = ivec + i; j < N; ++j)
                          {
                            std::unordered_map<unsigned int, unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(j);
                            if (it != globalToLocalRowIdMap.end())
                              overlapMatPar.local_el(it->second,
                                                     localColumnId) =
                                overlapMatrixBlock[i * D + j - ivec];
                          }
                      }
              } // band parallelization
          }     // block loop


        // accumulate contribution from all band parallelization groups
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
      }


      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *          subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<T> &                      overlapMatPar,
        const dftParameters &                            dftParams)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          overlapMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
         * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
         * has a much smaller memory compared to X^{T}*Xc.
         * X^{T} is a matrix with size number of wavefunctions times
         * number of local degrees of freedom (N x MLoc).
         * MLoc is denoted by numLocalDofs.
         * Xc denotes complex conjugate of X.
         * XcBlock is a matrix with size (MLoc x B). B is the block size.
         * A further optimization is done to reduce floating point operations:
         * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the
         * lower triangular part. To exploit this, we do X^{T}*Xc=Sum_{blocks}
         * XTrunc^{T}*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
         * X^{T} with the row indices ranging fromt the lowest global index of
         * XcBlock (denoted by ivec in the code) to N. D=N-ivec. The parallel
         * ScaLapack overlap matrix is directly filled from the
         * XTrunc^{T}*XcBlock result
         */
        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T> overlapMatrixBlock(N * vectorsBlockSize, 0.0);

        for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const unsigned int B = std::min(vectorsBlockSize, N - ivec);

            // If one plus the ending index of a block lies within a band
            // parallelization group do computations for that block within the
            // band group, otherwise skip that block. This is only activated if
            // NPBAND>1
            if ((ivec + B) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (ivec + B) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                const char transA = 'N',
                           transB =
                             std::is_same<T, std::complex<double>>::value ?
                               'C' :
                               'T';
                const T scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

                std::fill(overlapMatrixBlock.begin(),
                          overlapMatrixBlock.end(),
                          0.);

                const unsigned int D = N - ivec;

                // Comptute local XTrunc^{T}*XcBlock.
                xgemm(&transA,
                      &transB,
                      &D,
                      &B,
                      &numLocalDofs,
                      &scalarCoeffAlpha,
                      subspaceVectorsArray + ivec,
                      &N,
                      subspaceVectorsArray + ivec,
                      &N,
                      &scalarCoeffBeta,
                      &overlapMatrixBlock[0],
                      &D);

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock across domain decomposition
                // processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlock[0],
                              D * B,
                              dataTypes::mpi_type_id(&overlapMatrixBlock[0]),
                              MPI_SUM,
                              mpiComm);

                // Copying only the lower triangular part to the ScaLAPACK
                // overlap matrix
                if (processGrid->is_process_active())
                  for (unsigned int i = 0; i < B; ++i)
                    if (globalToLocalColumnIdMap.find(i + ivec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[i + ivec];
                        for (unsigned int j = ivec + i; j < N; ++j)
                          {
                            std::unordered_map<unsigned int, unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(j);
                            if (it != globalToLocalRowIdMap.end())
                              overlapMatPar.local_el(it->second,
                                                     localColumnId) =
                                overlapMatrixBlock[i * D + j - ivec];
                          }
                      }
              } // band parallelization
          }     // block loop


        // accumulate contribution from all band parallelization groups
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
      }


      template <typename T>
      void
      subspaceRotation(
        T *                subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<T> &                rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranspose,
        const bool                                       isRotationMatLowerTria,
        const bool                                       doCommAfterBandParal)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        const unsigned int maxNumLocalDofs =
          dealii::Utilities::MPI::max(numLocalDofs, mpiComm);

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          rotationMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Q*X^{T} is done in a blocked approach for memory optimization:
         * Sum_{dof_blocks} Sum_{vector_blocks} QBvec*XBdof^{T}.
         * The result of each QBvec*XBdof^{T}
         * has a much smaller memory compared to Q*X^{T}.
         * X^{T} (denoted by subspaceVectorsArray in the code with column major
         * format storage) is a matrix with size (N x MLoc). N is denoted by
         * numberWaveFunctions in the code. MLoc, which is number of local dofs
         * is denoted by numLocalDofs in the code. QBvec is a matrix of size
         * (BVec x N). BVec is the vectors block size. XBdof is a matrix of size
         * (N x BDof). BDof is the block size of dofs. A further optimization is
         * done to reduce floating point operations when Q is a lower triangular
         * matrix in the subspace rotation step of CGS: Then it suffices to
         * compute only the multiplication of lower triangular part of Q with
         * X^{T}. To exploit this, we do Sum_{dof_blocks} Sum_{vector_blocks}
         * QBvecTrunc*XBdofTrunc^{T}. where QBvecTrunc is a (BVec x D) sub
         * matrix of QBvec with the column indices ranging from O to D-1, where
         * D=jvec(lowest global index of QBvec) + BVec. XBdofTrunc is a (D x
         * BDof) sub matrix of XBdof with the row indices ranging from 0 to D-1.
         * X^{T} is directly updated from
         * the Sum_{vector_blocks} QBvecTrunc*XBdofTrunc^{T} result
         * for each {dof_block}.
         */
        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);
        const unsigned int dofsBlockSize =
          std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

        std::vector<T> rotationMatBlock(vectorsBlockSize * N, 0.0);
        std::vector<T> rotatedVectorsMatBlock(N * dofsBlockSize, 0.0);

        if (dftParams.verbosity >= 4)
          dftUtils::printCurrentMemoryUsage(mpiComm,
                                            "Inside Blocked susbpace rotation");
        int startIndexBandParal = N;
        int numVectorsBandParal = 0;

        for (unsigned int idof = 0; idof < maxNumLocalDofs;
             idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            unsigned int BDof = 0;
            if (numLocalDofs >= idof)
              BDof = std::min(dofsBlockSize, numLocalDofs - idof);

            std::fill(rotatedVectorsMatBlock.begin(),
                      rotatedVectorsMatBlock.end(),
                      0.);
            for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

                const unsigned int D =
                  isRotationMatLowerTria ? (jvec + BVec) : N;

                // If one plus the ending index of a block lies within a band
                // parallelization group do computations for that block within
                // the band group, otherwise skip that block. This is only
                // activated if NPBAND>1
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (jvec < startIndexBandParal)
                      startIndexBandParal = jvec;
                    numVectorsBandParal = jvec + BVec - startIndexBandParal;

                    const char transA = 'N', transB = 'N';
                    const T scalarCoeffAlpha = T(1.0), scalarCoeffBeta = T(0.0);

                    std::fill(rotationMatBlock.begin(),
                              rotationMatBlock.end(),
                              0.);

                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (rotationMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const unsigned int localRowId =
                                  globalToLocalRowIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)

                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      rotationMatBlock[i * BVec + j] =
                                        rotationMatPar.local_el(localRowId,
                                                                it->second);
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const unsigned int localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      rotationMatBlock[i * BVec + j] =
                                        rotationMatPar.local_el(it->second,
                                                                localColumnId);
                                  }
                              }
                      }


                    MPI_Barrier(mpiComm);
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatBlock[0],
                                  BVec * D,
                                  dataTypes::mpi_type_id(&rotationMatBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    if (BDof != 0)
                      {
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &D,
                              &scalarCoeffAlpha,
                              &rotationMatBlock[0],
                              &BVec,
                              subspaceVectorsArray + idof * N,
                              &N,
                              &scalarCoeffBeta,
                              &rotatedVectorsMatBlock[0] + jvec,
                              &N);
                      }

                  } // band parallelization
              }     // block loop over vectors

            if (BDof != 0)
              {
                for (unsigned int i = 0; i < BDof; ++i)
                  for (unsigned int j = 0; j < N; ++j)
                    *(subspaceVectorsArray + N * (i + idof) + j) =
                      rotatedVectorsMatBlock[i * N + j];
              }
          } // block loop over dofs

        if (numberBandGroups > 1 && doCommAfterBandParal)
          {
            if (!dftParams.bandParalOpt)
              {
                MPI_Barrier(interBandGroupComm);
                const unsigned int blockSize =
                  dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 / sizeof(T);

                for (unsigned int i = 0; i < N * numLocalDofs; i += blockSize)
                  {
                    const unsigned int currentBlockSize =
                      std::min(blockSize, N * numLocalDofs - i);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  subspaceVectorsArray + i,
                                  currentBlockSize,
                                  dataTypes::mpi_type_id(subspaceVectorsArray),
                                  MPI_SUM,
                                  interBandGroupComm);
                  }
              }
            else
              {
                MPI_Barrier(interBandGroupComm);

                std::vector<T> eigenVectorsBandGroup(numVectorsBandParal *
                                                       numLocalDofs,
                                                     T(0));
                std::vector<T> eigenVectorsBandGroupTransposed(
                  numVectorsBandParal * numLocalDofs, T(0));
                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroup[iNode * numVectorsBandParal + iWave] =
                      subspaceVectorsArray[iNode * N + startIndexBandParal +
                                           iWave];

                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroupTransposed[iWave * numLocalDofs +
                                                    iNode] =
                      eigenVectorsBandGroup[iNode * numVectorsBandParal +
                                            iWave];

                std::vector<int> recvcounts(numberBandGroups, 0);
                std::vector<int> displs(numberBandGroups, 0);

                int recvcount = numVectorsBandParal * numLocalDofs;
                MPI_Allgather(&recvcount,
                              1,
                              MPI_INT,
                              &recvcounts[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                int displ = startIndexBandParal * numLocalDofs;
                MPI_Allgather(&displ,
                              1,
                              MPI_INT,
                              &displs[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                std::vector<T> eigenVectorsTransposed(N * numLocalDofs, 0);
                MPI_Allgatherv(
                  &eigenVectorsBandGroupTransposed[0],
                  numVectorsBandParal * numLocalDofs,
                  dataTypes::mpi_type_id(&eigenVectorsBandGroupTransposed[0]),
                  &eigenVectorsTransposed[0],
                  &recvcounts[0],
                  &displs[0],
                  dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
                  interBandGroupComm);

                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < N; ++iWave)
                    subspaceVectorsArray[iNode * N + iWave] =
                      eigenVectorsTransposed[iWave * numLocalDofs + iNode];
              }
          }
      }



      template <typename T>
      void
      subspaceRotationSpectrumSplit(
        const T *          X,
        T *                Y,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const unsigned int                               numberTopVectors,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<T> &                QMat,
        const dftParameters &                            dftParams,
        const bool                                       QMatTranspose)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        const unsigned int maxNumLocalDofs =
          dealii::Utilities::MPI::max(numLocalDofs, mpiComm);

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, numberTopVectors, bandGroupLowHighPlusOneIndices);

        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid, QMat, globalToLocalRowIdMap, globalToLocalColumnIdMap);


        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);
        const unsigned int dofsBlockSize =
          std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

        std::vector<T> rotationMatBlock(vectorsBlockSize * N, T(0.0));
        std::vector<T> rotatedVectorsMatBlock(numberTopVectors * dofsBlockSize,
                                              T(0.0));

        if (dftParams.verbosity >= 4)
          dftUtils::printCurrentMemoryUsage(mpiComm,
                                            "Inside Blocked susbpace rotation");

        for (unsigned int idof = 0; idof < maxNumLocalDofs;
             idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            unsigned int BDof = 0;
            if (numLocalDofs >= idof)
              BDof = std::min(dofsBlockSize, numLocalDofs - idof);

            std::fill(rotatedVectorsMatBlock.begin(),
                      rotatedVectorsMatBlock.end(),
                      0.);
            for (unsigned int jvec = 0; jvec < numberTopVectors;
                 jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec =
                  std::min(vectorsBlockSize, numberTopVectors - jvec);

                // If one plus the ending index of a block lies within a band
                // parallelization group do computations for that block within
                // the band group, otherwise skip that block. This is only
                // activated if NPBAND>1
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    const char transA = 'N', transB = 'N';
                    const T scalarCoeffAlpha = T(1.0), scalarCoeffBeta = T(0.0);

                    std::fill(rotationMatBlock.begin(),
                              rotationMatBlock.end(),
                              T(0.));

                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (QMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < N; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const unsigned int localRowId =
                                  globalToLocalRowIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)

                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      rotationMatBlock[i * BVec + j] =
                                        QMat.local_el(localRowId, it->second);
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < N; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const unsigned int localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      rotationMatBlock[i * BVec + j] =
                                        QMat.local_el(it->second,
                                                      localColumnId);
                                  }
                              }
                      }

                    MPI_Barrier(mpiComm);
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatBlock[0],
                                  BVec * N,
                                  dataTypes::mpi_type_id(&rotationMatBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    if (BDof != 0)
                      {
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &N,
                              &scalarCoeffAlpha,
                              &rotationMatBlock[0],
                              &BVec,
                              X + idof * N,
                              &N,
                              &scalarCoeffBeta,
                              &rotatedVectorsMatBlock[0] + jvec,
                              &numberTopVectors);
                      }

                  } // band parallelization
              }     // block loop over vectors


            if (BDof != 0)
              {
                for (unsigned int i = 0; i < BDof; ++i)
                  for (unsigned int j = 0; j < numberTopVectors; ++j)
                    *(Y + numberTopVectors * (i + idof) + j) =
                      rotatedVectorsMatBlock[i * numberTopVectors + j];
              }
          } // block loop over dofs

        if (numberBandGroups > 1)
          {
            const unsigned int blockSize =
              dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 / sizeof(T);
            MPI_Barrier(interBandGroupComm);
            for (unsigned int i = 0; i < numberTopVectors * numLocalDofs;
                 i += blockSize)
              {
                const unsigned int currentBlockSize =
                  std::min(blockSize, numberTopVectors * numLocalDofs - i);

                MPI_Allreduce(MPI_IN_PLACE,
                              Y + i,
                              currentBlockSize,
                              dataTypes::mpi_type_id(Y),
                              MPI_SUM,
                              interBandGroupComm);
              }
          }
      }

      template <typename T, typename TLowPrec>
      void
      subspaceRotationSpectrumSplitMixedPrec(
        const T *          X,
        T *                Y,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const unsigned int                               numberTopVectors,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<T> &                QMat,
        const dftParameters &                            dftParams,
        const bool                                       QMatTranspose)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        const unsigned int maxNumLocalDofs =
          dealii::Utilities::MPI::max(numLocalDofs, mpiComm);

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, numberTopVectors, bandGroupLowHighPlusOneIndices);

        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid, QMat, globalToLocalRowIdMap, globalToLocalColumnIdMap);


        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);
        const unsigned int dofsBlockSize =
          std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

        const unsigned int Ncore = N - numberTopVectors;
        std::vector<T>     rotationMatTopCompBlock(vectorsBlockSize *
                                                 numberTopVectors,
                                               T(0.0));
        std::vector<T> rotatedVectorsMatBlock(numberTopVectors * dofsBlockSize,
                                              T(0.0));

        std::vector<TLowPrec> rotationMatCoreCompBlock(vectorsBlockSize * Ncore,
                                                       TLowPrec(0.0));
        std::vector<TLowPrec> rotatedVectorsMatCoreContrBlockTemp(
          vectorsBlockSize * dofsBlockSize, TLowPrec(0.0));

        std::vector<TLowPrec> XSinglePrec(X, X + subspaceVectorsArrayLocalSize);
        if (dftParams.verbosity >= 4)
          dftUtils::printCurrentMemoryUsage(mpiComm,
                                            "Inside Blocked susbpace rotation");

        for (unsigned int idof = 0; idof < maxNumLocalDofs;
             idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            unsigned int BDof = 0;
            if (numLocalDofs >= idof)
              BDof = std::min(dofsBlockSize, numLocalDofs - idof);

            std::fill(rotatedVectorsMatBlock.begin(),
                      rotatedVectorsMatBlock.end(),
                      T(0.));
            for (unsigned int jvec = 0; jvec < numberTopVectors;
                 jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec =
                  std::min(vectorsBlockSize, numberTopVectors - jvec);

                // If one plus the ending index of a block lies within a band
                // parallelization group do computations for that block within
                // the band group, otherwise skip that block. This is only
                // activated if NPBAND>1
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    const char transA = 'N', transB = 'N';
                    const T scalarCoeffAlpha = T(1.0), scalarCoeffBeta = T(0.0);
                    const TLowPrec scalarCoeffAlphaSinglePrec = TLowPrec(1.0),
                                   scalarCoeffBetaSinglePrec  = TLowPrec(0.0);

                    std::fill(rotationMatCoreCompBlock.begin(),
                              rotationMatCoreCompBlock.end(),
                              TLowPrec(0.));
                    std::fill(rotationMatTopCompBlock.begin(),
                              rotationMatTopCompBlock.end(),
                              T(0.));

                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (QMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < N; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const unsigned int localRowId =
                                  globalToLocalRowIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)

                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      {
                                        const T val =
                                          QMat.local_el(localRowId, it->second);
                                        if (i < Ncore)
                                          rotationMatCoreCompBlock[i * BVec +
                                                                   j] = val;
                                        else
                                          rotationMatTopCompBlock[(i - Ncore) *
                                                                    BVec +
                                                                  j] = val;
                                      }
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < N; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const unsigned int localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      {
                                        const T val =
                                          QMat.local_el(it->second,
                                                        localColumnId);
                                        if (i < Ncore)
                                          rotationMatCoreCompBlock[i * BVec +
                                                                   j] = val;
                                        else
                                          rotationMatTopCompBlock[(i - Ncore) *
                                                                    BVec +
                                                                  j] = val;
                                      }
                                  }
                              }
                      }

                    MPI_Barrier(mpiComm);
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatCoreCompBlock[0],
                                  BVec * Ncore,
                                  dataTypes::mpi_type_id(
                                    &rotationMatCoreCompBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatTopCompBlock[0],
                                  BVec * numberTopVectors,
                                  dataTypes::mpi_type_id(
                                    &rotationMatTopCompBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    if (BDof != 0)
                      {
                        // single precision
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &Ncore,
                              &scalarCoeffAlphaSinglePrec,
                              &rotationMatCoreCompBlock[0],
                              &BVec,
                              &XSinglePrec[0] + idof * N,
                              &N,
                              &scalarCoeffBetaSinglePrec,
                              &rotatedVectorsMatCoreContrBlockTemp[0],
                              &BVec);

                        // double precision
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &numberTopVectors,
                              &scalarCoeffAlpha,
                              &rotationMatTopCompBlock[0],
                              &BVec,
                              X + idof * N + Ncore,
                              &N,
                              &scalarCoeffBeta,
                              &rotatedVectorsMatBlock[0] + jvec,
                              &numberTopVectors);

                        for (unsigned int i = 0; i < BDof; ++i)
                          for (unsigned int j = 0; j < BVec; ++j)
                            rotatedVectorsMatBlock[i * numberTopVectors + j +
                                                   jvec] +=
                              rotatedVectorsMatCoreContrBlockTemp[i * BVec + j];
                      }

                  } // band parallelization
              }     // block loop over vectors


            if (BDof != 0)
              {
                for (unsigned int i = 0; i < BDof; ++i)
                  for (unsigned int j = 0; j < numberTopVectors; ++j)
                    *(Y + numberTopVectors * (i + idof) + j) =
                      rotatedVectorsMatBlock[i * numberTopVectors + j];
              }
          } // block loop over dofs

        if (numberBandGroups > 1)
          {
            const unsigned int blockSize =
              dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 / sizeof(T);
            MPI_Barrier(interBandGroupComm);
            for (unsigned int i = 0; i < numberTopVectors * numLocalDofs;
                 i += blockSize)
              {
                const unsigned int currentBlockSize =
                  std::min(blockSize, numberTopVectors * numLocalDofs - i);

                MPI_Allreduce(MPI_IN_PLACE,
                              Y + i,
                              currentBlockSize,
                              dataTypes::mpi_type_id(Y),
                              MPI_SUM,
                              interBandGroupComm);
              }
          }
      }

      template <typename T, typename TLowPrec>
      void
      subspaceRotationMixedPrec(
        T *                subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<T> &                rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranspose,
        const bool                                       doCommAfterBandParal)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        const unsigned int maxNumLocalDofs =
          dealii::Utilities::MPI::max(numLocalDofs, mpiComm);

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          rotationMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Q*X^{T} is done in a blocked approach for memory optimization:
         * Sum_{dof_blocks} Sum_{vector_blocks} QBvec*XBdof^{T}.
         * The result of each QBvec*XBdof^{T}
         * has a much smaller memory compared to Q*X^{T}.
         * X^{T} (denoted by subspaceVectorsArray in the code with column major
         * format storage) is a matrix with size (N x MLoc). N is denoted by
         * numberWaveFunctions in the code. MLoc, which is number of local dofs
         * is denoted by numLocalDofs in the code. QBvec is a matrix of size
         * (BVec x N). BVec is the vectors block size. XBdof is a matrix of size
         * (N x BDof). BDof is the block size of dofs. A further optimization is
         * done to reduce floating point operations when Q is a lower triangular
         * matrix in the subspace rotation step of CGS: Then it suffices to
         * compute only the multiplication of lower triangular part of Q with
         * X^{T}. To exploit this, we do Sum_{dof_blocks} Sum_{vector_blocks}
         * QBvecTrunc*XBdofTrunc^{T}. where QBvecTrunc is a (BVec x D) sub
         * matrix of QBvec with the column indices ranging from O to D-1, where
         * D=jvec(lowest global index of QBvec) + BVec. XBdofTrunc is a (D x
         * BDof) sub matrix of XBdof with the row indices ranging from 0 to D-1.
         * X^{T} is directly updated from
         * the Sum_{vector_blocks} QBvecTrunc*XBdofTrunc^{T} result
         * for each {dof_block}.
         */
        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);
        const unsigned int dofsBlockSize =
          std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

        std::vector<TLowPrec> rotationMatBlock(vectorsBlockSize * N,
                                               TLowPrec(0.0));
        std::vector<TLowPrec> rotatedVectorsMatBlockTemp(vectorsBlockSize *
                                                           dofsBlockSize,
                                                         TLowPrec(0.0));

        std::vector<TLowPrec> subspaceVectorsArraySinglePrec(
          subspaceVectorsArray,
          subspaceVectorsArray + subspaceVectorsArrayLocalSize);
        std::vector<T> diagValuesBlock(vectorsBlockSize, T(0.0));
        if (dftParams.verbosity >= 4)
          dftUtils::printCurrentMemoryUsage(mpiComm,
                                            "Inside Blocked susbpace rotation");

        int startIndexBandParal = N;
        int numVectorsBandParal = 0;
        for (unsigned int idof = 0; idof < maxNumLocalDofs;
             idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            unsigned int BDof = 0;
            if (numLocalDofs >= idof)
              BDof = std::min(dofsBlockSize, numLocalDofs - idof);

            for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

                const unsigned int D = N;

                // If one plus the ending index of a block lies within a band
                // parallelization group do computations for that block within
                // the band group, otherwise skip that block. This is only
                // activated if NPBAND>1
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (jvec < startIndexBandParal)
                      startIndexBandParal = jvec;
                    numVectorsBandParal = jvec + BVec - startIndexBandParal;

                    const char     transA = 'N', transB = 'N';
                    const TLowPrec scalarCoeffAlpha = TLowPrec(1.0),
                                   scalarCoeffBeta  = TLowPrec(0.0);

                    std::fill(rotationMatBlock.begin(),
                              rotationMatBlock.end(),
                              TLowPrec(0.));
                    std::fill(diagValuesBlock.begin(),
                              diagValuesBlock.end(),
                              T(0.));
                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (rotationMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const unsigned int localRowId =
                                  globalToLocalRowIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + j] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                      }
                                  }

                                if (i >= jvec && i < (jvec + BVec))
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(i);
                                    if (it != globalToLocalColumnIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + i - jvec] =
                                          TLowPrec(0.0);
                                        diagValuesBlock[i - jvec] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                      }
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const unsigned int localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + j] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                      }
                                  }

                                if (i >= jvec && i < (jvec + BVec))
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(i);
                                    if (globalToLocalRowIdMap.find(i) !=
                                        globalToLocalRowIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + i - jvec] =
                                          TLowPrec(0.0);
                                        diagValuesBlock[i - jvec] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                      }
                                  }
                              }
                      }

                    MPI_Barrier(mpiComm);
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatBlock[0],
                                  BVec * D,
                                  dataTypes::mpi_type_id(&rotationMatBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  &diagValuesBlock[0],
                                  BVec,
                                  dataTypes::mpi_type_id(&diagValuesBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    if (BDof != 0)
                      {
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &D,
                              &scalarCoeffAlpha,
                              &rotationMatBlock[0],
                              &BVec,
                              &subspaceVectorsArraySinglePrec[0] + idof * N,
                              &N,
                              &scalarCoeffBeta,
                              &rotatedVectorsMatBlockTemp[0],
                              &BVec);

                        for (unsigned int i = 0; i < BDof; ++i)
                          for (unsigned int j = 0; j < BVec; ++j)
                            *(subspaceVectorsArray + N * (idof + i) + j +
                              jvec) =
                              *(subspaceVectorsArray + N * (idof + i) + j +
                                jvec) *
                                diagValuesBlock[j] +
                              T(rotatedVectorsMatBlockTemp[i * BVec + j]);
                      }

                  } // band parallelization
                else
                  {
                    for (unsigned int i = 0; i < BDof; ++i)
                      for (unsigned int j = 0; j < BVec; ++j)
                        *(subspaceVectorsArray + N * (idof + i) + j + jvec) =
                          T(0.0);
                  }
              } // block loop over vectors
          }     // block loop over dofs

        if (numberBandGroups > 1 && doCommAfterBandParal)
          {
            if (!dftParams.bandParalOpt)
              {
                MPI_Barrier(interBandGroupComm);
                const unsigned int blockSize =
                  dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 / sizeof(T);

                for (unsigned int i = 0; i < N * numLocalDofs; i += blockSize)
                  {
                    const unsigned int currentBlockSize =
                      std::min(blockSize, N * numLocalDofs - i);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  subspaceVectorsArray + i,
                                  currentBlockSize,
                                  dataTypes::mpi_type_id(subspaceVectorsArray),
                                  MPI_SUM,
                                  interBandGroupComm);
                  }
              }
            else
              {
                MPI_Barrier(interBandGroupComm);

                std::vector<T> eigenVectorsBandGroup(numVectorsBandParal *
                                                       numLocalDofs,
                                                     T(0));
                std::vector<T> eigenVectorsBandGroupTransposed(
                  numVectorsBandParal * numLocalDofs, T(0));
                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroup[iNode * numVectorsBandParal + iWave] =
                      subspaceVectorsArray[iNode * N + startIndexBandParal +
                                           iWave];


                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroupTransposed[iWave * numLocalDofs +
                                                    iNode] =
                      eigenVectorsBandGroup[iNode * numVectorsBandParal +
                                            iWave];

                std::vector<int> recvcounts(numberBandGroups, 0);
                std::vector<int> displs(numberBandGroups, 0);

                int recvcount = numVectorsBandParal * numLocalDofs;
                MPI_Allgather(&recvcount,
                              1,
                              MPI_INT,
                              &recvcounts[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                int displ = startIndexBandParal * numLocalDofs;
                MPI_Allgather(&displ,
                              1,
                              MPI_INT,
                              &displs[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                std::vector<T> eigenVectorsTransposed(N * numLocalDofs, 0);
                MPI_Allgatherv(
                  &eigenVectorsBandGroupTransposed[0],
                  numVectorsBandParal * numLocalDofs,
                  dataTypes::mpi_type_id(&eigenVectorsBandGroupTransposed[0]),
                  &eigenVectorsTransposed[0],
                  &recvcounts[0],
                  &displs[0],
                  dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
                  interBandGroupComm);

                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < N; ++iWave)
                    subspaceVectorsArray[iNode * N + iWave] =
                      eigenVectorsTransposed[iWave * numLocalDofs + iNode];
              }
          }
      }

      template <typename T, typename TLowPrec>
      void
      subspaceRotationCGSMixedPrec(
        T *                subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<T> &                rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranspose,
        const bool                                       doCommAfterBandParal)
      {
        const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize / N;

        const unsigned int maxNumLocalDofs =
          dealii::Utilities::MPI::max(numLocalDofs, mpiComm);

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
        std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          rotationMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Q*X^{T} is done in a blocked approach for memory optimization:
         * Sum_{dof_blocks} Sum_{vector_blocks} QBvec*XBdof^{T}.
         * The result of each QBvec*XBdof^{T}
         * has a much smaller memory compared to Q*X^{T}.
         * X^{T} (denoted by subspaceVectorsArray in the code with column major
         * format storage) is a matrix with size (N x MLoc). N is denoted by
         * numberWaveFunctions in the code. MLoc, which is number of local dofs
         * is denoted by numLocalDofs in the code. QBvec is a matrix of size
         * (BVec x N). BVec is the vectors block size. XBdof is a matrix of size
         * (N x BDof). BDof is the block size of dofs. A further optimization is
         * done to reduce floating point operations when Q is a lower triangular
         * matrix in the subspace rotation step of CGS: Then it suffices to
         * compute only the multiplication of lower triangular part of Q with
         * X^{T}. To exploit this, we do Sum_{dof_blocks} Sum_{vector_blocks}
         * QBvecTrunc*XBdofTrunc^{T}. where QBvecTrunc is a (BVec x D) sub
         * matrix of QBvec with the column indices ranging from O to D-1, where
         * D=jvec(lowest global index of QBvec) + BVec. XBdofTrunc is a (D x
         * BDof) sub matrix of XBdof with the row indices ranging from 0 to D-1.
         * X^{T} is directly updated from
         * the Sum_{vector_blocks} QBvecTrunc*XBdofTrunc^{T} result
         * for each {dof_block}.
         */
        const unsigned int vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);
        const unsigned int dofsBlockSize =
          std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

        std::vector<TLowPrec> rotationMatBlock(vectorsBlockSize * N,
                                               TLowPrec(0.0));
        std::vector<TLowPrec> rotatedVectorsMatBlockTemp(vectorsBlockSize *
                                                           dofsBlockSize,
                                                         TLowPrec(0.0));

        std::vector<TLowPrec> subspaceVectorsArraySinglePrec(
          subspaceVectorsArray,
          subspaceVectorsArray + subspaceVectorsArrayLocalSize);
        std::vector<T> diagValuesBlock(vectorsBlockSize, T(0.0));
        if (dftParams.verbosity >= 4)
          dftUtils::printCurrentMemoryUsage(mpiComm,
                                            "Inside Blocked susbpace rotation");

        int startIndexBandParal = N;
        int numVectorsBandParal = 0;
        for (unsigned int idof = 0; idof < maxNumLocalDofs;
             idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            unsigned int BDof = 0;
            if (numLocalDofs >= idof)
              BDof = std::min(dofsBlockSize, numLocalDofs - idof);

            for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

                const unsigned int D = jvec + BVec;

                // If one plus the ending index of a block lies within a band
                // parallelization group do computations for that block within
                // the band group, otherwise skip that block. This is only
                // activated if NPBAND>1
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (jvec < startIndexBandParal)
                      startIndexBandParal = jvec;
                    numVectorsBandParal = jvec + BVec - startIndexBandParal;

                    const char     transA = 'N', transB = 'N';
                    const TLowPrec scalarCoeffAlpha = TLowPrec(1.0),
                                   scalarCoeffBeta  = TLowPrec(0.0);

                    std::fill(rotationMatBlock.begin(),
                              rotationMatBlock.end(),
                              TLowPrec(0.));
                    std::fill(diagValuesBlock.begin(),
                              diagValuesBlock.end(),
                              T(0.));
                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (rotationMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const unsigned int localRowId =
                                  globalToLocalRowIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + j] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                      }
                                  }

                                if (i >= jvec && i < (jvec + BVec))
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalColumnIdMap.find(i);
                                    if (it != globalToLocalColumnIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + i - jvec] =
                                          TLowPrec(0.0);
                                        diagValuesBlock[i - jvec] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                      }
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (unsigned int i = 0; i < D; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const unsigned int localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (unsigned int j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + j] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                      }
                                  }

                                if (i >= jvec && i < (jvec + BVec))
                                  {
                                    std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                                      globalToLocalRowIdMap.find(i);
                                    if (globalToLocalRowIdMap.find(i) !=
                                        globalToLocalRowIdMap.end())
                                      {
                                        rotationMatBlock[i * BVec + i - jvec] =
                                          TLowPrec(0.0);
                                        diagValuesBlock[i - jvec] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                      }
                                  }
                              }
                      }

                    MPI_Barrier(mpiComm);
                    MPI_Allreduce(MPI_IN_PLACE,
                                  &rotationMatBlock[0],
                                  BVec * D,
                                  dataTypes::mpi_type_id(&rotationMatBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  &diagValuesBlock[0],
                                  BVec,
                                  dataTypes::mpi_type_id(&diagValuesBlock[0]),
                                  MPI_SUM,
                                  mpiComm);

                    if (BDof != 0)
                      {
                        xgemm(&transA,
                              &transB,
                              &BVec,
                              &BDof,
                              &D,
                              &scalarCoeffAlpha,
                              &rotationMatBlock[0],
                              &BVec,
                              &subspaceVectorsArraySinglePrec[0] + idof * N,
                              &N,
                              &scalarCoeffBeta,
                              &rotatedVectorsMatBlockTemp[0],
                              &BVec);

                        for (unsigned int i = 0; i < BDof; ++i)
                          for (unsigned int j = 0; j < BVec; ++j)
                            *(subspaceVectorsArray + N * (idof + i) + j +
                              jvec) =
                              *(subspaceVectorsArray + N * (idof + i) + j +
                                jvec) *
                                diagValuesBlock[j] +
                              T(rotatedVectorsMatBlockTemp[i * BVec + j]);
                      }

                  } // band parallelization
                else
                  {
                    for (unsigned int i = 0; i < BDof; ++i)
                      for (unsigned int j = 0; j < BVec; ++j)
                        *(subspaceVectorsArray + N * (idof + i) + j + jvec) =
                          T(0.0);
                  }
              } // block loop over vectors
          }     // block loop over dofs

        if (numberBandGroups > 1 && doCommAfterBandParal)
          {
            if (!dftParams.bandParalOpt)
              {
                MPI_Barrier(interBandGroupComm);
                const unsigned int blockSize =
                  dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 / sizeof(T);

                for (unsigned int i = 0; i < N * numLocalDofs; i += blockSize)
                  {
                    const unsigned int currentBlockSize =
                      std::min(blockSize, N * numLocalDofs - i);

                    MPI_Allreduce(MPI_IN_PLACE,
                                  subspaceVectorsArray + i,
                                  currentBlockSize,
                                  dataTypes::mpi_type_id(subspaceVectorsArray),
                                  MPI_SUM,
                                  interBandGroupComm);
                  }
              }
            else
              {
                MPI_Barrier(interBandGroupComm);

                std::vector<T> eigenVectorsBandGroup(numVectorsBandParal *
                                                       numLocalDofs,
                                                     T(0));
                std::vector<T> eigenVectorsBandGroupTransposed(
                  numVectorsBandParal * numLocalDofs, T(0));
                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroup[iNode * numVectorsBandParal + iWave] =
                      subspaceVectorsArray[iNode * N + startIndexBandParal +
                                           iWave];


                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < numVectorsBandParal;
                       ++iWave)
                    eigenVectorsBandGroupTransposed[iWave * numLocalDofs +
                                                    iNode] =
                      eigenVectorsBandGroup[iNode * numVectorsBandParal +
                                            iWave];

                std::vector<int> recvcounts(numberBandGroups, 0);
                std::vector<int> displs(numberBandGroups, 0);

                int recvcount = numVectorsBandParal * numLocalDofs;
                MPI_Allgather(&recvcount,
                              1,
                              MPI_INT,
                              &recvcounts[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                int displ = startIndexBandParal * numLocalDofs;
                MPI_Allgather(&displ,
                              1,
                              MPI_INT,
                              &displs[0],
                              1,
                              MPI_INT,
                              interBandGroupComm);

                std::vector<T> eigenVectorsTransposed(N * numLocalDofs, 0);
                MPI_Allgatherv(
                  &eigenVectorsBandGroupTransposed[0],
                  numVectorsBandParal * numLocalDofs,
                  dataTypes::mpi_type_id(&eigenVectorsBandGroupTransposed[0]),
                  &eigenVectorsTransposed[0],
                  &recvcounts[0],
                  &displs[0],
                  dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
                  interBandGroupComm);

                for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  for (unsigned int iWave = 0; iWave < N; ++iWave)
                    subspaceVectorsArray[iNode * N + iWave] =
                      eigenVectorsTransposed[iWave * numLocalDofs + iNode];
              }
          }
      }

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<double> &           mat,
        std::unordered_map<unsigned int, unsigned int> &           globalToLocalRowIdMap,
        std::unordered_map<unsigned int, unsigned int> &globalToLocalColumnIdMap);

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &mat,
        std::unordered_map<unsigned int, unsigned int> &globalToLocalRowIdMap,
        std::unordered_map<unsigned int, unsigned int> &globalToLocalColumnIdMap);

      template void
      fillParallelOverlapMatrix(
        const double *                                   X,
        const unsigned int                               XLocalSize,
        const unsigned int                               numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<double> &                 overlapMatPar,
        const dftParameters &                            dftParams);

      template void
      fillParallelOverlapMatrix(
        const std::complex<double> *                     X,
        const unsigned int                               XLocalSize,
        const unsigned int                               numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &   overlapMatPar,
        const dftParameters &                            dftParams);


      template void
      fillParallelOverlapMatrixMixedPrec<double, float>(
        const double *                                   X,
        const unsigned int                               XLocalSize,
        const unsigned int                               numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<double> &                 overlapMatPar,
        const dftParameters &                            dftParams);

      template void
      fillParallelOverlapMatrixMixedPrec<std::complex<double>,
                                         std::complex<float>>(
        const std::complex<double> *                     X,
        const unsigned int                               XLocalSize,
        const unsigned int                               numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &   overlapMatPar,
        const dftParameters &                            dftParams);

      template void
      subspaceRotation(
        double *           subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<double> &           rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranpose,
        const bool                                       isRotationMatLowerTria,
        const bool                                       doCommAfterBandParal);

      template void
      subspaceRotation(
        std::complex<double> *subspaceVectorsArray,
        const unsigned int    subspaceVectorsArrayLocalSize,
        const unsigned int    N,
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const MPI_Comm &                                    interBandGroupComm,
        const MPI_Comm &                                    mpiComm,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &rotationMatPar,
        const dftParameters &                               dftParams,
        const bool                                          rotationMatTranpose,
        const bool isRotationMatLowerTria,
        const bool doCommAfterBandParal);


      template void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<double> &                 mat,
        const double                                     scalar);

      template void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &   mat,
        const std::complex<double>                       scalar);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<double> &                 mat,
        const MPI_Comm &                                 interComm);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &   mat,
        const MPI_Comm &                                 interComm);

      template void
      subspaceRotationSpectrumSplit(
        const double *     X,
        double *           Y,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const unsigned int                               numberTopVectors,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<double> &           QMat,
        const dftParameters &                            dftParams,
        const bool                                       QMatTranspose);

      template void
      subspaceRotationSpectrumSplit(
        const std::complex<double> *X,
        std::complex<double> *      Y,
        const unsigned int          subspaceVectorsArrayLocalSize,
        const unsigned int          N,
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const unsigned int                                  numberTopVectors,
        const MPI_Comm &                                    interBandGroupComm,
        const MPI_Comm &                                    mpiComm,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &QMat,
        const dftParameters &                               dftParams,
        const bool                                          QMatTranspose);


      template void
      subspaceRotationSpectrumSplitMixedPrec<double, float>(
        const double *     X,
        double *           Y,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const unsigned int                               numberTopVectors,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<double> &           QMat,
        const dftParameters &                            dftParams,
        const bool                                       QMatTranspose);

      template void
      subspaceRotationSpectrumSplitMixedPrec<std::complex<double>,
                                             std::complex<float>>(
        const std::complex<double> *X,
        std::complex<double> *      Y,
        const unsigned int          subspaceVectorsArrayLocalSize,
        const unsigned int          N,
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const unsigned int                                  numberTopVectors,
        const MPI_Comm &                                    interBandGroupComm,
        const MPI_Comm &                                    mpiComm,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &QMat,
        const dftParameters &                               dftParams,
        const bool                                          QMatTranspose);

      template void
      subspaceRotationMixedPrec<double, float>(
        double *           subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<double> &           rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranspose,
        const bool                                       doCommAfterBandParal);

      template void
      subspaceRotationMixedPrec<std::complex<double>, std::complex<float>>(
        std::complex<double> *subspaceVectorsArray,
        const unsigned int    subspaceVectorsArrayLocalSize,
        const unsigned int    N,
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const MPI_Comm &                                    interBandGroupComm,
        const MPI_Comm &                                    mpiComm,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &rotationMatPar,
        const dftParameters &                               dftParams,
        const bool rotationMatTranspose,
        const bool doCommAfterBandParal);

      template void
      subspaceRotationCGSMixedPrec<double, float>(
        double *           subspaceVectorsArray,
        const unsigned int subspaceVectorsArrayLocalSize,
        const unsigned int N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm &                                 interBandGroupComm,
        const MPI_Comm &                                 mpiComm,
        const dftfe::ScaLAPACKMatrix<double> &           rotationMatPar,
        const dftParameters &                            dftParams,
        const bool                                       rotationMatTranspose,
        const bool                                       doCommAfterBandParal);

      template void
      subspaceRotationCGSMixedPrec<std::complex<double>, std::complex<float>>(
        std::complex<double> *subspaceVectorsArray,
        const unsigned int    subspaceVectorsArrayLocalSize,
        const unsigned int    N,
        const std::shared_ptr<const dftfe::ProcessGrid> &   processGrid,
        const MPI_Comm &                                    interBandGroupComm,
        const MPI_Comm &                                    mpiComm,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &rotationMatPar,
        const dftParameters &                               dftParams,
        const bool rotationMatTranspose,
        const bool doCommAfterBandParal);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<double> &                 mat,
        const MPI_Comm &                                 interComm,
        const unsigned int                               broadcastRoot);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &   mat,
        const MPI_Comm &                                 interComm,
        const unsigned int                               broadcastRoot);
    } // namespace internal
  }   // namespace linearAlgebraOperations
} // namespace dftfe
