//
// -------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
//
// @author Sambit Das
//
#ifndef elpaScalaManager_h
#define elpaScalaManager_h

#include "headers.h"
#include "process_grid.h"

#include <vector>
extern "C"
{
#include <elpa.hh>
}

namespace dftfe
{
  inline void
  elpaCholesky(elpa_t &handle, double *a, int *error)
  {
    elpa_cholesky_d(handle, a, error);
  }

  inline void
  elpaCholesky(elpa_t &handle, std::complex<double> *a, int *error)
  {
    elpa_cholesky_dc(handle, reinterpret_cast<_Complex double *>(a), error);
  }

  inline void
  elpaEigenvectors(elpa_t &handle, double *a, double *ev, double *q, int *error)
  {
    elpa_eigenvectors_d(handle, a, ev, q, error);
  }

  inline void
  elpaEigenvectors(elpa_t &              handle,
                   std::complex<double> *a,
                   double *              ev,
                   std::complex<double> *q,
                   int *                 error)
  {
    elpa_eigenvectors_dc(handle,
                         reinterpret_cast<_Complex double *>(a),
                         ev,
                         reinterpret_cast<_Complex double *>(q),
                         error);
  }


  /**
   * @brief Manager class for ELPA and ScaLAPACK
   *
   * @author Sambit Das
   */
  class elpaScalaManager
  {
    //
    // methods
    //
  public:
    unsigned int
    getScalapackBlockSize() const;

    std::shared_ptr<const dftfe::ProcessGrid>
    getProcessGridDftfeScalaWrapper() const;

    void
    processGridOptionalELPASetup(const unsigned int na, const unsigned int nev);

    void
    elpaDeallocateHandles(const unsigned int na, const unsigned int nev);

    elpa_t &
    getElpaHandle();

    elpa_t &
    getElpaHandlePartialEigenVec();

    elpa_autotune_t &
    getElpaAutoTuneHandle();


    /**
     * @brief Get relevant mpi communicator
     *
     * @return mpi communicator
     */
    const MPI_Comm &
    getMPICommunicator() const;


    /**
     * @brief Constructor.
     */
    elpaScalaManager(const MPI_Comm &mpi_comm_replica);

    /**
     * @brief Destructor.
     */
    ~elpaScalaManager();

    //
    // mpi communicator
    //
    MPI_Comm d_mpi_communicator;

    /// ELPA handle
    elpa_t d_elpaHandle;

    /// ELPA handle for partial eigenvectors of full proj ham
    elpa_t d_elpaHandlePartialEigenVec;

    /// ELPA autotune handle
    elpa_autotune_t d_elpaAutoTuneHandle;

    /// processGrid mpi communicator
    MPI_Comm d_processGridCommunicatorActive;

    MPI_Comm d_processGridCommunicatorActivePartial;


    /// ScaLAPACK distributed format block size
    unsigned int d_scalapackBlockSize;

    std::shared_ptr<const dftfe::ProcessGrid> d_processGridDftfeWrapper;
  };

  /*--------------------- Inline functions --------------------------------*/

#ifndef DOXYGEN
  inline unsigned int
  elpaScalaManager::getScalapackBlockSize() const
  {
    return d_scalapackBlockSize;
  }

  inline std::shared_ptr<const dftfe::ProcessGrid>
  elpaScalaManager::getProcessGridDftfeScalaWrapper() const
  {
    return d_processGridDftfeWrapper;
  }

  inline elpa_t &
  elpaScalaManager::getElpaHandle()
  {
    return d_elpaHandle;
  }

  inline elpa_t &
  elpaScalaManager::getElpaHandlePartialEigenVec()
  {
    return d_elpaHandlePartialEigenVec;
  }


  inline elpa_autotune_t &
  elpaScalaManager::getElpaAutoTuneHandle()
  {
    return d_elpaAutoTuneHandle;
  }
#endif // ifndef DOXYGEN

} // namespace dftfe
#endif
