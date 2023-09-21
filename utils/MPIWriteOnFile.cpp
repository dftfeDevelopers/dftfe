// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2023 The Regents of the University of Michigan and DFT-FE
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
// @author nelrufus, vishal subramanian
//

//
// Created by nelrufus on 7/17/18.
//

#include "MPIWriteOnFile.h"
#include <iostream>
#include <numeric>

namespace dftfe
{
  namespace dftUtils
  {
    void
    MPIWriteOnFile::writeData(const std::vector<CompositeData *> &data,
                              const std::string &                 fileName,
                              const MPI_Comm &mpiCommunicator)
    {
      // assert(data.size() > 0);

      int rank, size;
      MPI_Comm_size(mpiCommunicator, &size);
      MPI_Comm_rank(mpiCommunicator, &rank);


      // create char array
      const auto localSize          = data.size();
      const int charsPerDataElement = data[0]->getNumberCharsPerCompositeData();

      // FIXME Playing a dangerous game ... the +1 is for the trailing NULL in
      // the sprintf
      std::vector<char> dataTxt(localSize * charsPerDataElement + 1);
      // auto dataTxt = new char[localSize*charsPerDataElement];
      for (auto i = decltype(localSize){0}; i < localSize; ++i)
        data[i]->getCharArray(&dataTxt[i * charsPerDataElement]);

      // create data type
      MPI_Datatype newType;
      data[0]->getMPIDataType(&newType);

      // create local array for set view
      std::vector<unsigned long> sizes(static_cast<unsigned long>(size)),
        offset(static_cast<unsigned long>(size), 0);
      MPI_Allgather(&localSize,
                    1,
                    MPI_UNSIGNED_LONG,
                    &sizes[0],
                    1,
                    MPI_UNSIGNED_LONG,
                    mpiCommunicator);

      const auto globalSize = (std::accumulate(sizes.begin(),
                                               sizes.end(),
                                               static_cast<unsigned long>(0)));

      for (int i = 1; i < size; ++i)
        {
          offset[i] = offset[i - 1] + sizes[i - 1];
        }


      int globalsizes[1] = {static_cast<int>(globalSize)};
      int localsizes[1]  = {static_cast<int>(localSize)};
      int starts[1]      = {static_cast<int>(offset[rank])};
      int order          = MPI_ORDER_C;

      MPI_Datatype localarray;
      MPI_Type_create_subarray(
        1, globalsizes, localsizes, starts, order, newType, &localarray);
      MPI_Type_commit(&localarray);

      // open file and write
      /* open the file, and set the view */
      MPI_File file;
      MPI_File_open(mpiCommunicator,
                    fileName.c_str(),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL,
                    &file);

      MPI_File_set_view(file, 0, MPI_CHAR, localarray, "native", MPI_INFO_NULL);

      MPI_Status status;
      MPI_File_write_all(file, &(dataTxt[0]), localSize, newType, &status);

      MPI_File_close(&file);

      MPI_Type_free(&localarray);
      MPI_Type_free(&newType);
      // delete [] dataTxt;

      return;
    }
  } // namespace dftUtils
} // namespace dftfe
