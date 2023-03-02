//
// Created by nelrufus on 7/17/18.
//

#ifndef MPI_COMPOSITEDATA_H
#define MPI_COMPOSITEDATA_H

#include <mpi.h>
namespace dftfe {
  namespace dftUtils {

    class CompositeData {

      public:
	CompositeData(){};

	virtual void getCharArray(char *data) = 0;

	virtual void getMPIDataType(MPI_Datatype *mpi_datatype) = 0;

	virtual int getNumberCharsPerCompositeData() = 0;
    };
  } //namespace dftUtils
} // namespace dftfe

#endif // MPI_COMPOSITEDATA_H
