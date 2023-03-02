//
// Created by nelrufus on 7/17/18.
//

#ifndef MPI_MPIWRITEONFILE_H
#define MPI_MPIWRITEONFILE_H

#include <CompositeData.h>
#include <string>
#include <vector>

namespace dftfe {
  namespace dftUtils {

	class MPIWriteOnFile {

	  public:
		static void writeData(const std::vector<CompositeData *> &data,
			const std::string &fileName,
			const MPI_Comm &mpiCommunicator);
	};

  } // namespace dftUtils
} // namespace dftfe

#endif // MPI_MPIWRITEONFILE_H
