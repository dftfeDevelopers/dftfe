//
// Created by nelrufus on 7/17/18.
//

#ifndef DFTFE_NODALDATA_H
#define DFTFE_NODALDATA_H

#include "CompositeData.h"
#include <vector>

namespace dftfe
{
  namespace dftUtils
  {
    class NodalData : public CompositeData
    {
    public:
      NodalData(const std::vector<double> &vals);

      virtual void
      getCharArray(char *data) override;

      virtual void
      getMPIDataType(MPI_Datatype *mpi_datatype) override;

      virtual int
      getNumberCharsPerCompositeData() override;

    private:
      unsigned int        d_charspernum;
      std::vector<double> d_vals;
    };
  } // namespace dftUtils
} // namespace dftfe
#endif // DFTFE_NODALDATA_H
