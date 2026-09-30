#include "QuadDataCompositeWrite.h"
#include <cstdio>

namespace dftfe
{
  namespace dftUtils
  {
    QuadDataCompositeWrite::QuadDataCompositeWrite(
      const std::vector<double> &vals)
      : d_vals(vals)
      , d_charspernum(20)
    {}

    void
    QuadDataCompositeWrite::getCharArray(char *data)
    {
      const char *doubleFmt    = "%19.12e ";
      const char *doubleEndFmt = "%19.12e\n";

      dftfe::Int count = 0;
      for (dftfe::Int i = 0; i < d_vals.size() - 1; ++i)
        {
          sprintf(&data[count], doubleFmt, d_vals[i]);
          count += d_charspernum;
        }
      sprintf(&data[count], doubleEndFmt, d_vals[d_vals.size() - 1]);
    }

    void
    QuadDataCompositeWrite::getMPIDataType(MPI_Datatype *mpi_datatype)
    {
      dftfe::Int numberChars = getNumberCharsPerCompositeData();
      MPI_Type_contiguous(numberChars, MPI_CHAR, mpi_datatype);
      MPI_Type_commit(mpi_datatype);
    }

    dftfe::Int
    QuadDataCompositeWrite::getNumberCharsPerCompositeData()
    {
      return d_charspernum * d_vals.size();
    }
  } // namespace dftUtils
} // namespace dftfe
