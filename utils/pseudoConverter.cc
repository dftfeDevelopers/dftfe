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
// @author Shukan Parekh, Phani Motamarri
//
#include <headers.h>
#include <pseudoConverter.h>
#include <sys/stat.h>
#include <xmlTodftfeParser.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../pseudoConverters/upfToxml.h"

namespace dftfe
{
  namespace pseudoUtils
  {
    bool
    ends_with(std::string const &value, std::string const &ending)
    {
      if (ending.size() > value.size())
        return false;
      return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    // Function to check if the file extension .qso
    bool
    isupf(const std::string &fname)
    {
      return (ends_with(fname, ".upf") || ends_with(fname, ".UPF"));
    }



    int
    convert(std::string &fileName,const MPI_Comm & mpi_comm_parent)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0));

      xmlTodftfeParser xmlParse;

      std::ifstream input_file;
      input_file.open(fileName);

      AssertThrow(!input_file.fail(),
                  dealii::ExcMessage(
                    "Not a valid list of pseudopotential files "));

      std::string              z;
      std::string              toParse;
      std::vector<std::string> atomTypes;
      unsigned int             nlccSum = 0;

      while (input_file >> z >> toParse)
        {
          std::string tempFolder = "temp";
          mkdir(tempFolder.c_str(), ACCESSPERMS);
          std::string newFolder = tempFolder + "/" + "z" + z;
          mkdir(newFolder.c_str(), ACCESSPERMS);
          AssertThrow(
            isupf(toParse),
            dealii::ExcMessage(
              "Not a valid pseudopotential format and upf format only is currently supported"));

          atomTypes.push_back(z);

          if (isupf(toParse))
            {
              // std::string xmlFileName = newFolder + "/" + toParse.substr(0,
              // toParse.find(".upf"));
              std::string xmlFileName = newFolder + "/" + "z" + z + ".xml";
              int         errorFlag;
              if (dftParameters::pseudoTestsFlag)
                {
                  std::string dftPath = DFT_PATH;
#ifdef USE_COMPLEX
                  std::string newPath =
                    dftPath + "/tests/dft/pseudopotential/complex/" + toParse;
#else
                  std::string newPath =
                    dftPath + "/tests/dft/pseudopotential/real/" + toParse;
#endif

                  unsigned int nlccFlag = 0;
                  errorFlag = upfToxml(newPath, xmlFileName, nlccFlag);
                  nlccSum += nlccFlag;
                }
              else
                {
                  unsigned int nlccFlag = 0;
                  errorFlag = upfToxml(toParse, xmlFileName, nlccFlag);
                  nlccSum += nlccFlag;

                  if (dftParameters::verbosity >= 1)
                    {
                      if (nlccFlag > 0)
                        pcout << " Reading Pseudopotential File: " << toParse
                              << ", with atomic number: " << z
                              << ", and has data for nonlinear core-correction"
                              << std::endl;
                      else
                        pcout << " Reading Pseudopotential File: " << toParse
                              << ", with atomic number: " << z
                              << ", and has no nonlinear core-correction"
                              << std::endl;
                    }
                }

              AssertThrow(errorFlag == 0,
                          dealii::ExcMessage("Error in reading upf format"));

              xmlParse.parseFile(xmlFileName);
              xmlParse.outputData(newFolder);
            }
        }

      // if(nlccSum > 0)
      // dftParameters::nonLinearCoreCorrection = true;

      // if(dftParameters::nonLinearCoreCorrection == true)
      // pcout<<"Few atoms have pseudopotentials with nonlinear core
      // correction"<<std::endl;


      AssertThrow(
        atomTypes.size() == dftParameters::natomTypes,
        dealii::ExcMessage(
          "Number of atom types in your pseudopotential file does not match with that given in the parameter file"));

      return nlccSum;
    }

  } // namespace pseudoUtils

} // namespace dftfe
