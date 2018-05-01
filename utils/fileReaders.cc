// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//
#include <fileReaders.h>
#include <headers.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace dftfe {

    namespace dftUtils{

	//Utility functions to read external files relevant to DFT
	void readFile(const unsigned int numColumns,
		      std::vector<std::vector<double> > &data,
		      const std::string & fileName)
	{
	  std::vector<double> rowData(numColumns, 0.0);
	  std::ifstream readFile(fileName.c_str());
	  if(readFile.fail())
	    {
	      std::cerr<< "Error opening file: " << fileName.c_str() << std::endl;
	      exit(-1);
	    }

	  //
	  // String to store line and word
	  //
	  std::string readLine;
	  std::string word;

	  //
	  // column index
	  //
	  int columnCount;

	  if(readFile.is_open())
	    {
	      while (std::getline(readFile, readLine))
		{
		  std::istringstream iss(readLine);

		  columnCount = 0;

		  while(iss >> word && columnCount < numColumns)
		    rowData[columnCount++] = atof(word.c_str());

		  data.push_back(rowData);
		}
	    }
	  readFile.close();
	}

        int readPsiFile(const unsigned int numColumns,
	 	        std::vector<std::vector<double> > &data,
		        const std::string & fileName)
	{
	  std::vector<double> rowData(numColumns, 0.0);
	  std::ifstream readFile(fileName.c_str());

	  if(readFile.fail())
	    {
	      //std::cerr<< "Warning: Psi file: " << fileName.c_str() << " not found "<<std::endl;
	      return 0;
	    }

	  //
	  // String to store line and word
	  //
	  std::string readLine;
	  std::string word;

	  //
	  // column index
	  //
	  int columnCount;

	  if(readFile.is_open())
	    {
	      while (std::getline(readFile, readLine))
		{
		  std::istringstream iss(readLine);

		  columnCount = 0;

		  while(iss >> word && columnCount < numColumns)
		    rowData[columnCount++] = atof(word.c_str());

		  data.push_back(rowData);
		}
	    }
	  readFile.close();
	  return 1;
	}

	void readRelaxationFlagsFile(const unsigned int numColumns,
				     std::vector<std::vector<int> > &data,
				     const std::string & fileName)
	{
	  std::vector<int> rowData(numColumns, 0.0);
	  std::ifstream readFile(fileName.c_str());
	  if(readFile.fail())
	    {
	      std::cerr<< "Error opening file: " << fileName.c_str() << std::endl;
	      exit(-1);
	    }

	  //
	  // String to store line and word
	  //
	  std::string readLine;
	  std::string word;

	  //
	  // column index
	  //
	  int columnCount;

	  if(readFile.is_open())
	    {
	      while (std::getline(readFile, readLine))
		{
		  std::istringstream iss(readLine);

		  columnCount = 0;

		  while(iss >> word && columnCount < numColumns)
		    rowData[columnCount++] = atoi(word.c_str());

		  data.push_back(rowData);
		}
	    }
	  readFile.close();
	  return;

	}

	// Move/rename a checkpoint file
	void moveFile(const std::string &old_name, const std::string &new_name)
	{

	    int error = system (("mv " + old_name + " " + new_name).c_str());

	    // If the above call failed, e.g. because there is no command-line
	    // available, try with internal functions.
	    if (error != 0)
	    {
		std::ifstream ifile(new_name);
		if (static_cast<bool>(ifile))
		{
		    error = remove(new_name.c_str());
		    AssertThrow (error == 0, dealii::ExcMessage(std::string ("Unable to remove file: "
		    + new_name
		    + ", although it seems to exist. "
		    + "The error code is "
		    + dealii::Utilities::to_string(error) + ".")));
		}

		error = rename(old_name.c_str(),new_name.c_str());
		AssertThrow (error == 0, dealii::ExcMessage(std::string ("Unable to rename files: ")
		+
		old_name + " -> " + new_name
		+ ". The error code is "
		+ dealii::Utilities::to_string(error) + "."));
	    }
	}

	void verifyCheckpointFileExists(const std::string & filename)
	{
	      std::ifstream in (filename);
	      if (!in)
	      {
		AssertThrow (false,
		   dealii::ExcMessage (std::string("DFT-FE Error: You are trying to restart a previous computation, "
		   "but the restart file <")
		    +
		    filename
		    +
		    "> does not appear to exist!"));
	      }
	 }


	 void writeDataIntoFile(const std::vector<std::vector<double> > &data,
			        const std::string & fileName)
	 {
	     if (std::ifstream(fileName) && dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		 moveFile(fileName, fileName+".old");

	     std::ofstream outFile(fileName);
	     if (outFile.is_open())
	     {
		for (unsigned int irow=0; irow < data.size(); ++irow)
		{
		     for (unsigned int icol=0; icol < data[irow].size(); ++icol)
		     {
			 outFile << std::setprecision(10)<< data[irow][icol];
			 if (icol< data[irow].size()-1)
			     outFile<<" ";
		     }
		     outFile<<std::endl;
		}

		outFile.close();
	     }
	 }

    }

}
