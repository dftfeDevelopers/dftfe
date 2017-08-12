// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//Utility functions to read external files relevant to DFT
void readFile(unsigned int numColumns, 
	      std::vector<std::vector<double> > &data, 
	      std::string fileName)
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
  return;
}

int readPsiFile(unsigned int numColumns, 
		 std::vector<std::vector<double> > &data, 
		 std::string fileName)
{
  std::vector<double> rowData(numColumns, 0.0);
  std::ifstream readFile(fileName.c_str());

  if(readFile.fail()) 
    {
      std::cerr<< "Warning: Psi file: " << fileName.c_str() << " not found "<<std::endl;
      
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
