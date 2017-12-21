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
// @author Sambit Das (2017)
//

#ifndef fileReaders_H_
#define fileReaders_H_
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>  
#include <vector>

//
//Declare dftUtils functions
//
namespace dftUtils
{

void readFile(unsigned int numColumns, 
	      std::vector<std::vector<double> > &data, 
	      std::string fileName);

int readPsiFile(unsigned int numColumns, 
		 std::vector<std::vector<double> > &data, 
		 std::string fileName);
};

#endif
