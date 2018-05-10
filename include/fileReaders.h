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

/** @file fileReaders.h
 *  @brief Contains commonly used I/O file utils functions
 *
 *  @author Shiva Rudraraju, Phani Motamarri, Sambit Das
 */

#ifndef fileReaders_H_
#define fileReaders_H_
#include <string>
#include <vector>

namespace dftfe {
    namespace dftUtils
    {

      /**
       * @brief Read from file containing only double data in columns.
       *
       * @param numColumns[in] number of data columsn in the file to be read
       * @param data[out] output double data in [rows][columns] format
       * @param fileName[in]
       */
	void readFile(const unsigned int numColumns,
		      std::vector<std::vector<double> > &data,
		      const std::string & fileName);
      /**
       * @brief Read from file containing only double data in columns.
       */
	int readPsiFile(const unsigned int numColumns,
			std::vector<std::vector<double> > &data,
			const std::string & fileName);

      /**
       * @brief Write data into file containing only double data in rows and columns.
       *
       * @param data[in] input double data in [rows][columns] format
       * @param fileName[in]
       */
	void writeDataIntoFile(const std::vector<std::vector<double> > &data,
			       const std::string & fileName);

      /**
       * @brief Read from file containing only integer data in columns.
       */
	void readRelaxationFlagsFile(const unsigned int numColumns,
				     std::vector<std::vector<int> > &data,
				     const std::string & fileName);

      /**
       * @brief Move/rename checkpoint file.
       */
	void moveFile(const std::string &old_name,
		      const std::string &new_name);

      /**
       * @brief Verify if checkpoint file exists.
       */
	void verifyCheckpointFileExists(const std::string & filename);
    };

}
#endif
