// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri
//

#ifndef PERIODICTABLE_H
#define PERIODICTABLE_H

#include <map>
#include <vector>
#include <string>

namespace dftfe
{
	//
	//Declare pseudoUtils function
	//

	/** @file PeriodicTable.h
	 *  @brief stores a map between atomic number and atomic symbol and atomic mass
	 *
	 *
	 *  @author Phani Motamarri
	 */
	namespace pseudoUtils
	{
		struct Element
		{
			int z;
			std::string symbol;
			std::string config;
			double mass;
			Element(int zz, std::string s, std::string c, double m) : z(zz), symbol(s), config(c),mass(m) {}
		};

		class PeriodicTable
		{
			private:

				std::vector<Element> ptable;
				std::map<std::string,int> zmap;

			public:

				PeriodicTable(void);
				int z(std::string symbol) const;
				std::string symbol(int zval) const;
				std::string configuration(int zval) const;
				std::string configuration(std::string symbol) const;
				double mass(int zval) const;
				double mass(std::string symbol) const;
				int size(void) const;

		};
	}
}
#endif
