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

//source file for implementation of boundary conditions

//boundary condition function
template <int dim>
class OnebyRBoundaryFunction : public Function<dim>{
public:
  std::vector<std::vector<double> >* atomLocationsPtr;
  OnebyRBoundaryFunction(std::vector<std::vector<double> >& atomLocations): Function<dim>(), atomLocationsPtr(&atomLocations){}
  virtual double value (const Point<dim> &p, const unsigned int component = 0) const{
    double value=0;
    //loop for multiple atoms
    if(isPseudopotential)
      {
	for (unsigned int z=0; z<atomLocationsPtr->size(); z++){
	  Point<3> atom((*atomLocationsPtr)[z][2],(*atomLocationsPtr)[z][3],(*atomLocationsPtr)[z][4]);
	  value+= -(*atomLocationsPtr)[z][1]/p.distance(atom);
	}
      }
    else
      {
	for (unsigned int z=0; z<atomLocationsPtr->size(); z++){
	  Point<3> atom((*atomLocationsPtr)[z][2],(*atomLocationsPtr)[z][3],(*atomLocationsPtr)[z][4]);
	  value+= -(*atomLocationsPtr)[z][0]/p.distance(atom);
	}
      }
    return value;
  }
};
