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
//
// @author Shiva Rudraraju (2016), Phani Motamarri (2018), Sambit Das (2017)
//



#ifdef ENABLE_PERIODIC_BC
#include "initkPointData.cc"
#endif

//
//source file for dft class initializations
//


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initUnmovedTriangulation(Triangulation<3,3> & triangulation)
{
  computing_timer.enter_section("unmoved setup");


  //
  //initialize FE objects
  //
  dofHandler.clear();dofHandlerEigen.clear();
  dofHandler.initialize(triangulation,FE);
  dofHandlerEigen.initialize(triangulation,FEEigen);
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  //
  //extract locally owned dofs
  //
  locally_owned_dofs = dofHandler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandler, locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);


  locally_owned_dofsEigen = dofHandlerEigen.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandlerEigen, locally_relevant_dofsEigen);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  

  //
  //Extract real and imag DOF indices from the global vector - this will be needed in XHX operation, etc.
  //
#ifdef ENABLE_PERIODIC_BC
  FEValuesExtractors::Scalar real(0); //For Eigen
  ComponentMask componentMaskForRealDOF = FEEigen.component_mask (real);
  std::vector<bool> selectedDofsReal(locally_owned_dofsEigen.n_elements(), false);
  DoFTools::extract_dofs(dofHandlerEigen, componentMaskForRealDOF, selectedDofsReal);
  std::vector<unsigned int> local_dof_indices(locally_owned_dofsEigen.n_elements());
  locally_owned_dofsEigen.fill_index_vector(local_dof_indices);
  for (unsigned int i = 0; i < locally_owned_dofsEigen.n_elements(); i++)
    {
      if (selectedDofsReal[i]) 
	{
	  local_dof_indicesReal.push_back(local_dof_indices[i]);
	  localProc_dof_indicesReal.push_back(i);
	}
      else
	{
	  local_dof_indicesImag.push_back(local_dof_indices[i]);
	  localProc_dof_indicesImag.push_back(i);	  
	}
    }
#endif


  
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;


  //
  //constraints
  //

  //
  //hanging node constraints
  //
  constraintsNone.clear(); constraintsNoneEigen.clear();
  constraintsNone.reinit(locally_relevant_dofs); constraintsNoneEigen.reinit(locally_relevant_dofsEigen);


#ifdef ENABLE_PERIODIC_BC
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator> > periodicity_vector2, periodicity_vector2Eigen;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(dofHandler, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2);
      GridTools::collect_periodic_faces(dofHandlerEigen, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2Eigen);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2, constraintsNone);
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2Eigen, constraintsNoneEigen);

  constraintsNone.close();
  constraintsNoneEigen.close();

  if(!dftParameters::meshFileName.empty())
    {
      applyPeriodicBCHigherOrderNodes();
    }

#endif

  //
  //create a constraint matrix without only hanging node constraints 
  //
  d_noConstraints.clear();d_noConstraintsEigen.clear();
  
  DoFTools::make_hanging_node_constraints(dofHandler, d_noConstraints);
  DoFTools::make_hanging_node_constraints(dofHandlerEigen,d_noConstraintsEigen);
  d_noConstraints.close();d_noConstraintsEigen.close();
  

  //
  //merge hanging node constraint matrix with constrains None and constraints None eigen
  //
  constraintsNone.merge(d_noConstraints,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  constraintsNoneEigen.merge(d_noConstraintsEigen,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  constraintsNone.close();
  constraintsNoneEigen.close();

  forcePtr->initUnmoved(triangulation);

  //
  //Initialize libxc (exchange-correlation)
  //
  int exceptParamX, exceptParamC;
  unsigned int xc_id = dftParameters::xc_id;


  if(xc_id == 1)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
    }
  else if(xc_id == 2)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_PW,XC_UNPOLARIZED);
    }
  else if(xc_id == 3)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
    }
  else if(xc_id == 4)
    {
      exceptParamX = xc_func_init(&funcX,XC_GGA_X_PBE,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_GGA_C_PBE,XC_UNPOLARIZED);
    }
  else if(xc_id > 4)
    {
      pcout<<"-------------------------------------"<<std::endl;
      pcout<<"Exchange or Correlation Functional not found"<<std::endl;
      pcout<<"-------------------------------------"<<std::endl;
      exit(-1);
    }

  if(exceptParamX != 0 || exceptParamC != 0)
    {
      pcout<<"-------------------------------------"<<std::endl;
      pcout<<"Exchange or Correlation Functional not found"<<std::endl;
      pcout<<"-------------------------------------"<<std::endl;
      exit(-1);
    }


  computing_timer.exit_section("unmoved setup");    
}
