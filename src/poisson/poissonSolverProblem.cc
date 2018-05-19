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

#include <poissonSolverProblem.h>
#include <constants.h>

namespace dftfe {
    //
    //constructor
    //
    template<unsigned int FEOrder>
    poissonSolverProblem<FEOrder>::poissonSolverProblem(const  MPI_Comm &mpi_comm):
      mpi_communicator (mpi_comm),
      n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm)),
      this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm)),
      pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    {

    }

    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::reinit
                    (const dealii::MatrixFree<3,double> & matrixFreeData,
		     vectorType & x,
		     const dealii::ConstraintMatrix & constraintMatrix,
		     const unsigned int matrixFreeVectorComponent,
	             const std::map<dealii::types::global_dof_index, double> & atoms,
		     const std::map<dealii::CellId,std::vector<double> > & rhoValues,
		     const bool isComputeDiagonalA)
    {
        d_matrixFreeDataPtr=&matrixFreeData;
	d_xPtr=&x;
	d_constraintMatrixPtr=&constraintMatrix;
	d_matrixFreeVectorComponent=matrixFreeVectorComponent;
	d_rhoValuesPtr=&rhoValues;
	d_atomsPtr=&atoms;

	if (isComputeDiagonalA)
	  computeDiagonalA();
    }


    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::reinit
                          (const dealii::MatrixFree<3,double> & matrixFreeData,
		           vectorType & x,
		           const dealii::ConstraintMatrix & constraintMatrix,
		           const unsigned int matrixFreeVectorComponent,
	                   const std::map<dealii::types::global_dof_index, double> & atoms,
			   const bool isComputeDiagonalA)
    {
        d_matrixFreeDataPtr=&matrixFreeData;
	d_xPtr=&x;
	d_constraintMatrixPtr=&constraintMatrix;
	d_matrixFreeVectorComponent=matrixFreeVectorComponent;
	d_rhoValuesPtr=NULL;
	d_atomsPtr=&atoms;

	if (isComputeDiagonalA)
	  computeDiagonalA();
    }


    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::distributeX()
    {
       d_constraintMatrixPtr->distribute(*d_xPtr);
    }

    template<unsigned int FEOrder>
    vectorType & poissonSolverProblem<FEOrder>::getX()
    {
       return *d_xPtr;
    }

    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::computeRhs(vectorType  & rhs)
    {

	rhs.reinit(*d_xPtr);

	const dealii::DoFHandler<3> & dofHandler=
	    d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

	dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
        dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
        const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
        const unsigned int   num_quad_points = quadrature.size();
        dealii::Vector<double>  elementalRhs(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

        //rhs contribution from static condensation of dirichlet boundary conditions
	typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
	for(; cell!=endc; ++cell)
	      if(cell->is_locally_owned())
		{
		  //compute values for the current element
		  fe_values.reinit(cell);
		  cell->get_dof_indices(local_dof_indices);

		  elementalRhs=0.0;
		  bool assembleFlag=false;

		  //local poissonClass operator
		  for(unsigned int j = 0; j < dofs_per_cell; ++j)
		    {
		      unsigned int columnID = local_dof_indices[j];
		      if(d_constraintMatrixPtr->is_inhomogeneously_constrained(columnID))
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
			    {
			      //compute contribution to rhs
			      double localStiffnessMatIJ = 0.0;
			      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
				  localStiffnessMatIJ += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i,q_point)*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);

			      elementalRhs(i)-=d_constraintMatrixPtr->
				  get_inhomogeneity(columnID)*localStiffnessMatIJ;
			      if (!assembleFlag)
				  assembleFlag=true;
			    }

		    }
		  if(assembleFlag)
		      d_constraintMatrixPtr->distribute_local_to_global(elementalRhs,local_dof_indices,rhs);
		}

        //rhs contribution from electronic charge
	if (d_rhoValuesPtr)
	{
	    cell = dofHandler.begin_active();
	    for(; cell!=endc; ++cell)
		if (cell->is_locally_owned())
		{
		   fe_values.reinit (cell);
		   elementalRhs=0.0;

		   for (unsigned int i=0; i<dofs_per_cell; ++i)
		       for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			      elementalRhs(i) += fe_values.shape_value(i, q_point)*d_rhoValuesPtr->find(cell->id())->second[q_point]*fe_values.JxW (q_point);

		   //assemble to global data structures
		   cell->get_dof_indices (local_dof_indices);
		   d_constraintMatrixPtr->distribute_local_to_global(elementalRhs, local_dof_indices, rhs);
		}
	}

	//rhs contribution from atomic charge at fem nodes
        for (std::map<dealii::types::global_dof_index, double>::const_iterator it=(*d_atomsPtr).begin(); it!=(*d_atomsPtr).end(); ++it)
	{
	  std::vector<dealii::ConstraintMatrix::size_type> local_dof_indices_origin(1, it->first); //atomic node
	  dealii::Vector<double> cell_rhs_origin (1);
	  cell_rhs_origin(0)=-(it->second); //atomic charge

	  d_constraintMatrixPtr->distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
	}

        //MPI operation to sync data
        rhs.compress(dealii::VectorOperation::add);

        //FIXME: check if this is really required
        d_constraintMatrixPtr->set_zero(rhs);

    }

    //Matrix-Free Jacobi preconditioner application
    template<unsigned int FEOrder>
    void  poissonSolverProblem<FEOrder>::precondition_Jacobi(vectorType& dst,
	                                                      const vectorType& src,
						              const double omega) const
    {
      dst = src;
      dst.scale(d_diagonalA);
    }

    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::computeDiagonalA()
    {
	d_diagonalA.reinit(*d_xPtr);

	const dealii::DoFHandler<3> & dofHandler=
	    d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

	dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
        dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
        const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
        const unsigned int   num_quad_points = quadrature.size();
        dealii::Vector<double>  elementalDiagonalA(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

        //parallel loop over all elements
        typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
        for(; cell!=endc; ++cell)
	  if (cell->is_locally_owned())
	    {
	      fe_values.reinit (cell);

	      cell->get_dof_indices (local_dof_indices);

	      elementalDiagonalA=0.0;
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		  for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		      elementalDiagonalA(i) += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (i, q_point))*fe_values.JxW(q_point);

	      d_constraintMatrixPtr->distribute_local_to_global(elementalDiagonalA,
		                                                local_dof_indices,
								d_diagonalA);
	    }

	//MPI operation to sync data
	d_diagonalA.compress(dealii::VectorOperation::add);

	for(dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
	      if(d_diagonalA.in_local_range(i))
		  if(! d_constraintMatrixPtr->is_constrained(i))
		      d_diagonalA(i) = 1.0/d_diagonalA(i);

	d_diagonalA.compress(dealii::VectorOperation::insert);
    }

    //Ax
    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::AX (const dealii::MatrixFree<3,double>  &matrixFreeData,
				     vectorType &dst,
				     const vectorType &src,
				     const std::pair<unsigned int,unsigned int> &cell_range) const
    {
      dealii::VectorizedArray<double>  quarter = dealii::make_vectorized_array (1.0/(4.0*M_PI));

      dealii::FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval(matrixFreeData,
	                                                             d_matrixFreeVectorComponent,
								     0);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
	{
	  fe_eval.reinit(cell);
	  fe_eval.read_dof_values(src);
	  fe_eval.evaluate(false,true,false);
	  for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
	    {
	      fe_eval.submit_gradient(fe_eval.get_gradient(q)*quarter, q);
	    }
	  fe_eval.integrate(false, true);
	  fe_eval.distribute_local_to_global(dst);
	}
    }


    template<unsigned int FEOrder>
    void poissonSolverProblem<FEOrder>::vmult(vectorType &Ax,const vectorType &x) const
    {
      Ax=0.0;
      d_matrixFreeDataPtr->cell_loop (&poissonSolverProblem<FEOrder>::AX, this, Ax, x);
    }


    template class poissonSolverProblem<1>;
    template class poissonSolverProblem<2>;
    template class poissonSolverProblem<3>;
    template class poissonSolverProblem<4>;
    template class poissonSolverProblem<5>;
    template class poissonSolverProblem<6>;
    template class poissonSolverProblem<7>;
    template class poissonSolverProblem<8>;
    template class poissonSolverProblem<9>;
    template class poissonSolverProblem<10>;
    template class poissonSolverProblem<11>;
    template class poissonSolverProblem<12>;
    template class poissonSolverProblem<13>;
    template class poissonSolverProblem<14>;
    template class poissonSolverProblem<15>;
    template class poissonSolverProblem<16>;
}
