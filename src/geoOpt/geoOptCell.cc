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
// @author Sambit Das (2018)
//

#ifdef ENABLE_PERIODIC_BC 
#include <geoOptCell.h>
#include <cgPRPNonLinearSolver.h>
#include <force.h>
#include <meshGenerator.h>
#include <dft.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::writeMesh(std::string meshFileName)
 {
      dftPtr->writeMesh(meshFileName);
 }

//
//constructor
//
template<unsigned int FEOrder>
geoOptCell<FEOrder>::geoOptCell(dftClass<FEOrder>* _dftPtr, MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  mpi_communicator (mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//
template<unsigned int FEOrder>
void geoOptCell<FEOrder>::init()
{
    // initialize d_strainEpsilon to identity
    for (unsigned int i=0; i<3;++i)
	d_strainEpsilon[i][i]=0.0;

    // stress tensor is a symmetric second order with six independent components
    d_relaxationFlags.clear();
    d_relaxationFlags.resize(6,0);

    if (dftParameters::stressConstraintType=1)//isotropic (shape fixed volume optimization)
    {
	d_relaxationFlags[0]=1;
        d_relaxationFlags[3]=1;
        d_relaxationFlags[5]=1;	
    }
    else if (dftParameters::stressConstraintType=2)// volume fixed shape optimization
    {
	d_relaxationFlags[1]=1;
        d_relaxationFlags[2]=1;
        d_relaxationFlags[4]=1;	
    }
    else if (dftParameters::stressConstraintType=3)// (only for orthoRhombic: relax lattice direction 1, fix other 2)
    {
	d_relaxationFlags[0]=1;
    }
    else if (dftParameters::stressConstraintType=4)// (only for orthoRhombic: relax lattice direction 2, fix other 2)
    {
        d_relaxationFlags[3]=1;
    }
    else if (dftParameters::stressConstraintType=5)// (only for orthoRhombic: relax lattice direction 3, fix other 2)
    {
        d_relaxationFlags[5]=1;	
    }
    else if (dftParameters::stressConstraintType=6)// (only for orthoRhombic: fix lattice direction 1, relax other 2)
    {
        d_relaxationFlags[3]=1;
        d_relaxationFlags[5]=1;	
    }
    else if (dftParameters::stressConstraintType=7)// (only for orthoRhombic: fix lattice direction 2, relax other 2)
    {
	d_relaxationFlags[0]=1;
        d_relaxationFlags[5]=1;	
    }
    else if (dftParameters::stressConstraintType=8)// (only for orthoRhombic: fix lattice direction 3, relax other 2)
    {
	d_relaxationFlags[0]=1;
        d_relaxationFlags[3]=1;
    }
    else if (dftParameters::stressConstraintType=9)// (all stress components relaxed)
    {
	d_relaxationFlags[0]=1;
        d_relaxationFlags[1]=1;
        d_relaxationFlags[2]=1;
	d_relaxationFlags[3]=1;
        d_relaxationFlags[4]=1;
        d_relaxationFlags[5]=1;		
    }
    else
    {
	AssertThrow(false,ExcMessage("The given value for STRESS CONSTRAINT TYPE doesn't match with any available options (1-9)."));		
    }

   pcout<<" --------------Cell stress relaxation flags----------------"<<std::endl;
   pcout<<" [0,0] " <<d_relaxationFlags[0] << ", [0,1] " <<d_relaxationFlags[1] <<
          " [0,2] " <<d_relaxationFlags[2] << ", [1,1] " <<d_relaxationFlags[3] <<
	  ", [1,2] " <<d_relaxationFlags[4] << ", [2,2] " <<d_relaxationFlags[5] <<std::endl;
   pcout<<" --------------------------------------------------"<<std::endl;
}

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::run()
{
   const double tol=dftParameters::stressRelaxTol;// (units: Hatree/Bohr^3)
   const unsigned int  maxIter=100;
   const double lineSearchTol=5e-2;
   const unsigned int maxLineSearchIter=4;
   const unsigned int debugLevel=this_mpi_process==0?1:0;

   d_totalUpdateCalls=0;
   cgPRPNonLinearSolver cgSolver(tol,maxIter,debugLevel,mpi_communicator, lineSearchTol,maxLineSearchIter);

   pcout<<" Starting CG Cell stress relaxation... "<<std::endl;
   pcout<<"   ---CG Parameters--------------  "<<std::endl;
   pcout<<"      stopping tol: "<< tol<<std::endl;
   pcout<<"      maxIter: "<< maxIter<<std::endl;
   pcout<<"      lineSearch tol: "<< lineSearchTol<<std::endl;
   pcout<<"      lineSearch maxIter: "<< maxLineSearchIter<<std::endl;
   pcout<<"   ------------------------------  "<<std::endl;

   if  (getNumberUnknowns()>0)
   {
       nonLinearSolver::ReturnValueType cgReturn=cgSolver.solve(*this);
       
       if (cgReturn == nonLinearSolver::SUCCESS )
       {
	    pcout<< " ...CG cell stress relaxation completed, total number of cell geometry updates: "<<d_totalUpdateCalls<<std::endl;
       }
       else if (cgReturn == nonLinearSolver::MAX_ITER_REACHED)
       {
	    pcout<< " ...Maximum CG iterations reached "<<std::endl;

       }        
       else if(cgReturn == nonLinearSolver::FAILURE)
       {
	    pcout<< " ...CG cell stress relaxation failed "<<std::endl;

       }     

   }
}


template<unsigned int FEOrder>
int geoOptCell<FEOrder>::getNumberUnknowns() const
{
    return std::accumulate(d_relaxationFlags.begin(), d_relaxationFlags.end(), 0 );
}



template<unsigned int FEOrder>
double geoOptCell<FEOrder>::value() const
{
   dftUtils::ExcNotImplementedYet;     
}

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::value(std::vector<double> & functionValue)
{
   dftUtils::ExcNotImplementedYet;     
}

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::gradient(std::vector<double> & gradient)
{
   gradient.clear();
   const Tensor<2,3,double> stress= dftPtr->forcePtr->getStress();
   
   if (d_relaxationFlags[0]==1)
       gradient.push_back(stress[0][0]);
   if (d_relaxationFlags[1]==1)
       gradient.push_back(stress[0][1]);  
   if (d_relaxationFlags[2]==1)
       gradient.push_back(stress[0][2]);  
   if (d_relaxationFlags[3]==1)
       gradient.push_back(stress[1][1]);
   if (d_relaxationFlags[4]==1)
       gradient.push_back(stress[1][2]);  
   if (d_relaxationFlags[5]==1)
       gradient.push_back(stress[2][2]);     

}


template<unsigned int FEOrder>
void geoOptCell<FEOrder>::precondition(std::vector<double>       & s,
			              const std::vector<double> & gradient) const
{
   dftUtils::ExcNotImplementedYet;     
}

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::update(const std::vector<double> & solution)
{


   std::vector<double> bcastSolution(solution.size());
   for (unsigned int i=0; i< solution.size(); ++i)
   {
       bcastSolution[i]=solution[i];
   }

   //for synchronization    
   MPI_Bcast(&(bcastSolution[0]),
	     bcastSolution.size(),
	     MPI_DOUBLE,
	     0,
	     MPI_COMM_WORLD);   

   Tensor<2,3,double> strainEpsilonNew=d_strainEpsilon;

   unsigned int count=0;
   if (d_relaxationFlags[0]==1)
   {
       strainEpsilonNew[0][0]+=bcastSolution[count];
       count++;
   }
   if (d_relaxationFlags[1]==1)
   {
       strainEpsilonNew[0][1]+=bcastSolution[count];
       count++;
   }
   if (d_relaxationFlags[2]==1)
   {
       strainEpsilonNew[0][2]+=bcastSolution[count];
       count++; 
   }
   if (d_relaxationFlags[3]==1)
   {
       strainEpsilonNew[1][1]+=bcastSolution[count];
       count++;
   }
   if (d_relaxationFlags[4]==1)
   {
       strainEpsilonNew[1][2]+=bcastSolution[count];
       count++;
   }
   if (d_relaxationFlags[5]==1)
   {
       strainEpsilonNew[2][2]+=bcastSolution[count];
       count++;
   }

   // To transform the domain under the strain we have to first do a inverse transformation
   // to bring the domain back to the unstrained state.
   const Tensor<2,3,double> deformationGradient=strainEpsilonNew*invert(d_strainEpsilon);

   //deform fem mesh and reinit
   d_totalUpdateCalls+=1;
   dftPtr->deformDomain(deformationGradient);
   
   
   // if ion optimization is on, then for every cell relaxation also relax the atomic forces
   if (dftParameters::isIonOpt)
   {
      dftPtr->geoOptIonPtr->init();
      dftPtr->geoOptIonPtr->run();
   }
   else
   {
      dftPtr->solve();
   }
}

template<unsigned int FEOrder>
void geoOptCell<FEOrder>::solution(std::vector<double> & solution)
{
   dftUtils::ExcNotImplementedYet;     
}

template<unsigned int FEOrder>
std::vector<int>  geoOptCell<FEOrder>::getUnknownCountFlag() const
{
   dftUtils::ExcNotImplementedYet;     
}


template class geoOptCell<1>;
template class geoOptCell<2>;
template class geoOptCell<3>;
template class geoOptCell<4>;
template class geoOptCell<5>;
template class geoOptCell<6>;
template class geoOptCell<7>;
template class geoOptCell<8>;
template class geoOptCell<9>;
template class geoOptCell<10>;
template class geoOptCell<11>;
template class geoOptCell<12>;

#endif
