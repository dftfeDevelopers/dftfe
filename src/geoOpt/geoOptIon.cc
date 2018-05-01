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

#include <geoOptIon.h>
#include <cgPRPNonLinearSolver.h>
#include <force.h>
#include <dft.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>

namespace dftfe {

//
//constructor
//
template<unsigned int FEOrder>
geoOptIon<FEOrder>::geoOptIon(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  mpi_communicator (mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//
template<unsigned int FEOrder>
void geoOptIon<FEOrder>::init()
{
   const int numberGlobalAtoms=dftPtr->atomLocations.size();
   std::vector<std::vector<int> > tempRelaxFlagsData;
   dftUtils::readRelaxationFlagsFile(3,tempRelaxFlagsData,dftParameters::ionRelaxFlagsFile);
   AssertThrow(tempRelaxFlagsData.size()==numberGlobalAtoms,ExcMessage("Incorrect number of entries in relaxationFlags file"));
   d_relaxationFlags.clear();
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
       for (unsigned int j=0; j< 3; ++j)
          d_relaxationFlags.push_back(tempRelaxFlagsData[i][j]);
   }
   //print relaxation flags
   pcout<<" --------------Ion force relaxation flags----------------"<<std::endl;
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
       pcout<<tempRelaxFlagsData[i][0] << "  "<< tempRelaxFlagsData[i][1] << "  "<<tempRelaxFlagsData[i][2]<<std::endl;
   }
   pcout<<" --------------------------------------------------"<<std::endl;
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::run()
{
   const double tol=dftParameters::forceRelaxTol;//(units: Hatree/Bohr)
   const unsigned int  maxIter=100;
   const double lineSearchTol=tol*2.0;
   const double lineSearchDampingParameter=0.7;
   const unsigned int maxLineSearchIter=4;
   const unsigned int debugLevel=Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==0?1:0;
   const unsigned int maxRestarts=2;
   unsigned int restartCount=0;

   d_totalUpdateCalls=0;
   cgPRPNonLinearSolver cgSolver(tol,
	                        maxIter,
				debugLevel,
				mpi_communicator,
				lineSearchTol,
				maxLineSearchIter,
				lineSearchDampingParameter);

   pcout<<" Starting CG Ion force relaxation... "<<std::endl;
   pcout<<"   ---CG Parameters--------------  "<<std::endl;
   pcout<<"      stopping tol: "<< tol<<std::endl;
   pcout<<"      maxIter: "<< maxIter<<std::endl;
   pcout<<"      lineSearch tol: "<< lineSearchTol<<std::endl;
   pcout<<"      lineSearch maxIter: "<< maxLineSearchIter<<std::endl;
   pcout<<"      lineSearch damping parameter: "<< lineSearchDampingParameter<<std::endl;
   pcout<<"   ------------------------------  "<<std::endl;

   if  (getNumberUnknowns()>0)
   {
       nonLinearSolver::ReturnValueType cgReturn=cgSolver.solve(*this);
       if (cgReturn == nonLinearSolver::RESTART && restartCount<maxRestarts )
       {
	   pcout<< " ...Restarting CG, restartCount: "<<restartCount<<std::endl;
	   cgReturn=cgSolver.solve(*this);
	   restartCount++;
       }

       if (cgReturn == nonLinearSolver::SUCCESS )
       {
	    pcout<< " ...CG Ion force relaxation completed as maximum force magnitude is less than force relaxation tolerance: "<< dftParameters::forceRelaxTol<<", total number of ion position updates: "<<d_totalUpdateCalls<<std::endl;
       }
       else if (cgReturn == nonLinearSolver::FAILURE)
       {
	    pcout<< " ...CG Ion force relaxation failed "<<std::endl;

       }
       else if (cgReturn == nonLinearSolver::MAX_ITER_REACHED)
       {
	    pcout<< " ...Maximum CG iterations reached "<<std::endl;

       }
       else if (cgReturn == nonLinearSolver::RESTART)
       {
	    pcout<< " ...Maximum restarts reached "<<std::endl;

       }

   }
}


template<unsigned int FEOrder>
unsigned int geoOptIon<FEOrder>::getNumberUnknowns() const
{
   unsigned int count=0;
   for (unsigned int i=0; i< d_relaxationFlags.size(); ++i)
   {
      count+=d_relaxationFlags[i];
   }
   return count;
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::value(std::vector<double> & functionValue)
{
   AssertThrow(false,dftUtils::ExcNotImplementedYet());
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::gradient(std::vector<double> & gradient)
{
   gradient.clear();
   const int numberGlobalAtoms=dftPtr->atomLocations.size();
   const std::vector<double> tempGradient= dftPtr->forcePtr->getAtomsForces();
   AssertThrow(tempGradient.size()==numberGlobalAtoms*3,ExcMessage("Atom forces have wrong size"));
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
      for (unsigned int j=0; j< 3; ++j)
      {
          if (d_relaxationFlags[3*i+j]==1)
	  {
              gradient.push_back(tempGradient[3*i+j]);
	  }
      }
   }

  d_maximumAtomForceToBeRelaxed=-1.0;

   for (unsigned int i=0;i<gradient.size();++i)
   {
       const double temp=std::sqrt(gradient[i]*gradient[i]);
       if (temp>d_maximumAtomForceToBeRelaxed)
	  d_maximumAtomForceToBeRelaxed=temp;
   }

}


template<unsigned int FEOrder>
void geoOptIon<FEOrder>::precondition(std::vector<double>       & s,
			              const std::vector<double> & gradient) const
{
   AssertThrow(false,dftUtils::ExcNotImplementedYet());
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::update(const std::vector<double> & solution)
{
   const unsigned int numberGlobalAtoms=dftPtr->atomLocations.size();
   std::vector<Point<3> > globalAtomsDisplacements(numberGlobalAtoms);
   int count=0;
   pcout<<" ----CG atom displacement update-----" << std::endl;
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
      for (unsigned int j=0; j< 3; ++j)
      {
	  globalAtomsDisplacements[i][j]=0.0;
	  if (this_mpi_process==0)
	  {
            if (d_relaxationFlags[3*i+j]==1)
	    {
               globalAtomsDisplacements[i][j]=solution[count];
	       count++;
	    }
	  }
      }

      MPI_Bcast(&(globalAtomsDisplacements[i][0]),
	        3,
	        MPI_DOUBLE,
	        0,
	        MPI_COMM_WORLD);
   }

   pcout<<" -----------------------------" << std::endl;
   pcout<< "  Maximum force to be relaxed: "<<  d_maximumAtomForceToBeRelaxed <<std::endl;
   dftPtr->updateAtomPositionsAndMoveMesh(globalAtomsDisplacements);
   d_totalUpdateCalls+=1;

   dftPtr->solve();

}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::solution(std::vector<double> & solution)
{
   AssertThrow(false,dftUtils::ExcNotImplementedYet());
}

template<unsigned int FEOrder>
std::vector<unsigned int>  geoOptIon<FEOrder>::getUnknownCountFlag() const
{
   AssertThrow(false,dftUtils::ExcNotImplementedYet());
}

template class geoOptIon<1>;
template class geoOptIon<2>;
template class geoOptIon<3>;
template class geoOptIon<4>;
template class geoOptIon<5>;
template class geoOptIon<6>;
template class geoOptIon<7>;
template class geoOptIon<8>;
template class geoOptIon<9>;
template class geoOptIon<10>;
template class geoOptIon<11>;
template class geoOptIon<12>;

}
