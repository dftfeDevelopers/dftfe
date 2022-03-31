// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
// authors.
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

#include <dftParameters.h>
#include <fileReaders.h>
#include <dftd.h>


namespace dftfe
{
    //initialize the variables needed for dispersion correction calculations
    
    void
    dispersionCorrection::initDispersionCorrection(
      const std::vector<std::vector<double>>  &atomLocations,
      const std::vector<std::vector<double>>  &d_domainBoundingVectors
    )
    {
      d_natoms=atomLocations.size();
      d_energyDispersion=0.0;
      d_forceDispersion.resize(3*d_natoms);
      d_atomCoordinates.resize(3*d_natoms);
      d_atomicNumbers.resize(d_natoms);

      std::fill(d_forceDispersion.begin(), d_forceDispersion.end(), 0.0);
      std::fill(d_stressDispersion.begin(), d_stressDispersion.end(), 0.0);
      for (unsigned int i = 0; i < d_natoms ; ++i )
      {
        d_atomicNumbers[i] = atomLocations[i][0];
      }

      for (unsigned int irow = 0 ; irow < d_natoms; ++irow)
      {
        for (unsigned int icol = 0 ; icol < 3 ; ++icol)
        {
          d_atomCoordinates[irow*3+icol] = atomLocations[irow][2+icol];
        }
      }
      for (unsigned int irow = 0 ; irow < 3; ++irow)
      {
        for (unsigned int icol = 0 ; icol < 3 ; ++icol)
        {
          d_latticeVectors[irow*3+icol] = d_domainBoundingVectors[irow][icol];
        }
      }
    }


    // Compute D3/4 correction
    void
    dispersionCorrection::computeDFTDCorrection()
    {
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && dealii::Utilities::MPI::this_mpi_process(interpoolcomm) == 0 && dealii::Utilities::MPI::this_mpi_process(interBandGroupComm) ==0)
      {

        bool periodic[3];
        periodic[0]=dftParameters::periodicX;
        periodic[1]=dftParameters::periodicY;
        periodic[2]=dftParameters::periodicZ;
        std::string functional;
        bool customParameters = !(dftParameters::dc_dampingParameterFilename=="");
        std::vector<std::vector<double>> parameterList;
        if(!customParameters)
        {
          switch(dftParameters::xc_id)
          {
                case 1:
                case 2:
                case 3:
                AssertThrow (false, dealii::ExcMessage(std::string ("DFTD3/4 have not been parametrized for this functional")));
                break;
                case 4:
                if (dftParameters::dc_dispersioncorrectiontype==1) AssertThrow (dftParameters::dc_d3dampingtype!=4, dealii::ExcMessage(std::string ("The OP damping functions has not been parametrized for this functional")));
                functional = "pbe";
                break;
                case 5:
                if (dftParameters::dc_dispersioncorrectiontype==1) AssertThrow (dftParameters::dc_d3dampingtype==0 || dftParameters::dc_d3dampingtype==1, dealii::ExcMessage(std::string ("The OP, BJM and ZEROM damping functions have not been parametrized for this functional")));
                functional = "rpbe";
                break;
                default:
                break;
          }
        }
        switch(dftParameters::dc_dispersioncorrectiontype)
        {
          case 1:
          {
            dftd3_error error = dftd3_new_error();
            dftd3_structure mol = NULL;
            dftd3_model disp = NULL;
            dftd3_param param = NULL;
            
            mol = dftd3_new_structure(error, d_natoms, d_atomicNumbers.data(), d_atomCoordinates.data(), d_latticeVectors.data(), periodic);
            AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));
            
            disp = dftd3_new_d3_model(error, mol);
            AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            dftd3_set_model_realspace_cutoff(error, disp, dftParameters::dc_d3cutoff2, dftParameters::dc_d3cutoff3, dftParameters::dc_d3cutoffCN); 
            AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            switch (dftParameters::dc_d3dampingtype)
            {
              case 0:
              if (!customParameters) 
              {
                param = dftd3_load_zero_damping(error, &functional[0], dftParameters::dc_d3ATM);
              }
              else 
              {
                dftUtils::readFile(6, parameterList, dftParameters::dc_dampingParameterFilename);
                param = dftd3_new_zero_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
              }
              break;
              case 1:
              if (!customParameters) 
              {
                param = dftd3_load_rational_damping(error, &functional[0], dftParameters::dc_d3ATM);
              }
              else 
              {
                dftUtils::readFile(6, parameterList, dftParameters::dc_dampingParameterFilename);
                param = dftd3_new_rational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
              }
              break;
              case 2:
              if (!customParameters) 
              {
                param = dftd3_load_mzero_damping(error, &functional[0], dftParameters::dc_d3ATM);
              }
              else 
              {
                dftUtils::readFile(7, parameterList, dftParameters::dc_dampingParameterFilename);
                param = dftd3_new_mzero_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5], parameterList[0][6]);
              }
              break;
              case 3:
              if (!customParameters) 
              {
                param = dftd3_load_mrational_damping(error, &functional[0], dftParameters::dc_d3ATM);
              }
              else 
              {
                dftUtils::readFile(6, parameterList, dftParameters::dc_dampingParameterFilename);
                param = dftd3_new_mrational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
              }
              break;
              case 4:
              if (!customParameters) 
              {
                param = dftd3_load_optimizedpower_damping(error, &functional[0], dftParameters::dc_d3ATM);
              }
              else 
              {
                dftUtils::readFile(7, parameterList, dftParameters::dc_dampingParameterFilename);
                param = dftd3_new_optimizedpower_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5], parameterList[0][6]);
              }
              break;
              default:
              break;
            }
            AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            dftd3_get_dispersion(error, mol, disp, param, &d_energyDispersion, d_forceDispersion.data(), d_stressDispersion.data());
            AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            dftd3_delete_error(&error);
            dftd3_delete_structure(&mol);
            dftd3_delete_model(&disp);
            dftd3_delete_param(&param);
            break;
          }
          case 2:
          {
            dftd4_error error = dftd4_new_error();
            dftd4_structure mol = NULL;
            dftd4_model disp = NULL;
            dftd4_param param = NULL;
            
            mol = dftd4_new_structure(error, d_natoms, d_atomicNumbers.data(), d_atomCoordinates.data(), NULL, d_latticeVectors.data(), periodic);
            AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));
            
            disp = dftd4_new_d4_model(error, mol);
            AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            if (!customParameters) 
            {
              param = dftd4_load_rational_damping(error, &functional[0], dftParameters::dc_d4MBD);
            }
            else 
            {
              dftUtils::readFile(6, parameterList, dftParameters::dc_dampingParameterFilename);
              param = dftd4_new_rational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
            }

            
            AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            dftd4_get_dispersion(error, mol, disp, param, &d_energyDispersion, d_forceDispersion.data(), d_stressDispersion.data());
            AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

            dftd4_delete_error(&error);
            dftd4_delete_structure(&mol);
            dftd4_delete_model(&disp);
            dftd4_delete_param(&param);
            break;
          }
          default:
          break;
        }
        for (unsigned int irow = 0 ; irow < d_natoms; ++irow)
        {
          for (unsigned int icol = 0 ; icol < 3 ; ++icol)
          {
            d_forceDispersion[irow*3+icol] = std::trunc(d_forceDispersion[irow*3+icol]*1e12)*1e-12;
          }
        }
        for (unsigned int irow = 0 ; irow < 3; ++irow)
        {
          for (unsigned int icol = 0 ; icol < 3 ; ++icol)
          {
            d_stressDispersion[irow*3+icol] = std::trunc(d_stressDispersion[irow*3+icol]*1e12)*1e-12 ;
          }
        }
      }
      d_energyDispersion = dealii::Utilities::MPI::broadcast(mpi_communicator, d_energyDispersion, 0);
      d_energyDispersion = dealii::Utilities::MPI::broadcast(interBandGroupComm, d_energyDispersion, 0);
      d_energyDispersion = dealii::Utilities::MPI::broadcast(interpoolcomm, d_energyDispersion, 0);

      d_forceDispersion = dealii::Utilities::MPI::broadcast(mpi_communicator, d_forceDispersion, 0);
      d_forceDispersion = dealii::Utilities::MPI::broadcast(interBandGroupComm, d_forceDispersion, 0);
      d_forceDispersion = dealii::Utilities::MPI::broadcast(interpoolcomm, d_forceDispersion, 0);

      d_stressDispersion = dealii::Utilities::MPI::broadcast(mpi_communicator, d_stressDispersion, 0);
      d_stressDispersion = dealii::Utilities::MPI::broadcast(interBandGroupComm, d_stressDispersion, 0);
      d_stressDispersion = dealii::Utilities::MPI::broadcast(interpoolcomm, d_stressDispersion, 0);

    }


  //Compute TS correction, placeholder



  dispersionCorrection::dispersionCorrection(const MPI_Comm &mpi_comm,
                                    const MPI_Comm &interpool_comm,
                                    const MPI_Comm &interbandgroup_comm)
    : mpi_communicator(mpi_comm)
    , interpoolcomm(interpool_comm)
    , interBandGroupComm(interbandgroup_comm)
  {}


  /**
    * Wrapper function for various dispersion corrections to energy, force and stress.
    *
    * @param atomLocations 
    * @param d_latticeVectors 
    */
  void
  dispersionCorrection::computeDispresionCorrection(
    const std::vector<std::vector<double>>  &atomLocations,
    const std::vector<std::vector<double>>  &d_latticeVectors
  ) 
  {
      initDispersionCorrection(atomLocations,d_latticeVectors);

      if (dftParameters::dc_dispersioncorrectiontype==1||dftParameters::dc_dispersioncorrectiontype==2)
        computeDFTDCorrection();
  }

  double
  dispersionCorrection::getEnergyCorrection() const
  {
    return d_energyDispersion;
  }

  double
  dispersionCorrection::getForceCorrection(int atomNo, int dim) const
  {
    return d_forceDispersion[atomNo * 3 + dim];
  }

  double
  dispersionCorrection::getStressCorrection(int dim1, int dim2) const
  {
    return d_stressDispersion[dim1*3+dim2];
  }

} // namespace dftfe
