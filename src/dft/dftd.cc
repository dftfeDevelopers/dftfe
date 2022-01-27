template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::computeDispersionCorrection()
{
  double energy;
  std::vector<double> gradient(natoms*3);
  std::vector<double> sigma(9);
  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    int natoms = atomLocations.size();
    int atomicnumbers[natoms];
    double positions[natoms*3];
    double lattice[9];
    bool periodic[3];
    std::string functional;
    bool customParameters = !(dftParameters::dampingParameterFilename=="");
    std::vector<std::vector<double>> parameterList;
    for (unsigned int i = 0; i < natoms ; ++i )
    {
      atomicnumbers[i] = atomLocations[i][0];
    }

    for (unsigned int irow = 0 ; irow < natoms; ++irow)
    {
      for (unsigned int icol = 0 ; icol < 3 ; ++icol)
      {
        positions[irow*3+icol] = atomLocations[irow][2+icol];
      }
    }
    for (unsigned int irow = 0 ; irow < 3; ++irow)
    {
      for (unsigned int icol = 0 ; icol < 3 ; ++icol)
      {
        lattice[irow*3+icol] = d_domainBoundingVectors[irow][icol];
      }
    }
    periodic[0]=dftParameters::periodicX;
    periodic[1]=dftParameters::periodicY;
    periodic[2]=dftParameters::periodicZ;

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
            if (dftParameters::dispersioncorrectiontype==1) AssertThrow (dftParameters::d3dampingtype!=4, dealii::ExcMessage(std::string ("The OP damping functions has not been parametrized for this functional")));
            functional = "pbe";
            break;
            case 5:
            if (dftParameters::dispersioncorrectiontype==1) AssertThrow (dftParameters::d3dampingtype==0 || dftParameters::d3dampingtype==1, dealii::ExcMessage(std::string ("The OP, BJM and ZEROM damping functions have not been parametrized for this functional")));
            functional = "rpbe";
            break;
            default:
            break;
      }
    }
    switch(dftParameters::dispersioncorrectiontype)
    {
      case 1:
      {
        dftd3_error error = dftd3_new_error();
        dftd3_structure mol = NULL;
        dftd3_model disp = NULL;
        dftd3_param param = NULL;
        
        mol = dftd3_new_structure(error, natoms, atomicnumbers, positions, lattice, periodic);
        AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));
        
        disp = dftd3_new_d3_model(error, mol);
        AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

        dftd3_set_model_realspace_cutoff(error, disp, dftParameters::d3cutoff2, dftParameters::d3cutoff3, dftParameters::d3cutoffCN); 
        AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

        switch (dftParameters::d3dampingtype)
        {
          case 0:
          if (!customParameters) 
          {
            param = dftd3_load_zero_damping(error, &functional[0], dftParameters::d3ATM);
          }
          else 
          {
            dftUtils::readFile(6, parameterList, dftParameters::dampingParameterFilename);
            param = dftd3_new_zero_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
          }
          break;
          case 1:
          if (!customParameters) 
          {
            param = dftd3_load_rational_damping(error, &functional[0], dftParameters::d3ATM);
          }
          else 
          {
            dftUtils::readFile(6, parameterList, dftParameters::dampingParameterFilename);
            param = dftd3_new_rational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
          }
          break;
          case 2:
          if (!customParameters) 
          {
            param = dftd3_load_mzero_damping(error, &functional[0], dftParameters::d3ATM);
          }
          else 
          {
            dftUtils::readFile(7, parameterList, dftParameters::dampingParameterFilename);
            param = dftd3_new_mzero_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5], parameterList[0][6]);
          }
          break;
          case 3:
          if (!customParameters) 
          {
            param = dftd3_load_mrational_damping(error, &functional[0], dftParameters::d3ATM);
          }
          else 
          {
            dftUtils::readFile(6, parameterList, dftParameters::dampingParameterFilename);
            param = dftd3_new_mrational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
          }
          break;
          case 4:
          if (!customParameters) 
          {
            param = dftd3_load_optimizedpower_damping(error, &functional[0], dftParameters::d3ATM);
          }
          else 
          {
            dftUtils::readFile(7, parameterList, dftParameters::dampingParameterFilename);
            param = dftd3_new_optimizedpower_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5], parameterList[0][6]);
          }
          break;
          default:
          break;
        }
        AssertThrow(dftd3_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

        dftd3_get_dispersion(error, mol, disp, param, &energy, &gradient[0], &sigma[0]);
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
        
        mol = dftd4_new_structure(error, natoms, atomicnumbers, positions, NULL, lattice, periodic);
        AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));
        
        disp = dftd4_new_d4_model(error, mol);
        AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

        if (!customParameters) 
        {
          param = dftd4_load_rational_damping(error, &functional[0], dftParameters::d4MBD);
        }
        else 
        {
          dftUtils::readFile(6, parameterList, dftParameters::dampingParameterFilename);
          param = dftd4_new_rational_damping(error, parameterList[0][0], parameterList[0][1], parameterList[0][2], parameterList[0][3], parameterList[0][4], parameterList[0][5]);
        }

        
        AssertThrow(dftd4_check_error(error)==0,dealii::ExcMessage(std::string ("Failure in DFTD Module ")));

        dftd4_get_dispersion(error, mol, disp, param, &energy, &gradient[0], &sigma[0]);
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
  }
  energy = Utilities::MPI::broadcast(mpi_communicator, energy, 0);
  gradient = Utilities::MPI::broadcast(mpi_communicator, gradient, 0);
  sigma = Utilities::MPI::broadcast(mpi_communicator, sigma, 0);

  forcePtr->d_forceDispersion=gradient;
  for (unsigned int irow=0; irow < 3; ++irow)
  {
    for (unsigned int icol=0; icol < 3; ++icol)
    {
      forcePtr->d_stressDispersion[irow][icol] = sigma[irow * 3 + icol];
    }
  }

  return energy;
}
