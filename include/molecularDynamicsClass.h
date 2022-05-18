// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Kartick Ramakrishnan
//

#ifndef molecularDynamicsClass_H_
#define molecularDynamicsClass_H_
#include "constants.h"
#include "headers.h"
#include <vector>
#include "dftBase.h"
#include "dftfeWrapper.h"

namespace dftfe
{
  using namespace dealii;
  class molecularDynamicsClass
  {
  public:
    /**
     * @brief molecularDynamicsClass constructor: copy data from dftparameters to the memebrs of molecularDynamicsClass
     *
     *
     *  @param[in] dftBase *_dftBasePtr pointer to base class of dftClass
     *  @param[in] mpi_comm_parent parent mpi communicator
     */
    molecularDynamicsClass(const MPI_Comm &mpi_comm_parent, bool restart);

    const double haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
    const double haToeV              = 27.211386245988;
    const double bohrToAng           = 0.529177210903;
    const double AngTobohr           = 1.0 / bohrToAng;
    const double kB = 8.617333262e-05; // eV/K **3.166811429e-6**;

    int d_startingTimeStep;
    /**
     * @brief runMD: Assign atom mass to charge. Create vectors for displacement, velocity, force.
     * Create KE vector, TE vector, PE vector. Initialise velocities from
     * Boltsmann distribution. Set Center of Mass velocities to be 0. Call the
     * resepective ensemble based on input file
     *
     *
     */
    void
    runMD();
    // ~molecularDynamicsClass();
    
    void
    set(dftfeWrapper &  dftfeWrapper);

    void
    init(std::string & coordinatesFile, std::string & domainVectorsFile);

  private:
    // pointer to dft class
    dftBase *d_dftPtr;

    // parallel communication objects
    const MPI_Comm     d_mpiCommParent;
    const unsigned int d_this_mpi_process;

    // conditional stream object
    dealii::ConditionalOStream pcout;


    unsigned int                     d_restartFlag;
    unsigned int                     d_numberGlobalCharges;
    double                           d_TimeStep;
    unsigned int                     d_TimeIndex;
    unsigned int                     d_numberofSteps;
    double                           d_startingTemperature;
    int                              d_ThermostatTimeConstant;
    std::string                      d_ThermostatType;
    double                           d_MDstartWallTime;
    double                           d_MaxWallTime;
    std::vector<std::vector<double>> d_atomFractionalunwrapped;
    std::vector<double>              d_domainLength;
    distributedCPUVec<double> d_extrapDensity_tmin2, d_extrapDensity_tmin1,
      d_extrapDensity_t0, d_extrapDensity_tp1;


    /**
     * @brief mdNVE Performs a Ccanonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Temperature is NOT_CONTROLLED. Controls the timeloop.

     * @param[in] atomMass Stores the mass of each Charge.
     * @param[out] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[out] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[out] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[out] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[out] displacements Stores the displacment of each Charge, updated
     at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each
     TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each
     TimeStep
     *
     *
     *
     */
    void
    mdNVE(std::vector<double> &                      KineticEnergyVector,
          std::vector<double> &                      InternalEnergyVector,
          std::vector<double> &                      EntropicEnergyVector,
          std::vector<double> &                      TotalEnergyVector,
          std::vector<dealii::Tensor<1, 3, double>> &displacements,
          std::vector<double> &                      velocity,
          std::vector<double> &                      force,
          const std::vector<double> &                atomMass);
    /**

    @brief mdNVTnosehoverchainsThermostat Performs a Canonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is NOSE_HOVER_CHAINS. Controls the timeloop.
     *
     * @param[in] atomMass Stores the mass of each Charge.
     * @param[out] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[out] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[out] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[out] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[out] displacements Stores the displacment of each Charge, updated
    at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each
    TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each
    TimeStep
     *
     *
     *
     */
    void
    mdNVTnosehoverchainsThermostat(
      std::vector<double> &                      KineticEnergyVector,
      std::vector<double> &                      InternalEnergyVector,
      std::vector<double> &                      EntropicEnergyVector,
      std::vector<double> &                      TotalEnergyVector,
      std::vector<dealii::Tensor<1, 3, double>> &displacements,
      std::vector<double> &                      velocity,
      std::vector<double> &                      force,
      const std::vector<double> &                atomMass);

    /**

    @brief mdNVTrescaleThermostat Performs a Constant Kinetic Energy Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is RESCALE. Controls the timeloop. At timestep which is
    multiple of Thermostat time constatn, the veloctites are rescaled *such that
    the temperature is set to inital temperature .
     *
     * @param[in] atomMass Stores the mass of each Charge.
     * @param[out] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[out] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[out] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[out] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[out] displacements Stores the displacment of each Charge, updated
    at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each
    TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each
    TimeStep
     *
     *
     *
     */
    void
    mdNVTrescaleThermostat(
      std::vector<double> &                      KineticEnergyVector,
      std::vector<double> &                      InternalEnergyVector,
      std::vector<double> &                      EntropicEnergyVector,
      std::vector<double> &                      TotalEnergyVector,
      std::vector<dealii::Tensor<1, 3, double>> &displacements,
      std::vector<double> &                      velocity,
      std::vector<double> &                      force,
      const std::vector<double> &                atomMass);


    /**

    @brief mdNVTsvrThermostat Performs a Canonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is SVR. Controls the timeloop.
     * @param[in] massAtoms Stores the mass of each Charge.     *
     * @param[out] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[out] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[out] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[out] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[out] displacements Stores the displacment of each Charge, updated
    at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each
    TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each
    TimeStep
     *
     */
    void
    mdNVTsvrThermostat(std::vector<double> &KineticEnergyVector,
                       std::vector<double> &InternalEnergyVector,
                       std::vector<double> &EntropicEnergyVector,
                       std::vector<double> &TotalEnergyVector,
                       std::vector<dealii::Tensor<1, 3, double>> &displacements,
                       std::vector<double> &                      velocity,
                       std::vector<double> &                      force,
                       const std::vector<double> &                atomMass);


    /**
    * @brief RescaleVelocities controls the velocity at timestep t. The scaling of
    * velocities depends on ratio of T at that timestep and inital Temperature.


     * @param[in] M Stores the mass of each Charge.
     * @param[in] Temperature  temperature at current Timestep
     * @param[out] v Stores the velocity of each Charge, updated at each
   Timestep
     *
     * @param[return] KE Kinetic Energy at current timestp in eV

   *
     *
     *
     */
    double
    RescaleVelocities(std::vector<double> &      v,
                      const std::vector<double> &M,
                      double                     Temperature);


    /**

    * @brief NoseHoverChains controls the velocity at timestep t. The temperature is contolled by
        2 thermostats. Thermostat 1 contols the velocity of all Charges.
    Thermostat 2 controls thermostat 1. Employs Extended Lagrangian approach.


     * @param[in] Q stores mass of each Thermostat
     * @param[in] Temperature  temperature of previous timestep
     * @param[out] v Stores the velocity of each Charge, updated at each
    TimeStep
     * @param[out] v_e Stores the thermostat velocity
     * @param[out] e Stores the position of each thermosat
     *
     */
    void
    NoseHoverChains(std::vector<double> &v,
                    std::vector<double> &v_e,
                    std::vector<double> &e,
                    std::vector<double>  Q,
                    double               KE,
                    double               Temperature);

    /**

    * @brief

     *
     * @param[in] KEref Target value of Kinetic Enegy from Temperature
     * @param[out] v Stores the velocity of each Charge, updated at each
    TimeStep
     * @param[out] KE rescaled Kinetic Energy from svr thermostat
     *
     *
     */
    double
    svr(std::vector<double> &v, double KE, double KEref);



    /**

    * @brief velocityVerlet


     * @param[in] atomMass Stores the mass of each Charge.

     *
     * @param[return] KE Kinetic Energy at current timestp in eV
     * @param[out] forceonAtoms Updated -ve forces on each charge.
     * @param[out] r Updated displacement
     * @param[out] v Updated velocity of each atom
     *
     *
     *
     */
    double
    velocityVerlet(std::vector<double> &                      v,
                   std::vector<dealii::Tensor<1, 3, double>> &r,
                   const std::vector<double> &                atomMass,
                   std::vector<double> &                      forceOnAtoms);



    /**
    * @brief  writeRestartFile: Writing files at each timestep to mdRestart

     * @param[in] velocity Velocity updated from restart
     * @param[in] force Force data at each timeStep
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] TE  temperature at current Timestep
     * @param[in] time Current TimeStep
     *
     *
     */

    void
    writeRestartFile(const std::vector<dealii::Tensor<1, 3, double>> &disp,
                     const std::vector<double> &                      velocity,
                     const std::vector<double> &                      force,
                     const std::vector<double> &KineticEnergyVector,
                     const std::vector<double> &InternalEnergyVector,
                     const std::vector<double> &TotalEnergyVector,
                     int                        time);

    /**

* @brief  InitialiseFromRestartFile : Initialise atomcordinates, velocity and force at restart

 * @param[out] disp Displacements of previous timestep from restart
 * @param[out] velocity Velocity updated from restart
 * @param[out] force Force updated from dft->Solve
 * @param[out] PE  Free energy of system at current Timestep
 * @param[out] KE  Kinetic ENergy of nuclei at current Timestep
 * @param[out] TE  temperature at current Timestep
 *

 *
 *
 *
 */
    void
      InitialiseFromRestartFile(std::vector<dealii::Tensor<1, 3, double>> &disp,
                                std::vector<double> &velocity,
                                std::vector<double> &force,
                                std::vector<double> &KE,
                                std::vector<double> &IE,
                                std::vector<double> &TE);

    /**

* @brief  NoseHoverExtendedLagrangian Writes the NHC parameters at each timeStep

 * @param[in] thermovelocity Velocity of each, updated at each TimeStep
 * @param[in] thermoposition Position of each thermostat , updated at each
      TimeStep
 * @param[in] thermomass Stores the mass of each thermostat.
 * @param[in] time Current TimeStep
 *
 *
 */
    void
    writeRestartNHCfile(const std::vector<double> &v_e,
                        const std::vector<double> &e,
                        const std::vector<double> &Q,
                        const int                  time);

    /**

* @brief  InitialiseFromRestartNHCFile: Reads the NHC parameters during restart

 * @param[out] thermovelocity Velocity of each, updated at each TimeStep
 * @param[out] thermoposition Position of each thermostat , updated at each
                TimeStep
 * @param[out] thermomass Stores the mass of each thermostat.
 *
 *
 *
 */
    void
    InitialiseFromRestartNHCFile(std::vector<double> &v_e,
                                 std::vector<double> &e,
                                 std::vector<double> &Q);

    /**

* @brief  writeTotalDisplacementFile: Updates Displacement.chk and appends TotalDisplacement.chk

 * @param[in] r Displacemnt of each atom, updated at each TimeStep
 * @param[in] time  each TimeStep

 *
 *
 *
 */
    void
    writeTotalDisplacementFile(
      const std::vector<dealii::Tensor<1, 3, double>> &r,
      int                                              time);

    /**

    * @brief  NoseHoverExtendedLagrangian: Computes the Nose-Hover Hamiltonian when using NHC thermostat

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each
    TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     *
     * @return Hnose Nose Hamiltonian at each timestep
     *
     *
     *
     */

    double
    NoseHoverExtendedLagrangian(const std::vector<double> &thermovelocity,
                                const std::vector<double> &thermoposition,
                                const std::vector<double> &thermomass,
                                double                     PE,
                                double                     KE,
                                double                     T);
    /**
     * @brief  checkRestart: Identifies the folder containing the restart file, sets the path of coordinates file and restursn the starting timestep    *
     * @return StartingTimeStep the timestep to restart the MD from.
     *
     *
     *
     */
    int
    checkRestart(std::string & coordinatesFile, std::string domainVectorsFile );

    /**
     * @brief  DensityExtrapolation Identifies the folder containing the restart file, sets the path of coordinates file and restursn the starting timestep    *
     *
     *
     *
     *
     */
    void
    DensityExtrapolation(int TimeStep);

    /**
     * @brief  DensityExtrapolation Identifies the folder containing the restart file, sets the path of coordinates file and restursn the starting timestep    *
     *
     *
     *
     *
     */
    void
    DensitySplitExtrapolation(int TimeStep);
  };
} // namespace dftfe
#endif
