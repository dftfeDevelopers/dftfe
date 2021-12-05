#ifndef molecularDynamicsClass_H_
#define molecularDynamicsClass_H_
#include "constants.h"
#include "headers.h"
#include <vector>
#include "dft.h"

namespace dftfe
{
    using namespace dealii;
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    class dftClass;
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    class molecularDynamicsClass
    {
    public:
    /**
     * @brief molecularDynamicsClass constructor: copy data from dftparameters to the memebrs of molecularDynamicsClass
     * 
     *
     *  @param[in] dftClass<FEOrder, FEOrderElectro> *_dftPtr dftclass pointer
     * used to access friend class dft.cc
     *  @param[in] mpi_comm_replica  mpi_communicator for domain decomposition
     * parallelization
    
     */
        molecularDynamicsClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
                      const MPI_Comm &                   mpi_comm_replica);

        const double haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
        const double haToeV              = 27.211386245988;
        const double bohrToAng           = 0.529177210903;
        const double AngTobohr           = 1.0 / bohrToAng;
        const double kB                  = 8.617333262e-05; // eV/K **3.166811429e-6**;
        
         int d_startingTimeStep;
    /**
     * @brief runMD: Assign atom mass to charge. Create vectors for displacement, velocity, force. 
     * Create KE vector, TE vector, PE vector. Initialise velocities from Boltsmann distribution. 
     * Set Center of Mass velocities to be 0. Call the resepective ensemble based on input file
     * 
     *
     * @param[out] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[out] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[out] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[out] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] massAtoms Stores the mass of each Charge.
     *  
     *     
     */
        void runMD();
       // ~molecularDynamicsClass();

    private:
    // pointer to dft class
        dftClass<FEOrder, FEOrderElectro> *dftPtr;

    // parallel communication objects
        const MPI_Comm     d_mpi_communicator;
        const unsigned int d_this_mpi_process;

    // conditional stream object
        dealii::ConditionalOStream pcout;
    
    
        unsigned int d_restartFlag;
        unsigned int d_numberGlobalCharges;   
        double d_TimeStep;
        unsigned int d_TimeIndex;
        unsigned int d_numberofSteps;
        double d_startingTemperature;
        int d_ThermostatTimeConstant;
        std::string d_ThermostatType;
        double d_MDstartWallTime;
        double d_MaxWallTime;
    /**
     * @brief mdNVE Performs a Ccanonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Temperature is NOT_CONTROLLED. Controls the timeloop.
     * 
     *
     * @param[in] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[in] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[in] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[in] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[in] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] atomMass Stores the mass of each Charge.

     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] atomMass Stores the mass of each Charge.     * 
     *  
     *     
     */    
        void mdNVE(std::vector<double> &KineticEnergyVector , std::vector<double>  &InternalEnergyVector,
                std::vector<double> &EntropicEnergyVector, std::vector<double> &TotalEnergyVector ,
                std::vector<dealii::Tensor<1, 3, double>> &displacements ,std::vector<double> &velocity ,
                std::vector<double> &force, std::vector<double> atomMass );
    /**

    @brief mdNVTnosehoverchainsThermostat Performs a Canonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is NOSE_HOVER_CHAINS. Controls the timeloop. 
     *
     * @param[in] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[in] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[in] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[in] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[in] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] massAtoms Stores the mass of each Charge.

     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] massAtoms Stores the mass of each Charge.     * 
     *  
     *     
     */               
        void mdNVTnosehoverchainsThermostat(std::vector<double> &KineticEnergyVector ,
                                            std::vector<double> &InternalEnergyVector,
                                            std::vector<double> &EntropicEnergyVector , 
                                            std::vector<double> &TotalEnergyVector ,
                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                            std::vector<double> &velocity ,
                                            std::vector<double>  &force , 
                                                std::vector<double> atomMass );

    /**

    @brief mdNVTrescaleThermostat Performs a Constant Kinetic Energy Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is RESCALE. Controls the timeloop. At timestep which is multiple of Thermostat time constatn, the veloctites are rescaled
     *such that the temperature is set to inital temperature .      
     *
     * @param[in] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[in] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[in] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[in] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[in] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] massAtoms Stores the mass of each Charge.

     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] massAtoms Stores the mass of each Charge.     * 
     *  
     *     
     */ 
        void mdNVTrescaleThermostat(std::vector<double> &KineticEnergyVector ,
                                            std::vector<double> &InternalEnergyVector,
                                            std::vector<double> &EntropicEnergyVector , 
                                            std::vector<double> &TotalEnergyVector ,
                                            std::vector<dealii::Tensor<1, 3, double>>  &displacements ,
                                            std::vector<double> &velocity ,
                                            std::vector<double> &force , 
                                                std::vector<double> atomMass );


    /**

    @brief mdNVTsvrThermostat Performs a Canonical Ensemble MD calculation. The inital temperature is set by runMD().
     * Thermostat type is SVR. Controls the timeloop.      
     *
     * @param[in] KineticEnergyVector Stores KineticEnergy at each TimeStep
     * @param[in] InternalEnergyVector Stores InternalEnergy at each TimeStep
     * @param[in] EntropicEnergyVector Stores PotentialEnergy at each TimeStep
     * @param[in] TotalEnergyVector Stores TotalEnergy at each TimeStep
     * @param[in] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] massAtoms Stores the mass of each Charge.

     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] massAtoms Stores the mass of each Charge.     * 
     *  
     *     
     */ 
        void mdNVTsvrThermostat(std::vector<double> &KineticEnergyVector ,
                                            std::vector<double>  &InternalEnergyVector,
                                            std::vector<double> &EntropicEnergyVector , 
                                            std::vector<double> &TotalEnergyVector ,
                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                            std::vector<double> &velocity ,
                                            std::vector<double> &force , 
                                                std::vector<double> atomMass  );


    /**
    * @brief RescaleVelocities controls the velocity at timestep t. The scaling of     
    * velocities depends on ratio of T at that timestep and inital Temperature.

     * @param[in] v Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] KE Kinetic Energy at current timestp in eV
     * @param[in] M Stores the mass of each Charge.
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] KE Kinetic Energy at current timestp in eV

   * 
     *  
     *     
     */ 
        void RescaleVelocities(std::vector<double> &v ,double &KE ,
                         std::vector<double> M , double Temperature ); 

        
    /**

    * @brief NoseHoverChains controls the velocity at timestep t. The temperature is contolled by
        2 thermostats. Thermostat 1 contols the velocity of all Charges. Thermostat 2 controls thermostat 1.
        Employs Extended Lagrangian approach.

     * @param[in] v Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] v_e Stores the thermostat velocity 
     * @param[in] e Stores the position of each thermosat
     * @param[in] Q stores mass of each Thermostat
     * @param[in] Temperature  temperature of previous timestep   
     *  
     *     
     */                   
        void NoseHoverChains(std::vector<double> &v , std::vector<double> &v_e ,
         std::vector<double> &e , std::vector<double> Q , double KE , double Temperature) ; 

    /**

    * @brief 

     * @param[in] v Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] KEref Target value of Kinetic Enegy from Temperature
     * @param[out] KE rescaled Kinetic Energy from svr thermostat   
     *  
     *     
     */                   
        void svr(std::vector<double> &v , double &KE, double KEref ) ; 



    /**

    * @brief velocityVerlet

     * @param[in] v Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] forceOnAtoms Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] atomMass Stores the mass of each Charge.
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] KE Kinetic Energy at current timestp in eV
     * @param[out] forceonAtoms Updated -ve forces on each charge.
     * @param[out] r Updated displacement
     * @param[out] v Updated velocity of each atom
     *    
     *  
     *     
     */ 
        void velocityVerlet(std::vector<double> &v, std::vector<dealii::Tensor<1, 3, double>> &r, 
                        std::vector<double> atomMass , double &KE , std::vector<double> &forceOnAtoms );

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */  
       
       void writeRestartFile(std::vector<double> velocity , std::vector<double> force  , std::vector<double> KineticEnergyVector,
                              std::vector<double> InternalEnergyVector, std::vector<double> TotalEnergyVector, int time );   

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */  
        void InitialiseFromRestartFile( std::vector<double> &velocity, std::vector<double> &force , std::vector<double> &KE , std::vector<double> &IE , std::vector<double> &TE  );   

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */  
       void writeRestartNHCfile(std::vector<double> v_e, std::vector<double> e , std::vector<double> Q, int time  );   

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */  
        void InitialiseFromRestartNHCFile( std::vector<double> &v_e, std::vector<double> &e , std::vector<double> &Q   );  

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */  
        void writeTotalDisplacementFile(std::vector<dealii::Tensor<1, 3, double>> r , int time ) ;  

        /**

    * @brief  NoseHoverExtendedLagrangian

     * @param[in] thermovelocity Velocity of each, updated at each TimeStep
     * @param[in] thermoposition Position of each thermostat , updated at each TimeStep
     * @param[in] thermomass Stores the mass of each thermostat.
     * @param[in] PE  Free energy of system at current Timestep
     * @param[in] KE  Kinetic ENergy of nuclei at current Timestep
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] Hnose Nose Hamiltonian at each timestep
     *    
     *  
     *     
     */    
        
        double NoseHoverExtendedLagrangian(std::vector<double> thermovelocity  , std::vector<double> thermoposition  , std::vector<double> thermomass , double PE, double KE, double  T );                                    
        


    
    };
}
#endif