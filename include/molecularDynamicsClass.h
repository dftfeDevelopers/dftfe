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
        
        unsigned int startingTimeStep;
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
        const MPI_Comm     mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;

    // conditional stream object
        dealii::ConditionalOStream pcout;
    
    
        unsigned int restartFlag;
        bool velocityFlag;
        unsigned int numberGlobalCharges;
        unsigned int numberofAtomTypes;    
        double timeStep;
        unsigned int TimeIndex;
        unsigned int numberofSteps;
        double startingTemperature;
        int thermostatTimeConstant;
        std::string thermostatType;
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
     * @param[in] massAtoms Stores the mass of each Charge.

     * @param[out] displacements Stores the displacment of each Charge, updated at each TimeStep
     * @param[out] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[out] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[out] massAtoms Stores the mass of each Charge.     * 
     *  
     *     
     */    
        void mdNVE(std::vector<double> & , std::vector<double> & ,
                std::vector<double> &, std::vector<double> & ,
                std::vector<dealii::Tensor<1, 3, double>> & ,std::vector<double> & ,
                std::vector<double> &, std::vector<double> );
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
        void mdNVTnosehoverchainsThermostat(std::vector<double> & ,
                                            std::vector<double> &,
                                            std::vector<double> & , 
                                            std::vector<double> & ,
                                            std::vector<dealii::Tensor<1, 3, double>> & ,
                                            std::vector<double> & ,
                                            std::vector<double> & , 
                                                std::vector<double>  );

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
        void mdNVTrescaleThermostat(std::vector<double> & ,
                                            std::vector<double> &,
                                            std::vector<double> & , 
                                            std::vector<double> & ,
                                            std::vector<dealii::Tensor<1, 3, double>> & ,
                                            std::vector<double> & ,
                                            std::vector<double> & , 
                                                std::vector<double>  );


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
        void mdNVTsvrThermostat(std::vector<double> & ,
                                            std::vector<double> &,
                                            std::vector<double> & , 
                                            std::vector<double> & ,
                                            std::vector<dealii::Tensor<1, 3, double>> & ,
                                            std::vector<double> & ,
                                            std::vector<double> & , 
                                                std::vector<double>  );


    /**
    * @brief RescaleVelocities controls the velocity at timestep t. The scaling of     
    * velocities depends on ratio of T at that timestep and inital Temperature.

     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] massAtoms Stores the mass of each Charge.
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] KE Kinetic Energy at current timestp in eV

   * 
     *  
     *     
     */ 
        void RescaleVelocities(std::vector<double> & ,double & ,
                         std::vector<double> , double ); 

        
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
        void NoseHoverChains(std::vector<double> & , std::vector<double> & ,
         std::vector<double> & , std::vector<double> , double , double  ) ; 

    /**

    * @brief 

     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] KinetricEnergyreference Target value of Kinetic Enegy from Temperature
     * @param[out] KineticEnergy rescaled Kinetic Energy from svr thermostat   
     *  
     *     
     */                   
        void svr(std::vector<double> & , double &, double ) ; 



    /**

    * @brief velocityVerlet

     * @param[in] velocity Stores the velocity of each Charge, updated at each TimeStep
     * @param[in] force Stores the -ve of force on each charge, updated at each TimeStep
     * @param[in] massAtoms Stores the mass of each Charge.
     * @param[in] Temperature  temperature at current Timestep
     * 
     * @param[out] KE Kinetic Energy at current timestp in eV
     *    
     *  
     *     
     */ 
        void velocityVerlet(std::vector<double> &, std::vector<dealii::Tensor<1, 3, double>> &, 
                        std::vector<double> , double & , std::vector<double> & );

       
       void writeRestartFile(std::vector<double> , std::vector<double> , std::vector<double> ,
                              std::vector<double> , std::vector<double>, int );   

        void InitialiseFromRestartFile( std::vector<double> &, std::vector<double> & , std::vector<double> & , std::vector<double> & , std::vector<double> &  );   

       void writeRestartNHCfile(std::vector<double> , std::vector<double> , std::vector<double>, int  );   

        void InitialiseFromRestartNHCFile( std::vector<double> &, std::vector<double> & , std::vector<double> &   );  

        void writeTotalDisplacementFile(std::vector<dealii::Tensor<1, 3, double>> , int ) ;  

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
        
        double NoseHoverExtendedLagrangian(std::vector<double>  , std::vector<double>  , std::vector<double> , double, double, double  );                                    
        


    
    };
}
#endif