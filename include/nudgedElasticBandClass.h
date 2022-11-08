#ifndef nebGlobalClass_H_
#define nebGlobalClass_H_
#include "constants.h"
#include "headers.h"
#include <vector>
#include "nonlinearSolverProblem.h"
#include "dft.h"
namespace dftfe
{
    using namespace dealii;
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    class dftClass;
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    class nebGlobalClass : public nonlinearSolverProblem
    {
        public:

            const double haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
            const double haToeV              = 27.211386245988;
            const double bohrToAng           = 0.529177210903;
            const double pi                            = 3.14159265359;
            const double AngTobohr           = 1.0 / bohrToAng;
            double kmax                        = 0.1; //0.1 Ha/bohr
            double kmin                        = 0.1; //0.1Ha/bohr
            unsigned int NEBImageno                  ;

            nebGlobalClass(std::vector<dftClass<FEOrder, FEOrderElectro>*> &_dftPtr,
                            const MPI_Comm &mpi_comm_parent,
                            int startStep);



            void runNEB();         
            void ReturnNormedVector(std::vector<double> & , int);
            void LNorm(double &, std::vector<double> , int, int); 
            void gradient(std::vector<double> &gradient); 
 

            unsigned int
            getNumberUnknowns() const;

            void
            update(const std::vector<double> &solution,
            const bool                 computeForces      = true,
            const bool useSingleAtomSolutionsInitialGuess = false);

            void 
            save();

            void value(std::vector<double> &functionValue);

            void
            precondition(std::vector<double> &      s,
             const std::vector<double> &gradient) const;  

            void
            solution(std::vector<double> &solution); 
            void
            Fire();
            
            
            std::vector<unsigned int>
            getUnknownCountFlag() const;      

        private:
            std::vector<dftClass<FEOrder, FEOrderElectro>*> dftPtr;

        // parallel communication objects
            const MPI_Comm     d_mpiCommParent;
            const unsigned int n_mpi_processes;
            const unsigned int this_mpi_process;

        // conditional stream object
            dealii::ConditionalOStream pcout;
    
        /// total number of calls to update()
            unsigned int d_totalUpdateCalls;    
            int           d_startStep;  
            
            unsigned int restartFlag;
            unsigned int numberGlobalCharges;
            unsigned int numberofAtomTypes;  
            double d_maximumAtomForceToBeRelaxed; 
            unsigned int numberofImages;
            unsigned int maximumIterationNEB;
            double optimizertolerance;
            unsigned int optimizermatItr;
            double       Forcecutoff;
            unsigned int countrelaxationFlags;
            std::vector<unsigned int> d_relaxationFlags;
             std::vector<double>       d_externalForceOnAtom;
            std::vector<double> ForceonImages;
            std::vector<double> d_ImageError;
            std::vector<double> d_Length;
            const MPI_Comm &
            getMPICommunicator();
            void CalculatePathTangent(int , std::vector<double> &);
            void CalculateForceparallel(int , std::vector<double> & , std::vector<double> );
            void CalculateForceperpendicular(int , std::vector<double> & , std::vector<double> , std::vector<double>);
            void CalculateSpringForce(int , std::vector<double> &, std::vector<double> );
            void CalculateForceonImage(std::vector<double>, std::vector<double>, std::vector<double> &);
            void CalculatePathLength(double &);
            void WriteRestartFiles(int step);
            void CalculateSpringConstant(int, double & );
            void ImageError(int image, double &Force);

            



    
    };


}
#endif