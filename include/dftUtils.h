 namespace dftUtils
 {
   inline double getPartialOccupancy(const double eigenValue,const double fermiEnergy,const double kb,const double T)
   {
      const double factor=(eigenValue-fermiEnergy)/(kb*T);       
      return (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
   }
 }
