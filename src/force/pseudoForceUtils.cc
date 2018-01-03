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
// @author Sambit Das (2017)
//


#include <boost/math/special_functions/spherical_harmonic.hpp>


namespace pseudoForceUtils
{
     double tolerance = 1e-12;
   
     inline 
     void getRadialFunctionVal(const double radialCoordinate,
			  double &splineVal,
			  const alglib::spline1dinterpolant * spline) 
     {
  
       splineVal = alglib::spline1dcalc(*spline,
				        radialCoordinate);
       return;
     }

     inline
     void
     getSphericalHarmonicVal(const double theta, const double phi, const int l, const int m, double & sphericalHarmonicVal)
     {
      
       if(m < 0)
          sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_i(l,-m,theta,phi);
      
       else if (m == 0)
          sphericalHarmonicVal = boost::math::spherical_harmonic_r(l,m,theta,phi);

       else if (m > 0)
          sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_r(l,m,theta,phi);

       return;

     }
     
     inline
     void
     convertCartesianToSpherical(double *x, double & r, double & theta, double & phi)
     {

       r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
 
       if(r == 0)
        {
           theta = 0.0;
           phi = 0.0;
        }
       else
        {
           theta = acos(x[2]/r);
	   //
	   // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
	   // If yes, assign phi = 0.0.
	   // NOTE: In case theta = 0 or PI, phi is undetermined. The actual value 
	   // of phi doesn't matter in computing the enriched function value or 
	   // its gradient. We assign phi = 0.0 here just as a dummy value
	   //
	   if(fabs(theta - 0.0) >= tolerance && fabs(theta - M_PI) >= tolerance)
	      phi = atan2(x[1],x[0]);
	   else
	      phi = 0.0;
	}
    }



    inline 
    void
    getPolarFunctionVal(const double theta, const int l, const int m, double & polarFunctionVal)
    {

      const int modM = abs(m);

      //
      // if |m| > l , polarVal = 0.0
      //
      if(modM > l)
	{

	  polarFunctionVal = 0.0;

	}

      else
	{

	  //
	  // get the value of cosine(theta)
	  //
	  const double cosTheta = cos(theta);

	  //
	  // get the Associated Lengendre Polynomial value
	  //
	  polarFunctionVal =  boost::math::legendre_p(l, modM, cosTheta);

	  //
	  // compute the normalizing constant
	  //
	  const double normalizingCoeff = sqrt(((2.0*l + 1)*boost::math::factorial<double>(l-modM))/(2.0*boost::math::factorial<double>(l+modM)));

	  //
	  // normalize the legendre polynomial
	  //
	  polarFunctionVal *= normalizingCoeff;

	}

      return;
    
    }

    //
    // Compute the azimuthal part of the enriched function.
    // Real form of spherical harmonic is used for the non-periodic problem.
    // The azimuthal part is:
    // cos(m*phi)/sqrt(PI) 	if  m > 0;
    // sin(|m|*phi)/sqrt(PI)  	if m < 0
    // 1/sqrt(2*PI) 		if m = 0;
    //
 
    inline 
    void
    getAzimuthalFunctionVal(const double phi, const int m, double  & azimuthalFunctionVal)
    {

      if(m > 0) 
        azimuthalFunctionVal = cos(m*phi)/sqrt(M_PI);

      else if(m == 0)
        azimuthalFunctionVal = 1.0/sqrt(2*M_PI);

      else if(m < 0)
        azimuthalFunctionVal = sin(abs(m)*phi)/sqrt(M_PI);

      else
	{

	  std::string message("Invalid m quantum number.");
	  Assert(false,ExcMessage(message));
	}
      
      return;

    }

    //
    // compute the derivative of the polar part of the enriched w.r.t the polar angle (theta)
    // Real form of spherical harmonic used for non-periodic problem.
    // dP(l, |m|, cos(theta))/d(theta) = 0.5*(P(l, |m|+1, cos(theta)) - (l+|m|)*(l-|m|+1)*P(l, |m|-1, cos(theta)))
    //
    inline
    void
    getPolarFunctionDerivative(const double theta, const int l, const int m, double & polarDerivative)
    {

      //
      // |m|
      //
      const int modM = abs(m);

      //
      // get the value of cosine(theta)
      //
      const double cosTheta = cos(theta);
     
      const double term1 = (abs(modM+1) <= l) ? boost::math::legendre_p(l, modM + 1, cosTheta) : 0.0;
      const double term2 = (abs(modM-1) <= l) ? -1.0*(l+modM)*(l-modM+1)*boost::math::legendre_p(l, modM - 1, cosTheta) : 0.0;

      //
      // compute the normalizing constant
      //
      const double normalizingCoeff = sqrt(((2.0*l + 1)*boost::math::factorial<double>(l-modM))/(2.0*boost::math::factorial<double>(l+modM)));

      //
      // multiply the normalizing coeff 
      //
      polarDerivative = 0.5*(term1 + term2)*normalizingCoeff;
     
      return;

    } 
 
    //
    // compute the derivative of the azimuthal part of the enriched function w.r.t the azimuthal angle (phi)
    // Real form of spherical harmonic used for non-periodic problem.
    // dG(|m|, phi)/d(phi) is:
    // -|m|*sin(|m|*phi)/sqrt(PI) if m > 0;
    //  |m|*cos(|m|*phi)/sqrt(PI) if m < 0;
    //  0.0 if 			  if m = 0;
    // 
    inline
    void 
    getAzimuthalFunctionDerivative(const double phi, const int m, double & azimuthalDerivative)
    {

      if(m > 0)
        azimuthalDerivative = -m*sin(m*phi)/sqrt(M_PI);
 
      else if(m < 0)
	azimuthalDerivative = abs(m)*cos(abs(m)*phi)/sqrt(M_PI);

      else if(m == 0)
 	azimuthalDerivative = 0.0;

      else
	{

	  std::string message("Invalid m quantum number.");
	  Assert(false,ExcMessage(message));
	}
      
      return;

    }

    inline
    void
    getGradientForPointOnZAxis(const int l, const int m, const double theta, const double r, const double radialVal, const double radialDerivative, std::vector<double> & pseudoWaveFunctionDerivatives)
    {

      //
      // |m|
      //
      const int modM = abs(m);

      //
      // cosine(theta)
      //
      const double cosTheta = cos(theta);

      //
      // compute the normalizing constant
      //
      const double normalizingCoeff = sqrt(((2.0*l + 1)*boost::math::factorial<double>(l-modM))/(2.0*boost::math::factorial<double>(l+modM)));

      const double alpha = radialVal*normalizingCoeff/(2.0*r*sqrt(M_PI));

      if(m == 0)
	{
	  pseudoWaveFunctionDerivatives[0] = 0.0;
	  pseudoWaveFunctionDerivatives[1] = 0.0;
	  pseudoWaveFunctionDerivatives[2] = normalizingCoeff/sqrt(2.0*M_PI)*radialDerivative*boost::math::legendre_p(l, modM, cosTheta)*cosTheta;
 
	}

      else if(m == 1)
	{

	  if(fabs(theta - 0.0) < tolerance)
	    {

	      pseudoWaveFunctionDerivatives[0] = -1.0*alpha*l*(l+1);
	      pseudoWaveFunctionDerivatives[1] = 0.0;
	      pseudoWaveFunctionDerivatives[2] = 0.0;

	    }

	  else if(fabs(theta - M_PI) < tolerance)
	    {

	      pseudoWaveFunctionDerivatives[0] = -1.0*alpha*l*(l+1)*pow(-1, l-1);
	      pseudoWaveFunctionDerivatives[1] = 0.0;
	      pseudoWaveFunctionDerivatives[2] = 0.0;
	    }

	  else
	    {
	
	      std::string message("Value of theta for the point lying on Z axis is neither zero nor PI");
	      Assert(false,ExcMessage(message));

	    }

	}

      else if(m == -1)
	{

	  if(fabs(theta - 0.0) < tolerance)
	    {

	      pseudoWaveFunctionDerivatives[0] = 0.0;
	      pseudoWaveFunctionDerivatives[1] = -1.0*alpha*l*(l+1);
	      pseudoWaveFunctionDerivatives[2] = 0.0;
	
	    }

	  else if(fabs(theta - M_PI) < tolerance)
	    {

	      pseudoWaveFunctionDerivatives[0] = 0.0;
	      pseudoWaveFunctionDerivatives[1] = -1.0*alpha*l*(l+1)*pow(-1, l-1);
	      pseudoWaveFunctionDerivatives[2] = 0.0;
	    }

	  else
	    {
	
	      std::string message("Value of theta for the point lying on Z axis is neither zero nor PI");
	      Assert(false,ExcMessage(message));

	    }

	}
	
      else
	{
	  pseudoWaveFunctionDerivatives[0] = 0.0;
	  pseudoWaveFunctionDerivatives[1] = 0.0;
	  pseudoWaveFunctionDerivatives[2] = 0.0;
	}



      return;

    }

  
    inline 
    void 
    getRadialFunctionDerivative(const double radialCoordinate,
	 			double & splineVal,
                                double & dSplineVal,
				double & ddSplineVal,
				const alglib::spline1dinterpolant * spline) 
    {
  
      alglib::spline1ddiff(*spline,
                           radialCoordinate,
                           splineVal,
                           dSplineVal,
                           ddSplineVal);
  
      return;
  
    }

    inline
    void
    getPseudoWaveFunctionDerivatives(const double r, 
				     const double theta, 
				     const double phi, 
				     const int lQuantumNumber, 
				     const int mQuantumNumber,
				     std::vector<double> & pseudoWaveFunctionDerivatives,
				     const alglib::spline1dinterpolant & spline)
    {

      //
      // define variable to store the radial function value, radial function radial derivatives (first and second)
      //
      double radialVal, dRadialValDr, ddRadialValD2r;

      double jacobianInverse[3][3], partialDerivativesR, partialDerivativesTheta, partialDerivativesPhi;

      alglib::spline1ddiff(spline,
                           r,
                           radialVal,
                           dRadialValDr,
                           ddRadialValD2r);

      //
      // define variable to store the polar function value, polar function derivative (w.r.t polar angle theta), azimuthal function value 
      // and azimuthal angle derivative (w.r.t azimuthal angle phi)
      //
      double polarVal, dPolarValDtheta, azimuthalVal, dAzimuthalValDphi;
      if(fabs(theta - 0.0) < tolerance || fabs(theta - M_PI) < tolerance)
	{

	  getGradientForPointOnZAxis(lQuantumNumber, mQuantumNumber, theta, r, radialVal, dRadialValDr, pseudoWaveFunctionDerivatives);

	}
      else
	{
	  //
	  // get the value for polar part of the enriched function
	  //
	  getPolarFunctionVal(theta, lQuantumNumber, mQuantumNumber, polarVal); 

	  //
	  // get the derivative of the polar function w.r.t theta
	  //
	  getPolarFunctionDerivative(theta, lQuantumNumber, mQuantumNumber, dPolarValDtheta);

	  //
	  // get the value for azimuthal part of the enriched function
	  //
	  getAzimuthalFunctionVal(phi, mQuantumNumber, azimuthalVal);

	  //
	  // get the derivative of the azimuthal function w.r.t phi
	  //
	  getAzimuthalFunctionDerivative(phi, mQuantumNumber, dAzimuthalValDphi);

	  //
	  // assign values to the entries in jacobian inverse
	  //
	  jacobianInverse[0][0] = sin(theta)*cos(phi); jacobianInverse[0][1] = cos(theta)*cos(phi)/r; jacobianInverse[0][2] = -1.0*sin(phi)/(r*sin(theta));
	  jacobianInverse[1][0] = sin(theta)*sin(phi); jacobianInverse[1][1] = cos(theta)*sin(phi)/r; jacobianInverse[1][2] = cos(phi)/(r*sin(theta));
	  jacobianInverse[2][0] = cos(theta); 	   jacobianInverse[2][1] = -1.0*sin(theta)/r;     jacobianInverse[2][2] = 0.0;

	  //
	  // assign values to the partialDerivativesVector i.e., the partial derivatives of the pseudowave function w.r.t r, theta and phi
	  //
	  partialDerivativesR = dRadialValDr*polarVal*azimuthalVal;
	  partialDerivativesTheta = radialVal*dPolarValDtheta*azimuthalVal;
	  partialDerivativesPhi = radialVal*polarVal*dAzimuthalValDphi;

	  //
	  //fill in pseudoWaveFunctionDerivativesVector
	  //
	  pseudoWaveFunctionDerivatives[0] = jacobianInverse[0][0]*partialDerivativesR + jacobianInverse[0][1]*partialDerivativesTheta + jacobianInverse[0][2]*partialDerivativesPhi;

	  pseudoWaveFunctionDerivatives[1] = jacobianInverse[1][0]*partialDerivativesR + jacobianInverse[1][1]*partialDerivativesTheta + jacobianInverse[1][2]*partialDerivativesPhi;

	  pseudoWaveFunctionDerivatives[2] = jacobianInverse[2][0]*partialDerivativesR + jacobianInverse[2][1]*partialDerivativesTheta;


	}

      return;

    }

    inline
    void 
    getDeltaVlDerivatives(const double r,
			  double *x,
			  std::vector<double> & deltaVlDerivatives,
			  const alglib::spline1dinterpolant & spline)
    {
      //
      // define variable to store the radial function value, radial function radial derivatives (first and second)
      //
      double radialVal, dRadialValDr, ddRadialValD2r;

      alglib::spline1ddiff(spline,
			   r,
			   radialVal,
			   dRadialValDr,
			   ddRadialValD2r);

      deltaVlDerivatives[0] = dRadialValDr*(x[0]/r);
      deltaVlDerivatives[1] = dRadialValDr*(x[1]/r);
      deltaVlDerivatives[2] = dRadialValDr*(x[2]/r);

    }

   
}
