#ifndef _CGD_WRAPPER_H__
#define _CGD_WRAPPER_H__

#include <math.h>
#include <memory>
#include <dftUtils.h>
#include <nonlinearSolverProblem.h>
#include <cg_user.h>

/* entries in cg_descent
 * cg_descent (x, n, NULL, NULL, grad_tol, user_value, user_gradient, user_value_gradient, NULL) ;
 * where n is x.size(), grad_tol is a float
 * must have signature
 *
 *  double pycgd_value(double *x, INT n);
 *  void pycgd_gradient(double *g, double *x, INT n);
 *
 * Performance is often improved if the user also provides a routine to
 * simultaneously evaluate the objective function and its gradient
 *
 *  double pycgd_value_gradient(double *g, double *x, INT n);
 *
 *  NOTE:   this version relies on the pele library
 *
 */

namespace {

	dftfe::nonlinearSolverProblem* functionToBeMinimized;
	size_t glob_nfev;

	inline double cgd_value(double* x, INT n)
	{

		//compute the increment in the solution vector
		std::vector<double> xPrev,solutionInc;
		functionToBeMinimized->solution(xPrev);
		solutionInc.resize(xPrev.size());

		for(unsigned int i = 0; i < xPrev.size(); ++i)
			solutionInc[i] = *(x+i) - xPrev[i];

		MPI_Bcast(&solutionInc[0],
				solutionInc.size(),
				MPI_DOUBLE,
				0,
				MPI_COMM_WORLD);

		double sumAbsDisp=0.0;
		for(unsigned int i = 0; i < solutionInc.size(); ++i)
			sumAbsDisp+=std::fabs(solutionInc[i]);

		if (sumAbsDisp>1e-12)
			functionToBeMinimized->update(solutionInc,false);

		std::vector<double> funcValue;
		functionToBeMinimized->value(funcValue);

		functionToBeMinimized->save();

		++glob_nfev;
		return funcValue[0];

		//AssertThrow(false,dftfe::dftUtils::ExcNotImplementedYet());


	}

	inline void cgd_gradient(double* g, double* x, INT n)
	{
		//pele::Array<double> xarray(x, (size_t) n);
		//pele::Array<double> garray(g, (size_t) n);

		//glob_pot->get_energy_gradient(xarray, garray);

		//compute the increment in the solution vector
		std::vector<double> xPrev,solutionInc;
		functionToBeMinimized->solution(xPrev);
		solutionInc.resize(xPrev.size());

		for(unsigned int i = 0; i < xPrev.size(); ++i)
			solutionInc[i] = *(x+i) - xPrev[i];

		MPI_Bcast(&solutionInc[0],
				solutionInc.size(),
				MPI_DOUBLE,
				0,
				MPI_COMM_WORLD);

		double sumAbsDisp=0.0;
		for(unsigned int i = 0; i < solutionInc.size(); ++i)
			sumAbsDisp+=std::fabs(solutionInc[i]);

		if (sumAbsDisp>1e-12)
			functionToBeMinimized->update(solutionInc);

		functionToBeMinimized->save();

		std::vector<double> gradient;
		functionToBeMinimized->gradient(gradient);

		for(unsigned int i = 0; i < xPrev.size(); ++i)
			*(g+i) = gradient[i];

		++glob_nfev;

		//AssertThrow(false,dftfe::dftUtils::ExcNotImplementedYet());

	}

	inline double cgd_value_gradient(double* g, double* x, INT n)
	{
		//pele::Array<double> xarray(x, (size_t) n);
		//pele::Array<double> garray(g, (size_t) n);
		++glob_nfev;
		//return glob_pot->get_energy_gradient(xarray, garray);

		//compute the increment in the solution vector
		std::vector<double> xPrev,solutionInc;
		functionToBeMinimized->solution(xPrev);
		solutionInc.resize(xPrev.size());

		for(unsigned int i = 0; i < xPrev.size(); ++i)
			solutionInc[i] = *(x+i) - xPrev[i];

		MPI_Bcast(&solutionInc[0],
				solutionInc.size(),
				MPI_DOUBLE,
				0,
				MPI_COMM_WORLD);

		double sumAbsDisp=0.0;
		for(unsigned int i = 0; i < solutionInc.size(); ++i)
			sumAbsDisp+=std::fabs(solutionInc[i]);

		if (sumAbsDisp>1e-12)
			functionToBeMinimized->update(solutionInc);

		functionToBeMinimized->save();  

		std::vector<double> funcValue;
		functionToBeMinimized->value(funcValue);

		std::vector<double> gradient;
		functionToBeMinimized->gradient(gradient);

		for(unsigned int i = 0; i < xPrev.size(); ++i)
			*(g+i) = gradient[i];

		return funcValue[0];

	}

	inline int cgd_test_callback(double f, double* x, double* g, INT n, void* user_data);

} // namespace

namespace dftfe {

	class CGDescent {
		protected:
			cg_parameter m_parm;
			cg_stats m_stats;
			std::vector<double> m_x0, m_x;
			double m_tol;
			size_t m_nfev;
			bool m_success;
		public:
			CGDescent(double tol, size_t maxIter,  size_t PrintLevel=0)
				: m_parm(),
				m_stats(),
				m_tol(tol),
				m_nfev(0),
				m_success(false)
		{
			/*set default parameter values*/
			cg_default(&m_parm);
			m_parm.PrintFinal = FALSE;
			m_parm.AWolfeFac = 0;
			m_parm.AWolfe = FALSE;
			m_parm.memory = 0;
			m_parm.maxit = maxIter;
			m_parm.PrintLevel = PrintLevel;
			m_parm.psi2 = 10.0;

		}

			virtual ~CGDescent() {}

			virtual bool test_convergence(double energy, std::vector<double> x, std::vector<double> g) { return false; }

			//run
			inline bool run(nonlinearSolverProblem & problem)
			{
				functionToBeMinimized = &problem;
				glob_nfev = 0; //reset global variable
				m_nfev = 0;
				m_success = false;

				//Get initial values of x
				problem.solution(m_x);
				std::vector<double> gradient;
				problem.gradient(gradient);

				//using ::cg_descent call in the global namespace (the one in CG_DESCENT 6.8), this resolves the ambiguity
				INT cgout = ::cg_descent(&m_x[0], m_x.size(), &m_stats, &m_parm, m_tol, cgd_value,
						cgd_gradient, cgd_value_gradient, NULL, cgd_test_callback, (void*) this);
				m_success = this->test_success(cgout);
				m_nfev = glob_nfev;

				return m_success;


			}

			/*inline void run(size_t maxiter)
			  {
			  this->set_maxit(maxiter);
			  this->run();
			  }*/

			inline void reset(std::vector<double> &x)
			{
				this->set_x(x);
				m_nfev = 0;
				m_success = false;
			}

			/*============================================================================
			  CG_DESCENT6.8 parameters that the user may wish to modify
			  ----------------------------------------------------------------------------*/
			/* Level 0 = no printing, ... , Level 3 = maximum printing */
			inline void set_PrintLevel(int val) { m_parm.PrintLevel = val; }
			/* abort cg after maxit iterations */
			inline void set_maxit(size_t val) { m_parm.maxit = (INT) val; }
			/* number of vectors stored in memory */
			inline void set_memory(int memory) { m_parm.memory = memory; }
			/* T => use approximate Wolfe line search
			   F => use ordinary Wolfe line search, switch to approximate Wolfe when
			   |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost  */
			inline void set_AWolfeFac(double val) { m_parm.AWolfeFac = val; }
			inline void set_AWolfe(bool val) { m_parm.AWolfe = (int) val; }
			/* T => use LBFGS
			   F => only use L-BFGS when memory >= n */
			inline void set_lbfgs(bool val) { m_parm.LBFGS = (int) val; }
			/* T => attempt quadratic interpolation in line search when
			   |f_k+1 - f_k|/f_k <= QuadCutoff
			   F => no quadratic interpolation step */
			inline void set_QuadStep(bool val) { m_parm.QuadStep = (int) val; }
			inline void set_QuadCutOff(double val) { m_parm.QuadCutOff = val; }
			/* maximum factor by which a quad step can reduce the step size */
			inline void set_QuadSafe(double val){ m_parm.QuadSafe = val; }
			/* T => when possible, use a cubic step in the line search */
			inline void set_UseCubic(bool val){ m_parm.UseCubic = (int) val; }
			/* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
			inline void set_CubicCutOff(double val) { m_parm.CubicCutOff = val; }
			/* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = FALSE*/
			inline void set_SmallCost(double val) { m_parm.SmallCost = val; }
			/* if step is nonzero, it is the initial step of the initial line search */
			inline void set_step(double val) { m_parm.step = val; }
			/* terminate after nslow iterations without strict improvement in
			   either function value or gradient */
			inline void set_nslow(int val) { m_parm.nslow = val; }
			/* Stop Rules:
			   T => ||proj_grad||_infty <= max(grad_tol,initial ||grad||_infty*StopFact)
			   F => ||proj_grad||_infty <= grad_tol*(1 + |f_k|) */
			inline void set_StopRule(bool val) { m_parm.StopRule = (int) val; }
			inline void set_StopFac(double val) { m_parm.StopFac = val; }

			inline void set_x(std::vector<double> & x)
			{
				m_x = x;
				m_x0 = m_x;
			}
			/* SubCheck and SubSkip control the frequency with which the subspace
			   condition is checked. It is checked for SubCheck*mem iterations and
			   if not satisfied, then it is skipped for Subskip*mem iterations
			   and Subskip is doubled. Whenever the subspace condition is statisfied,
			   SubSkip is returned to its original value. */
			inline void set_SubCheck(int val) { m_parm.SubCheck = val; }
			inline void set_SubSkip(int val) { m_parm.SubSkip = val; }
			/* when relative distance from current gradient to subspace <= eta0,
			   enter subspace if subspace dimension = mem */
			inline void set_eta0(double val) { m_parm.eta0 = val; }
			/* when relative distance from current gradient to subspace >= eta1,
			   leave subspace */
			inline void set_eta1(double val) { m_parm.eta1 = val; }
			/* when relative distance from current direction to subspace <= eta2,
			   always enter subspace (invariant space) */
			inline void set_eta2(double val) { m_parm.eta2 = val; }
			/* factor in [0, 1] used to compute average cost magnitude C_k as follows:
			   Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
			inline void set_Qdecay(double val) { m_parm.Qdecay = val; }
			/* T => estimated error in function value is eps*Ck,
			   F => estimated error in function value is eps */
			inline void set_PertRule(bool val) { m_parm.PertRule = (int)val; }
			inline void set_eps(double val) { m_parm.eps = val; }
			/* factor by which eps grows when line search fails during contraction */
			inline void set_egrow(double val) { m_parm.egrow = val; }
			/* T => check that f_k+1 - f_k <= debugtol*C_k
			   F => no checking of function values */
			inline void set_debug(bool val) { m_parm.debug = (int)val; }
			inline void set_debugtol(double val) { m_parm.debugtol = val; }
			/* maximum number of times the bracketing interval grows during expansion */
			inline void set_ntries(int val) { m_parm.ntries = val; }
			/* maximum factor secant step increases stepsize in expansion phase */
			inline void set_ExpandSafe(double val) { m_parm.ExpandSafe = val; }
			/* factor by which secant step is amplified during expansion phase
			   where minimizer is bracketed */
			inline void set_SecantAmp(double val) { m_parm.SecantAmp = val; }
			/* factor by which rho grows during expansion phase where minimizer is
			   bracketed */
			inline void set_RhoGrow(double val) { m_parm.RhoGrow = val; }
			/* maximum number of times that eps is updated */
			inline void set_neps(int val) { m_parm.neps = val; }
			/* maximum number of times the bracketing interval shrinks */
			inline void set_nshrink(int val) { m_parm.nshrink = val; }
			/* maximum number of iterations in line search */
			inline void set_nline(int val) { m_parm.nline = val; }
			/* conjugate gradient method restarts after (n*restart_fac) iterations */
			inline void set_restart_fac(double val) { m_parm.restart_fac = val; }
			/* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
			inline void set_feps(double val) { m_parm.feps = val; }
			/* after encountering nan, growth factor when searching for
			   a bracketing interval */
			inline void set_nan_rho(double val) { m_parm.nan_rho = val; }
			/* after encountering nan, decay factor for stepsize */
			inline void set_nan_decay(double val) { m_parm.nan_decay = val; }

			/*============================================================================
			  CG_DESCENT6.8 technical parameters which the user probably should not touch
			  ----------------------------------------------------------------------------*/
			/* Wolfe line search parameter */
			inline void set_delta(double val) { m_parm.delta = val; }
			/* Wolfe line search parameter */
			inline void set_sigma(double val) { m_parm.sigma = val; }
			/* decay factor for bracket interval width */
			inline void set_gamma(double val) { m_parm.gamma = val; }
			/* growth factor when searching for initial bracketing interval */
			inline void set_rho(double val) { m_parm.rho = val; }
			/* factor used in starting guess for iteration 1 */
			inline void set_psi0(double val) { m_parm.psi0 = val; }
			/* in performing a QuadStep, we evaluate at point betweeen
			   [psi_lo, psi_hi]*psi2*previous step */
			inline void set_psi_lo(double val) { m_parm.psi_lo = val; }
			inline void set_psi_hi(double val) { m_parm.psi_hi = val; }
			/* for approximate quadratic, use gradient at psi1*psi2*previous step
			   for initial stepsize */
			inline void set_psi1(double val) { m_parm.psi1 = val; }
			/* when starting a new cg iteration, our initial guess for the line search
			   stepsize is psi2*previous step */
			inline void set_psi2(double val) { m_parm.psi2 = val; }
			/* T => choose beta adaptively, F => use theta */
			inline void set_AdaptiveBeta(bool val) { m_parm.AdaptiveBeta = (int)val; }
			/* lower bound factor for beta */
			inline void set_BetaLower(double val) { m_parm.BetaLower = val; }
			/* parameter describing the cg_descent family */
			inline void set_theta(double val) { m_parm.theta = val; }
			/* parameter in cost error for quadratic restart criterion */
			inline void set_qeps(double val) { m_parm.qeps = val; }
			/* parameter used to decide if cost is quadratic */
			inline void set_qrule(double val) { m_parm.qrule = val; }
			/* number of iterations the function should be nearly quadratic before
			   a restart */
			inline void set_qrestart(int val) { m_parm.qrestart = val; }

			/*============================================================================
			  results returned that can be retrieved by the user
			  ----------------------------------------------------------------------------*/

			/*function value at solution */
			inline double get_f() { return m_stats.f; }
			/* max abs component of gradient */
			inline double get_gnorm() { return m_stats.gnorm; }
			/* number of iterations */
			inline size_t get_iter() { return m_stats.iter; }
			/* number of subspace iterations */
			inline size_t get_IterSub() { return m_stats.IterSub; }
			/* total number subspaces */
			inline size_t get_NumSub() { return m_stats.NumSub; }
			/* number of function evaluations */
			inline size_t get_nfunc() { return m_stats.nfunc; }
			/* number of gradient evaluations */
			inline size_t get_ngrad() { return m_stats.ngrad; }
			/* total number of function evaluations from global counter*/
			inline size_t get_nfev() { return m_nfev; }
			/*return success status*/
			inline bool success() { return m_success; }
			/*return modified coordinates*/
			//inline pele::Array<double> get_x() { return m_x.copy(); }
			/*return gradient*/
			//inline pele::Array<double> get_g()
			//{
			//  pele::Array<double> g(m_x.size());
			//    m_pot->get_energy_gradient(m_x, g);
			//    return g.copy();
			// }
			/*get root mean square gradient*/
			//inline double get_rms()
			//{
			//  pele::Array<double> g = this->get_g();
			//    return norm(g) / sqrt(m_x.size());
			// }
			inline void set_tol(double val) { m_tol = val; }
			inline double get_tol() { return m_tol; }

		protected:
			inline bool test_success(INT cgout)
			{
				if (cgout == 0) {
					return true;
				}
				else {
					switch (cgout) {
						case -2:
							std::cout << "function value became nan" << std::endl;
							break;
						case -1:
							std::cout << "starting function value is nan" << std::endl;
							break;
						case 1:
							std::cout << "change in func <= feps*|f|" << std::endl;
							break;
						case 2:
							std::cout << "total iterations exceeded maxit" << std::endl;
							break;
						case 3:
							std::cout << "slope always negative in line search" << std::endl;
							break;
						case 4:
							std::cout << "number secant iterations exceed nsecant" << std::endl;
							break;
						case 5:
							std::cout << "search direction not a descent direction" << std::endl;
							break;
						case 6:
							std::cout << "line search fails in initial interval" << std::endl;
							break;
						case 7:
							std::cout << "line search fails during bisection" << std::endl;
							break;
						case 8:
							std::cout << "line search fails during interval update" << std::endl;
							break;
						case 9:
							std::cout << "debugger is on and the function value increases" << std::endl;
							break;
						case 10:
							std::cout << "out of memory" << std::endl;
							break;
						case 11:
							std::cout << "function nan or +-INF and could not be repaired" << std::endl;
							break;
						case 12:
							std::cout << "invalid choice for memory parameter" << std::endl;
							break;
						default:
							std::cout << "failed, value not known" << std::endl;
							break;
					}
					return false;
				}
			}
	};

} // namespace pycgd

namespace {

	inline int cgd_test_callback(double f, double* x, double* g, INT n, void* user_data)
	{
		dftfe::CGDescent * cgd = static_cast<dftfe::CGDescent*>(user_data);
		std::vector<double> xarray;
		std::vector<double> garray;
		return (int) cgd->test_convergence(f, xarray, garray);
	}

} // namespace

#endif // #ifndef _CGD_WRAPPER_H__
