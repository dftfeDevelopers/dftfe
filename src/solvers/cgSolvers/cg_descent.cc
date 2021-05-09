/* =========================================================================
   ============================ CG_DESCENT =================================
   =========================================================================
   ________________________________________________________________
   |      A conjugate gradient method with guaranteed descent       |
   |             C-code Version 1.1  (October 6, 2005)              |
   |                    Version 1.2  (November 14, 2005)            |
   |                    Version 2.0  (September 23, 2007)           |
   |                    Version 3.0  (May 18, 2008)                 |
   |                    Version 4.0  (March 28, 2011)               |
   |                    Version 4.1  (April 8, 2011)                |
   |                    Version 4.2  (April 14, 2011)               |
   |                    Version 5.0  (May 1, 2011)                  |
   |                    Version 5.1  (January 31, 2012)             |
   |                    Version 5.2  (April 17, 2012)               |
   |                    Version 5.3  (May 18, 2012)                 |
   |                    Version 6.0  (November 6, 2012)             |
   |                    Version 6.1  (January 27, 2013)             |
   |                    Version 6.2  (February 2, 2013)             |
   |                    Version 6.3  (April 21, 2013)               |
   |                    Version 6.4  (April 29, 2013)               |
   |                    Version 6.5  (April 30, 2013)               |
   |                    Version 6.6  (May 28, 2013)                 |
   |                    Version 6.7  (April 7, 2014)                |
   |                    Version 6.8  (March 7, 2015)                |
   |                                                                |
   |           William W. Hager    and   Hongchao Zhang             |
   |          hager@math.ufl.edu       hozhang@math.lsu.edu         |
   |                   Department of Mathematics                    |
   |                     University of Florida                      |
   |                 Gainesville, Florida 32611 USA                 |
   |                      352-392-0281 x 244                        |
   |                                                                |
   |                 Copyright by William W. Hager                  |
   |                                                                |
   |          http://www.math.ufl.edu/~hager/papers/CG              |
   |                                                                |
   |  Disclaimer: The views expressed are those of the authors and  |
   |              do not reflect the official policy or position of |
   |              the Department of Defense or the U.S. Government. |
   |                                                                |
   |      Approved for Public Release, Distribution Unlimited       |
   |________________________________________________________________|
   ________________________________________________________________
   |This program is free software; you can redistribute it and/or   |
   |modify it under the terms of the GNU General Public License as  |
   |published by the Free Software Foundation; either version 2 of  |
   |the License, or (at your option) any later version.             |
   |This program is distributed in the hope that it will be useful, |
   |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
   |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
   |GNU General Public License for more details.                    |
   |                                                                |
   |You should have received a copy of the GNU General Public       |
   |License along with this program; if not, write to the Free      |
   |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
   |MA  02110-1301  USA                                             |
   |________________________________________________________________|

References:
1. W. W. Hager and H. Zhang, A new conjugate gradient method
with guaranteed descent and an efficient line search,
SIAM Journal on Optimization, 16 (2005), 170-192.
2. W. W. Hager and H. Zhang, Algorithm 851: CG_DESCENT,
A conjugate gradient method with guaranteed descent,
ACM Transactions on Mathematical Software, 32 (2006), 113-137.
3. W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58.
4. W. W. Hager and H. Zhang, Limited memory conjugate gradients,
SIAM Journal on Optimization, 23 (2013), 2150-2168. */

#include "cg_user.h"
#include "cg_descent.h"
#include "cg_blas.h"

/* begin external variables */
double   one[1], zero[1];
BLAS_INT blas_one[1];
/* end external variables */

int cg_descent             /*  return status of solution process:
                   0 (convergence tolerance satisfied)
                   1 (change in func <= feps*|f|)
                   2 (total number of iterations exceeded maxit)
                   3 (slope always negative in line search)
                   4 (number of line search iterations exceeds nline)
                   5 (search direction not a descent direction)
                   6 (excessive updating of eps)
                   7 (Wolfe conditions never satisfied)
                   8 (debugger is on and the function value increases)
                   9 (no cost or gradient improvement in
                   2n + Parm->nslow iterations)
                   10 (out of memory)
                   11 (function nan or +-INF and could not be repaired)
                   12 (invalid choice for memory parameter) */
  (double *      x,        /* input: starting guess, output: the solution */
   INT           n,        /* problem dimension */
   cg_stats *    Stat,     /* structure with statistics (can be NULL) */
   cg_parameter *UParm,    /* user parameters, NULL = use default parameters */
   double        grad_tol, /* StopRule = 1: |g|_infty <= max (grad_tol,
            StopFac*initial |g|_infty) [default]
            StopRule = 0: |g|_infty <= grad_tol(1+|f|) */
   double (*value)(double *, INT),             /* f = value (x, n) */
   void (*grad)(double *, double *, INT),      /* grad (g, x, n) */
   double (*valgrad)(double *, double *, INT), /* f = valgrad (g, x, n),
              NULL = compute value & gradient using value & grad */
   double *Work, /* NULL => let code allocate memory
  not NULL => use array Work for required memory
  The amount of memory needed depends on the value
  of the parameter memory in the Parm structure.
  memory > 0 => need (mem+6)*n + (3*mem+9)*mem + 5
  where mem = MIN(memory, n)
  memory = 0 => need 4*n */
   int (*user_test)(double, double *, double *, INT, void *),
   void *user_data)
{
  INT    i, iter, IterRestart, maxit, n5, nrestart, nrestartsub;
  int    nslow, slowlimit, IterQuad, status, PrintLevel, QuadF, StopRule;
  double delta2, Qk, Ck, fbest, gbest, f, ftemp, gnorm, xnorm, gnorm2, dnorm2,
    denom, t, dphi, dphi0, alpha, ykyk, ykgk, dkyk, beta, QuadTrust, tol, *d,
    *g, *xtemp, *gtemp, *work;

  /* new variables added in Version 6.0 */
  int l1, l2, j, k, mem, memsq, memk, memk_begin, mlast, mlast_sub, mp,
    mp_begin, mpp, nsub, spp, spp1, SkFstart, SkFlast, Subspace, UseMemory,
    Restart, LBFGS, InvariantSpace, IterSub, NumSub, IterSubStart,
    IterSubRestart, FirstFull, SubSkip, SubCheck, StartSkip, StartCheck,
    DenseCol1, NegDiag, memk_is_mem, d0isg, qrestart;
  double gHg, scale, gsubnorm2, ratio, stgkeep, alphaold, zeta, yty, ytg, t1,
    t2, t3, t4, *Rk, *Re, *Sk, *SkF, *stemp, *Yk, *SkYk, *dsub, *gsub,
    *gsubtemp, *gkeep, *tau, *vsub, *wsub;

  cg_parameter *Parm, ParmStruc;
  cg_com Com; /*this struct is not initialised, this is a very bad idea. 1 bug
                 found and fixed below*/

  /* assign values to the external variables */
  one[0]      = (double)1;
  zero[0]     = (double)0;
  blas_one[0] = (BLAS_INT)1;

  /* initialize the parameters */
  if (UParm == NULL)
    {
      Parm = &ParmStruc;
      cg_default(Parm);
    }
  else
    Parm = UParm;
  PrintLevel   = Parm->PrintLevel;
  qrestart     = MIN(n, Parm->qrestart);
  Com.Parm     = Parm;
  Com.eps      = Parm->eps;
  Com.PertRule = Parm->PertRule;
  Com.Wolfe    = FALSE;  /* initially Wolfe line search not performed */
  Com.nf       = (INT)0; /* number of function evaluations */
  Com.ng       = (INT)0; /* number of gradient evaluations */
  iter         = (INT)0; /* total number of iterations */
  QuadF        = FALSE;  /* initially function assumed to be nonquadratic */
  NegDiag      = FALSE;  /* no negative diagonal elements in QR factorization */
  mem          = Parm->memory; /* cg_descent corresponds to mem = 0 */

  if (Parm->PrintParms)
    cg_printParms(Parm);
  if ((mem != 0) && (mem < 3))
    {
      status = 12;
      goto Exit;
    }

  /* allocate work array */
  mem = MIN(mem, n);
  if (Work == NULL)
    {
      if (mem == 0) /* original CG_DESCENT without memory */
        {
          work = (double *)malloc(4 * n * sizeof(double));
        }
      else if (Parm->LBFGS || (mem >= n)) /* use L-BFGS */
        {
          work = (double *)malloc((2 * mem * (n + 1) + 4 * n) * sizeof(double));
        }
      else /* limited memory CG_DESCENT */
        {
          i    = (mem + 6) * n + (3 * mem + 9) * mem + 5;
          work = (double *)malloc(i * sizeof(double));
        }
    }
  else
    work = Work;
  if (work == NULL)
    {
      status = 10;
      goto Exit;
    }

  /* set up Com structure */
  Com.x     = x;
  Com.xtemp = xtemp = work;
  Com.d = d = xtemp + n;
  Com.g = g = d + n;
  Com.gtemp = gtemp = g + n;
  Com.df            = 0;
  Com.df0           = 0;
  Com.f             = 0;
  Com.f0            = 0;
  Com.n             = n;            /* problem dimension */
  Com.neps          = 0;            /* number of times eps updated */
  Com.AWolfe        = Parm->AWolfe; /* do not touch user's AWolfe */
  Com.cg_value      = value;
  Com.cg_grad       = grad;
  Com.cg_valgrad    = valgrad;
  StopRule          = Parm->StopRule;
  LBFGS             = FALSE;
  UseMemory         = FALSE; /* do not use memory */
  Subspace  = FALSE; /* full space, check subspace condition if UseMemory */
  FirstFull = FALSE; /* not first full iteration after leaving subspace */
  memk      = 0;     /* number of vectors in current memory */

  /* the conjugate gradient algorithm is restarted every nrestart iteration */
  nrestart = (INT)(((double)n) * Parm->restart_fac);

  /* allocate storage connected with limited memory CG */
  if (mem > 0)
    {
      if ((mem == n) || Parm->LBFGS)
        {
          LBFGS = TRUE; /* use L-BFGS */
          mlast = -1;
          Sk    = gtemp + n;
          Yk    = Sk + mem * n;
          SkYk  = Yk + mem * n;
          tau   = SkYk + mem;
        }
      else
        {
          UseMemory  = TRUE; /* previous search direction will be saved */
          SubSkip    = 0;    /* number of iterations to skip checking memory*/
          SubCheck   = mem * Parm->SubCheck; /* number of iterations to check */
          StartCheck = 0;         /* start checking memory at iteration 0 */
          InvariantSpace = FALSE; /* iterations not in invariant space */
          FirstFull      = TRUE;  /* first iteration in full space */
          nsub           = 0;     /* initial subspace dimension */
          memsq          = mem * mem;
          SkF            = gtemp + n; /* directions in memory (x_k+1 - x_k) */
          stemp          = SkF + mem * n; /* stores x_k+1 - x_k */
          gkeep = stemp + n;  /* store gradient when first direction != -g */
          Sk    = gkeep + n;  /* Sk = Rk at start of LBFGS in subspace */
          Rk    = Sk + memsq; /* upper triangular factor in SkF = Zk*Rk */
          /* zero out Rk to ensure lower triangle is 0 */
          cg_init(Rk, ZERO, memsq);
          Re   = Rk + memsq; /* end column of Rk, used for new direction */
          Yk   = Re + mem + 1;
          SkYk = Yk + memsq + mem + 2; /* dot products sk'yk in the subspace */
          tau  = SkYk + mem;           /* stores alpha in Nocedal and Wright */
          dsub = tau + mem;            /* direction projection in subspace */
          gsub = dsub + mem;           /* gradient projection in subspace */
          gsubtemp = gsub + mem + 1;   /* new gsub before update */
          wsub     = gsubtemp + mem; /* mem+1 work array for triangular solve */
          vsub     = wsub + mem + 1; /* mem work array for triangular solve */
        }
    }

  /* abort when number of iterations reaches maxit */
  maxit = Parm->maxit;

  f         = ZERO;
  fbest     = INF;
  gbest     = INF;
  nslow     = 0;
  slowlimit = 2 * n + Parm->nslow;
  n5        = n % 5;

  Ck = ZERO;
  Qk = ZERO;

  /* initial function and gradient evaluations, initial direction */
  Com.alpha = ZERO;

  status = cg_evaluate("fg", "n", &Com);
  f      = Com.f;
  if (status)
    {
      if (PrintLevel > 0)
        printf("Function undefined at starting point\n");
      printf("RuntimeError: line %d ", __LINE__);
      printf("fvalue %e ->", f);

      goto Exit;
    }

  Com.f0        = f + f;
  Com.SmallCost = fabs(f) * Parm->SmallCost;
  xnorm         = cg_inf(x, n);

  /* set d = -g, compute gnorm  = infinity norm of g and
     gnorm2 = square of 2-norm of g */
  gnorm  = cg_update_inf2(g, g, d, &gnorm2, n);
  dnorm2 = gnorm2;

  /* check if the starting function value is nan */
  if (f != f)
    {
      status = 11;
      printf("RuntimeError: line %d ->", __LINE__);
      goto Exit;
    }

  if (Parm->StopRule)
    tol = MAX(gnorm * Parm->StopFac, grad_tol);
  else
    tol = grad_tol;
  Com.tol = tol;

  if (PrintLevel >= 1)
    {
      printf(
        "iter: %5i f: %13.6e gnorm: %13.6e memk: %i\n", (int)0, f, gnorm, memk);
    }

  if (cg_tol(gnorm, &Com))
    {
      iter   = 0;
      status = 0;
      goto Exit;
    }

  dphi0  = -gnorm2;
  delta2 = 2 * Parm->delta - ONE;
  alpha  = Parm->step;
  if (alpha == ZERO)
    {
      if (xnorm == ZERO)
        {
          if (f != ZERO)
            alpha = 2. * fabs(f) / gnorm2;
          else
            alpha = ONE;
        }
      else
        alpha = Parm->psi0 * xnorm / gnorm;
    }

  Com.df0 = -2.0 * fabs(f) / alpha;

  Restart     = FALSE; /* do not restart the algorithm */
  IterRestart = 0;     /* counts number of iterations since last restart */
  IterSub     = 0;     /* counts number of iterations in subspace */
  NumSub      = 0;     /* total number of subspaces */
  IterQuad    = 0;     /* counts number of iterations that function change
        is close to that of a quadratic */
  scale = (double)1;   /* scale is the initial approximation to inverse
        Hessian in LBFGS; after the initial iteration,
        scale is estimated by the BB formula */

  /* Start the conjugate gradient iteration.
     alpha starts as old step, ends as final step for current iteration
     f is function value for alpha = 0
     QuadOK = TRUE means that a quadratic step was taken */

  for (iter = 1; iter <= maxit; iter++)
    {
      /* save old alpha to simplify formula computing subspace direction */
      alphaold   = alpha;
      Com.QuadOK = FALSE;
      alpha      = Parm->psi2 * alpha;
      if (f != ZERO)
        t = fabs((f - Com.f0) / f);
      else
        t = ONE;
      Com.UseCubic = TRUE;
      if ((t < Parm->CubicCutOff) || !Parm->UseCubic)
        Com.UseCubic = FALSE;
      if (Parm->QuadStep)
        {
          /* test if quadratic interpolation step should be tried */
          if (((t > Parm->QuadCutOff) && (fabs(f) >= Com.SmallCost)) || QuadF)
            {
              if (QuadF)
                {
                  Com.alpha = Parm->psi1 * alpha;
                  status    = cg_evaluate("g", "y", &Com);
                  if (status)
                    {
                      printf("RuntimeError: line %d ->", __LINE__);
                      goto Exit;
                    }
                  if (Com.df > dphi0)
                    {
                      alpha      = -dphi0 / ((Com.df - dphi0) / Com.alpha);
                      Com.QuadOK = TRUE;
                    }
                  else if (LBFGS)
                    {
                      if (memk >= n)
                        {
                          alpha      = ONE;
                          Com.QuadOK = TRUE;
                        }
                      else
                        alpha = 2.;
                    }
                  else if (Subspace)
                    {
                      if (memk >= nsub)
                        {
                          alpha      = ONE;
                          Com.QuadOK = TRUE;
                        }
                      else
                        alpha = 2.;
                    }
                }
              else
                {
                  t         = MAX(Parm->psi_lo, Com.df0 / (dphi0 * Parm->psi2));
                  Com.alpha = MIN(t, Parm->psi_hi) * alpha;
                  status    = cg_evaluate("f", "y", &Com);
                  if (status)
                    {
                      printf("RuntimeError: line %d ->", __LINE__);
                      goto Exit;
                    }
                  ftemp = Com.f;
                  denom = 2. * (((ftemp - f) / Com.alpha) - dphi0);
                  if (denom > ZERO)
                    {
                      t = -dphi0 * Com.alpha / denom;
                      /* safeguard */
                      if (ftemp >= f)
                        alpha = MAX(t, Com.alpha * Parm->QuadSafe);
                      else
                        alpha = t;
                      Com.QuadOK = TRUE;
                    }
                }
              if (PrintLevel >= 1)
                {
                  if (denom <= ZERO)
                    {
                      printf("Quad step fails (denom = %14.6e)\n", denom);
                    }
                  else if (Com.QuadOK)
                    {
                      printf("Quad step %14.6e OK\n", alpha);
                    }
                  else
                    printf("Quad step %14.6e done, but not OK\n", alpha);
                }
            }
          else if (PrintLevel >= 1)
            {
              printf("No quad step (chg: %14.6e, cut: %10.2e)\n",
                     t,
                     Parm->QuadCutOff);
            }
        }
      Com.f0  = f; /* f0 saved as prior value */
      Com.df0 = dphi0;

      /* parameters in Wolfe and approximate Wolfe conditions, and in update*/

      Qk = Parm->Qdecay * Qk + ONE;
      Ck = Ck + (fabs(f) - Ck) / Qk; /* average cost magnitude */

      if (Com.PertRule)
        Com.fpert = f + Com.eps * fabs(f);
      else
        Com.fpert = f + Com.eps;

      Com.wolfe_hi  = Parm->delta * dphi0;
      Com.wolfe_lo  = Parm->sigma * dphi0;
      Com.awolfe_hi = delta2 * dphi0;
      Com.alpha     = alpha;

      /* perform line search */
      status = cg_line(&Com);

      /*try approximate Wolfe line search if ordinary Wolfe fails */
      if ((status > 0) && !Com.AWolfe)
        {
          if (PrintLevel >= 1)
            {
              printf("\nWOLFE LINE SEARCH FAILS\n");
            }
          if (status != 3)
            {
              Com.AWolfe = TRUE;
              status     = cg_line(&Com);
            }
        }

      alpha = Com.alpha;
      f     = Com.f;
      dphi  = Com.df;

      if (status)
        {
          printf("RuntimeError: line %d ->", __LINE__);
          goto Exit;
        }

      /* Test for convergence to within machine epsilon
         [set feps to zero to remove this test] */

      if (-alpha * dphi0 <= Parm->feps * fabs(f))
        {
          status = 1;
          goto Exit;
        }

      /* test how close the cost function changes are to that of a quadratic
         QuadTrust = 0 means the function change matches that of a quadratic*/
      t = alpha * (dphi + dphi0);
      if (fabs(t) <= Parm->qeps * MIN(Ck, ONE))
        QuadTrust = ZERO;
      else
        QuadTrust = fabs((2.0 * (f - Com.f0) / t) - ONE);
      if (QuadTrust <= Parm->qrule)
        IterQuad++;
      else
        IterQuad = 0;

      if (IterQuad == qrestart)
        QuadF = TRUE;
      IterRestart++;
      if (!Com.AWolfe)
        {
          if (fabs(f - Com.f0) < Parm->AWolfeFac * Ck)
            {
              Com.AWolfe = TRUE;
              if (Com.Wolfe)
                Restart = TRUE;
            }
        }

      if ((mem > 0) && !LBFGS)
        {
          if (UseMemory)
            {
              if ((iter - StartCheck > SubCheck) && !Subspace)
                {
                  StartSkip = iter;
                  UseMemory = FALSE;
                  if (SubSkip == 0)
                    SubSkip = mem * Parm->SubSkip;
                  else
                    SubSkip *= 2;
                  if (PrintLevel >= 1)
                    {
                      printf("skip subspace %i iterations\n", SubSkip);
                    }
                }
            }
          else
            {
              if (iter - StartSkip > SubSkip)
                {
                  StartCheck = iter;
                  UseMemory  = TRUE;
                  memk       = 0;
                }
            }
        }

      if (!UseMemory)
        {
          if (!LBFGS)
            {
              if ((IterRestart >= nrestart) ||
                  ((IterQuad == qrestart) && (IterQuad != IterRestart)))
                Restart = TRUE;
            }
        }
      else
        {
          if (Subspace) /* the iteration is in the subspace */
            {
              IterSubRestart++;

              /* compute project of g into subspace */
              gsubnorm2 = ZERO;
              mp        = SkFstart;
              j         = nsub - mp;

              /* multiply basis vectors by new gradient */
              cg_matvec(wsub, SkF, gtemp, nsub, n, 0);

              /* rearrange wsub and store in gsubtemp
                 (elements associated with old vectors should
                 precede elements associated with newer vectors */
              cg_copy0(gsubtemp, wsub + mp, j);
              cg_copy0(gsubtemp + j, wsub, mp);

              /* solve Rk'y = gsubtemp */
              cg_trisolve(gsubtemp, Rk, mem, nsub, 0);
              gsubnorm2 = cg_dot0(gsubtemp, gsubtemp, nsub);
              gnorm2    = cg_dot(gtemp, gtemp, n);
              ratio     = sqrt(gsubnorm2 / gnorm2);
              if (ratio < ONE - Parm->eta1) /* Exit Subspace */
                {
                  if (PrintLevel >= 1)
                    {
                      printf("iter: %i RuntimeError subspace\n", (int)iter);
                    }
                  FirstFull      = TRUE;  /* first iteration in full space */
                  Subspace       = FALSE; /* leave the subspace */
                  InvariantSpace = FALSE;
                  /* check the subspace condition for SubCheck iterations
                     starting from the current iteration (StartCheck) */
                  StartCheck = iter;
                  if (IterSubRestart > 1)
                    dnorm2 = cg_dot0(dsub, dsub, nsub);
                }
              else
                {
                  /* Check if a restart should be done in subspace */
                  if (IterSubRestart == nrestartsub)
                    Restart = TRUE;
                }
            }
          else /* in full space */
            {
              if ((IterRestart == 1) || FirstFull)
                memk = 0;
              if ((memk == 1) && InvariantSpace)
                {
                  memk           = 0;
                  InvariantSpace = FALSE;
                }
              if (memk < mem)
                {
                  memk_is_mem = FALSE;
                  SkFstart    = 0;
                  /* SkF stores basis vector of the form alpha*d
                     We factor SkF = Zk*Rk where Zk has orthonormal columns
                     and Rk is upper triangular. Zk is not stored; wherever
                     it is needed, we use SkF * inv (Rk) */
                  if (memk == 0)
                    {
                      mlast = 0; /* starting pointer in the memory */
                      memk  = 1; /* dimension of current subspace */

                      t     = sqrt(dnorm2);
                      zeta  = alpha * t;
                      Rk[0] = zeta;
                      cg_scale(SkF, d, alpha, n);
                      Yk[0]     = (dphi - dphi0) / t;
                      gsub[0]   = dphi / t;
                      SkYk[0]   = alpha * (dphi - dphi0);
                      FirstFull = FALSE;
                      if (IterRestart > 1)
                        {
                          /* Need to save g for later correction of first
                             column of Yk. Since g does not lie in the
                             subspace and the first column is dense */
                          cg_copy(gkeep, g, n);
                          /* Also store dot product of g with the first
                             direction vector -- this saves a later dot
                             product when we fix the first column of Yk */
                          stgkeep = dphi0 * alpha;
                          d0isg   = FALSE;
                        }
                      else
                        d0isg = TRUE;
                    }
                  else
                    {
                      mlast = memk; /* starting pointer in the memory */
                      memk++;       /* total number of Rk in the memory */
                      mpp = mlast * n;
                      spp = mlast * mem;
                      cg_scale(SkF + mpp, d, alpha, n);

                      /* check if the alphas are far from 1 */
                      if ((fabs(alpha - 5.05) > 4.95) ||
                          (fabs(alphaold - 5.05) > 4.95))
                        {
                          /* multiply basis vectors by new direction vector */
                          cg_matvec(Rk + spp, SkF, SkF + mpp, mlast, n, 0);

                          /* solve Rk'y = wsub to obtain the components of the
                             new direction vector relative to the orthonormal
                             basis Z in S = ZR, store in next column of Rk */
                          cg_trisolve(Rk + spp, Rk, mem, mlast, 0);
                        }
                      else /* alphas are close to 1 */
                        {
                          t1 = -alpha;
                          t2 = beta * alpha / alphaold;
                          for (j = 0; j < mlast; j++)
                            {
                              Rk[spp + j] =
                                t1 * gsub[j] + t2 * Rk[spp - mem + j];
                            }
                        }
                      t  = alpha * alpha * dnorm2;
                      t1 = cg_dot0(Rk + spp, Rk + spp, mlast);
                      if (t <= t1)
                        {
                          zeta    = t * 1.e-12;
                          NegDiag = TRUE;
                        }
                      else
                        zeta = sqrt(t - t1);

                      Rk[spp + mlast] = zeta;
                      t = -zeta / alpha; /* t = cg_dot0 (Zk+mlast*n, g, n)*/
                      Yk[spp - mem + mlast] = t;
                      gsub[mlast]           = t;

                      /* multiply basis vectors by new gradient */
                      cg_matvec(wsub, SkF, gtemp, mlast, n, 0);
                      /* exploit dphi for last multiply */
                      wsub[mlast] = alpha * dphi;
                      /* solve for new gsub */
                      cg_trisolve(wsub, Rk, mem, memk, 0);
                      /* subtract old gsub from new gsub = column of Yk */
                      cg_Yk(Yk + spp, gsub, wsub, NULL, memk);

                      SkYk[mlast] = alpha * (dphi - dphi0);
                    }
                }
              else /* memk = mem */
                {
                  memk_is_mem = TRUE;
                  mlast       = mem - 1;
                  cg_scale(stemp, d, alpha, n);
                  /* compute projection of s_k = alpha_k d_k into subspace
                     check if the alphas are far from 1 */
                  if ((fabs(alpha - 5.05) > 4.95) ||
                      (fabs(alphaold - 5.05) > 4.95))
                    {
                      mp = SkFstart;
                      j  = mem - mp;

                      /* multiply basis vectors by sk */
                      cg_matvec(wsub, SkF, stemp, mem, n, 0);
                      /* rearrange wsub and store in Re = end col Rk */
                      cg_copy0(Re, wsub + mp, j);
                      cg_copy0(Re + j, wsub, mp);

                      /* solve Rk'y = Re */
                      cg_trisolve(Re, Rk, mem, mem, 0);
                    }
                  else /* alphas close to 1 */
                    {
                      t1 = -alpha;
                      t2 = beta * alpha / alphaold;
                      for (j = 0; j < mem; j++)
                        {
                          Re[j] = t1 * gsub[j] + t2 * Re[j - mem];
                        }
                    }

                  /* t = 2-norm squared of s_k */
                  t = alpha * alpha * dnorm2;
                  /* t1 = 2-norm squared of projection */
                  t1 = cg_dot0(Re, Re, mem);
                  if (t <= t1)
                    {
                      zeta    = t * 1.e-12;
                      NegDiag = TRUE;
                    }
                  else
                    zeta = sqrt(t - t1);

                  /* dist from new search direction to prior subspace*/
                  Re[mem] = zeta;

                  /* projection of prior g on new orthogonal
                     subspace vector */
                  t         = -zeta / alpha; /* t = cg_dot(Zk+mpp, g, n)*/
                  gsub[mem] = t;
                  Yk[memsq] = t; /* also store it in Yk */

                  spp = memsq + 1;
                  mp  = SkFstart;
                  j   = mem - mp;

                  /* multiply basis vectors by gtemp */
                  cg_matvec(vsub, SkF, gtemp, mem, n, 0);

                  /* rearrange and store in wsub */
                  cg_copy0(wsub, vsub + mp, j);
                  cg_copy0(wsub + j, vsub, mp);

                  /* solve Rk'y = wsub */
                  cg_trisolve(wsub, Rk, mem, mem, 0);
                  wsub[mem] = (alpha * dphi - cg_dot0(wsub, Re, mem)) / zeta;

                  /* add new column to Yk, store new gsub */
                  cg_Yk(Yk + spp, gsub, wsub, NULL, mem + 1);

                  /* store sk (stemp) at SkF+SkFstart */
                  cg_copy(SkF + SkFstart * n, stemp, n);
                  SkFstart++;
                  if (SkFstart == mem)
                    SkFstart = 0;

                  mp = SkFstart;
                  for (k = 0; k < mem; k++)
                    {
                      spp = (k + 1) * mem + k;
                      t1  = Rk[spp];
                      t2  = Rk[spp + 1];
                      t   = sqrt(t1 * t1 + t2 * t2);
                      t1  = t1 / t;
                      t2  = t2 / t;

                      /* update Rk */
                      Rk[k * mem + k] = t;
                      for (j = (k + 2); j <= mem; j++)
                        {
                          spp1        = spp;
                          spp         = j * mem + k;
                          t3          = Rk[spp];
                          t4          = Rk[spp + 1];
                          Rk[spp1]    = t1 * t3 + t2 * t4;
                          Rk[spp + 1] = t1 * t4 - t2 * t3;
                        }
                      /* update Yk */
                      if (k < 2) /* mem should be greater than 2 */
                        {
                          /* first 2 rows are dense */
                          spp = k;
                          for (j = 1; j < mem; j++)
                            {
                              spp1        = spp;
                              spp         = j * mem + k;
                              t3          = Yk[spp];
                              t4          = Yk[spp + 1];
                              Yk[spp1]    = t1 * t3 + t2 * t4;
                              Yk[spp + 1] = t1 * t4 - t2 * t3;
                            }
                          spp1        = spp;
                          spp         = mem * mem + 1 + k;
                          t3          = Yk[spp];
                          t4          = Yk[spp + 1];
                          Yk[spp1]    = t1 * t3 + t2 * t4;
                          Yk[spp + 1] = t1 * t4 - t2 * t3;
                        }
                      else if ((k == 2) && (2 < mem - 1))
                        {
                          spp = k;

                          /* col 1 dense since the oldest direction
                             vector has been dropped */
                          j    = 1;
                          spp1 = spp;
                          spp  = j * mem + k;
                          /* single nonzero percolates down the column */
                          t3          = Yk[spp]; /* t4 = 0. */
                          Yk[spp1]    = t1 * t3;
                          Yk[spp + 1] = -t2 * t3;
                          /* process rows in Hessenberg part of matrix */
                          for (j = 2; j < mem; j++)
                            {
                              spp1        = spp;
                              spp         = j * mem + k;
                              t3          = Yk[spp];
                              t4          = Yk[spp + 1];
                              Yk[spp1]    = t1 * t3 + t2 * t4;
                              Yk[spp + 1] = t1 * t4 - t2 * t3;
                            }
                          spp1        = spp;
                          spp         = mem * mem + 1 + k;
                          t3          = Yk[spp];
                          t4          = Yk[spp + 1];
                          Yk[spp1]    = t1 * t3 + t2 * t4;
                          Yk[spp + 1] = t1 * t4 - t2 * t3;
                        }
                      else if (k < (mem - 1))
                        {
                          spp = k;

                          /* process first column */
                          j           = 1;
                          spp1        = spp;
                          spp         = j * mem + k;
                          t3          = Yk[spp]; /* t4 = 0. */
                          Yk[spp1]    = t1 * t3;
                          Yk[spp + 1] = -t2 * t3;

                          /* process rows in Hessenberg part of matrix */
                          j        = k - 1;
                          spp      = (j - 1) * mem + k;
                          spp1     = spp;
                          spp      = j * mem + k;
                          t3       = Yk[spp];
                          Yk[spp1] = t1 * t3; /* t4 = 0. */
                          /* Yk [spp+1] = -t2*t3 ;*/
                          /* Theoretically this element is zero */
                          for (j = k; j < mem; j++)
                            {
                              spp1        = spp;
                              spp         = j * mem + k;
                              t3          = Yk[spp];
                              t4          = Yk[spp + 1];
                              Yk[spp1]    = t1 * t3 + t2 * t4;
                              Yk[spp + 1] = t1 * t4 - t2 * t3;
                            }
                          spp1        = spp;
                          spp         = mem * mem + 1 + k;
                          t3          = Yk[spp];
                          t4          = Yk[spp + 1];
                          Yk[spp1]    = t1 * t3 + t2 * t4;
                          Yk[spp + 1] = t1 * t4 - t2 * t3;
                        }
                      else /* k = mem-1 */
                        {
                          spp = k;

                          /* process first column */
                          j        = 1;
                          spp1     = spp;
                          spp      = j * mem + k;
                          t3       = Yk[spp]; /* t4 = 0. */
                          Yk[spp1] = t1 * t3;

                          /* process rows in Hessenberg part of matrix */
                          j        = k - 1;
                          spp      = (j - 1) * mem + k;
                          spp1     = spp;
                          spp      = j * mem + k;
                          t3       = Yk[spp]; /* t4 = 0. */
                          Yk[spp1] = t1 * t3;

                          j        = k;
                          spp1     = spp;
                          spp      = j * mem + k; /* j=mem-1 */
                          t3       = Yk[spp];
                          t4       = Yk[spp + 1];
                          Yk[spp1] = t1 * t3 + t2 * t4;

                          spp1     = spp;
                          spp      = mem * mem + 1 + k; /* j=mem */
                          t3       = Yk[spp];
                          t4       = Yk[spp + 1];
                          Yk[spp1] = t1 * t3 + t2 * t4;
                        }
                      /* update g in subspace */
                      if (k < (mem - 1))
                        {
                          t3          = gsub[k];
                          t4          = gsub[k + 1];
                          gsub[k]     = t1 * t3 + t2 * t4;
                          gsub[k + 1] = t1 * t4 - t2 * t3;
                        }
                      else /* k = mem-1 */
                        {
                          t3      = gsub[k];
                          t4      = gsub[k + 1];
                          gsub[k] = t1 * t3 + t2 * t4;
                        }
                    }

                  /* update SkYk */
                  for (k = 0; k < mlast; k++)
                    SkYk[k] = SkYk[k + 1];
                  SkYk[mlast] = alpha * (dphi - dphi0);
                }

              /* calculate t = ||gsub|| / ||gtemp||  */
              gsubnorm2 = cg_dot0(gsub, gsub, memk);
              gnorm2    = cg_dot(gtemp, gtemp, n);
              ratio     = sqrt(gsubnorm2 / gnorm2);
              if (ratio > ONE - Parm->eta2)
                InvariantSpace = TRUE;

              /* check to see whether to enter subspace */
              if (((memk > 1) && InvariantSpace) ||
                  ((memk == mem) && (ratio > ONE - Parm->eta0)))
                {
                  NumSub++;
                  if (PrintLevel >= 1)
                    {
                      if (InvariantSpace)
                        {
                          printf("iter: %i invariant space, "
                                 "enter subspace\n",
                                 (int)iter);
                        }
                      else
                        {
                          printf("iter: %i enter subspace\n", (int)iter);
                        }
                    }
                  /* if the first column is dense, we need to correct it
                     now since we do not know the entries until the basis
                     is determined */
                  if (!d0isg && !memk_is_mem)
                    {
                      wsub[0] = stgkeep;
                      /* mlast = memk -1 */
                      cg_matvec(wsub + 1, SkF + n, gkeep, mlast, n, 0);
                      /* solve Rk'y = wsub */
                      cg_trisolve(wsub, Rk, mem, memk, 0);
                      /* corrected first column of Yk */
                      Yk[1] -= wsub[1];
                      cg_scale0(Yk + 2, wsub + 2, -ONE, memk - 2);
                    }
                  if (d0isg && !memk_is_mem)
                    DenseCol1 = FALSE;
                  else
                    DenseCol1 = TRUE;

                  Subspace = TRUE;
                  /* reset subspace skipping to 0, need to test invariance */
                  SubSkip        = 0;
                  IterSubRestart = 0;
                  IterSubStart   = IterSub;
                  nsub           = memk; /* dimension of subspace */
                  nrestartsub    = (int)(((double)nsub) * Parm->restart_fac);
                  mp_begin       = mlast;
                  memk_begin     = nsub;
                  SkFlast        = (SkFstart + nsub - 1) % mem;
                  cg_copy0(gsubtemp, gsub, nsub);
                  /* Rk contains the sk for subspace, initialize Sk = Rk */
                  cg_copy(Sk, Rk, (int)mem * nsub);
                }
              else
                {
                  if ((IterRestart == nrestart) ||
                      ((IterQuad == qrestart) && (IterQuad != IterRestart)))
                    {
                      Restart = TRUE;
                    }
                }
            } /* done checking the full space */
        }     /* done using the memory */

      /* compute search direction */
      if (LBFGS)
        {
          gnorm = cg_inf(gtemp, n);
          if (cg_tol(gnorm, &Com))
            {
              status = 0;
              cg_copy(x, xtemp, n);
              goto Exit;
            }

          if (IterRestart == nrestart) /* restart the l-bfgs method */
            {
              IterRestart = 0;
              IterQuad    = 0;
              mlast       = -1;
              memk        = 0;
              scale       = (double)1;

              /* copy xtemp to x */
              cg_copy(x, xtemp, n);

              /* set g = gtemp, d = -g, compute 2-norm of g */
              gnorm2 = cg_update_2(g, gtemp, d, n);

              dnorm2 = gnorm2;
              dphi0  = -gnorm2;
            }
          else
            {
              mlast = (mlast + 1) % mem;
              spp   = mlast * n;
              cg_step(Sk + spp, xtemp, x, -ONE, n);
              cg_step(Yk + spp, gtemp, g, -ONE, n);
              SkYk[mlast] = alpha * (dphi - dphi0);
              if (memk < mem)
                memk++;

              /* copy xtemp to x */
              cg_copy(x, xtemp, n);

              /* copy gtemp to g and compute 2-norm of g */
              gnorm2 = cg_update_2(g, gtemp, NULL, n);

              /* calculate Hg = H g, saved in gtemp */
              mp = mlast; /* memk is the number of vectors in the memory */
              for (j = 0; j < memk; j++)
                {
                  mpp     = mp * n;
                  t       = cg_dot(Sk + mpp, gtemp, n) / SkYk[mp];
                  tau[mp] = t;
                  cg_daxpy(gtemp, Yk + mpp, -t, n);
                  mp -= 1;
                  if (mp < 0)
                    mp = mem - 1;
                }
              /* scale = (alpha*dnorm2)/(dphi-dphi0) ; */
              t = cg_dot(Yk + mlast * n, Yk + mlast * n, n);
              if (t > ZERO)
                {
                  scale = SkYk[mlast] / t;
                }

              cg_scale(gtemp, gtemp, scale, n);

              for (j = 0; j < memk; j++)
                {
                  mp += 1;
                  if (mp == mem)
                    mp = 0;
                  mpp = mp * n;
                  t   = cg_dot(Yk + mpp, gtemp, n) / SkYk[mp];
                  cg_daxpy(gtemp, Sk + mpp, tau[mp] - t, n);
                }

              /* set d = -gtemp, compute 2-norm of gtemp */
              dnorm2 = cg_update_2(NULL, gtemp, d, n);
              dphi0  = -cg_dot(g, gtemp, n);
            }
        } /* end of LBFGS */

      else if (Subspace) /* compute search direction in subspace */
        {
          IterSub++;

          /* set x = xtemp */
          cg_copy(x, xtemp, n);
          /* set g = gtemp and compute infinity norm of g */
          gnorm = cg_update_inf(g, gtemp, NULL, n);

          if (cg_tol(gnorm, &Com))
            {
              status = 0;
              goto Exit;
            }

          if (Restart) /*restart in subspace*/
            {
              scale          = (double)1;
              Restart        = FALSE;
              IterRestart    = 0;
              IterSubRestart = 0;
              IterQuad       = 0;
              mp_begin       = -1;
              memk_begin     = 0;
              memk           = 0;

              if (PrintLevel >= 1)
                printf("RESTART Sub-CG\n");

              /* search direction d = -Zk gsub, gsub = Zk' g, dsub = -gsub
                 => d =  Zk dsub = SkF (Rk)^{-1} dsub */
              cg_scale0(dsub, gsubtemp, -ONE, nsub);
              cg_copy0(gsub, gsubtemp, nsub);
              cg_copy0(vsub, dsub, nsub);
              cg_trisolve(vsub, Rk, mem, nsub, 1);
              /* rearrange and store in wsub */
              mp = SkFlast;
              j  = nsub - (mp + 1);
              cg_copy0(wsub, vsub + j, mp + 1);
              cg_copy0(wsub + (mp + 1), vsub, j);
              cg_matvec(d, SkF, wsub, nsub, n, 1);

              dphi0  = -gsubnorm2; /* gsubnorm2 was calculated before */
              dnorm2 = gsubnorm2;
            }
          else /* continue in subspace without restart */
            {
              mlast_sub = (mp_begin + IterSubRestart) % mem;

              if (IterSubRestart > 0) /* not first iteration in subspace  */
                {
                  /* add new column to Yk memory,
                     calculate yty, Sk, Yk and SkYk */
                  spp = mlast_sub * mem;
                  cg_scale0(Sk + spp, dsub, alpha, nsub);
                  /* yty = (gsubtemp-gsub)'(gsubtemp-gsub),
                     set gsub = gsubtemp */
                  cg_Yk(Yk + spp, gsub, gsubtemp, &yty, nsub);
                  SkYk[mlast_sub] = alpha * (dphi - dphi0);
                  if (yty > ZERO)
                    {
                      scale = SkYk[mlast_sub] / yty;
                    }
                }
              else
                {
                  yty =
                    cg_dot0(Yk + mlast_sub * mem, Yk + mlast_sub * mem, nsub);
                  if (yty > ZERO)
                    {
                      scale = SkYk[mlast_sub] / yty;
                    }
                }

              /* calculate gsubtemp = H gsub */
              mp = mlast_sub;
              /* memk = size of the L-BFGS memory in subspace */
              memk = MIN(memk_begin + IterSubRestart, mem);
              l1   = MIN(IterSubRestart, memk);
              /* l2 = number of triangular columns in Yk with a zero */
              l2 = memk - l1;
              /* l1 = number of dense column in Yk (excluding first) */
              l1++;
              l1 = MIN(l1, memk);

              /* process dense columns */
              for (j = 0; j < l1; j++)
                {
                  mpp     = mp * mem;
                  t       = cg_dot0(Sk + mpp, gsubtemp, nsub) / SkYk[mp];
                  tau[mp] = t;
                  /* update gsubtemp -= t*Yk+mpp */
                  cg_daxpy0(gsubtemp, Yk + mpp, -t, nsub);
                  mp--;
                  if (mp < 0)
                    mp = mem - 1;
                }

              /* process columns from triangular (Hessenberg) matrix */
              for (j = 1; j < l2; j++)
                {
                  mpp     = mp * mem;
                  t       = cg_dot0(Sk + mpp, gsubtemp, mp + 1) / SkYk[mp];
                  tau[mp] = t;
                  /* update gsubtemp -= t*Yk+mpp */
                  if (mp == 0 && DenseCol1)
                    {
                      cg_daxpy0(gsubtemp, Yk + mpp, -t, nsub);
                    }
                  else
                    {
                      cg_daxpy0(gsubtemp, Yk + mpp, -t, MIN(mp + 2, nsub));
                    }
                  mp--;
                  if (mp < 0)
                    mp = mem - 1;
                }
              cg_scale0(gsubtemp, gsubtemp, scale, nsub);

              /* process columns from triangular (Hessenberg) matrix */
              for (j = 1; j < l2; j++)
                {
                  mp++;
                  if (mp == mem)
                    mp = 0;
                  mpp = mp * mem;
                  if (mp == 0 && DenseCol1)
                    {
                      t = cg_dot0(Yk + mpp, gsubtemp, nsub) / SkYk[mp];
                    }
                  else
                    {
                      t = cg_dot0(Yk + mpp, gsubtemp, MIN(mp + 2, nsub)) /
                          SkYk[mp];
                    }
                  /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                  cg_daxpy0(gsubtemp, Sk + mpp, tau[mp] - t, mp + 1);
                }

              /* process dense columns */
              for (j = 0; j < l1; j++)
                {
                  mp++;
                  if (mp == mem)
                    mp = 0;
                  mpp = mp * mem;
                  t   = cg_dot0(Yk + mpp, gsubtemp, nsub) / SkYk[mp];
                  /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                  cg_daxpy0(gsubtemp, Sk + mpp, tau[mp] - t, nsub);
                } /* done computing H gsubtemp */

              /* compute d = Zk dsub = SkF (Rk)^{-1} dsub */
              cg_scale0(dsub, gsubtemp, -ONE, nsub);
              cg_copy0(vsub, dsub, nsub);
              cg_trisolve(vsub, Rk, mem, nsub, 1);
              /* rearrange and store in wsub */
              mp = SkFlast;
              j  = nsub - (mp + 1);
              cg_copy0(wsub, vsub + j, mp + 1);
              cg_copy0(wsub + (mp + 1), vsub, j);

              cg_matvec(d, SkF, wsub, nsub, n, 1);
              dphi0 = -cg_dot0(gsubtemp, gsub, nsub);
            }
        }  /* end of subspace search direction */
      else /* compute the search direction in the full space */
        {
          if (Restart) /*restart in fullspace*/
            {
              Restart     = FALSE;
              IterRestart = 0;
              IterQuad    = 0;
              if (PrintLevel >= 1)
                printf("RESTART CG\n");

              /* set x = xtemp */
              cg_copy(x, xtemp, n);

              if (UseMemory)
                {
                  /* set g = gtemp, d = -g, compute infinity norm of g,
                     gnorm2 was already computed above */
                  gnorm = cg_update_inf(g, gtemp, d, n);
                }
              else
                {
                  /* set g = gtemp, d = -g, compute infinity and 2-norm of g*/
                  gnorm = cg_update_inf2(g, gtemp, d, &gnorm2, n);
                }

              if (cg_tol(gnorm, &Com))
                {
                  status = 0;
                  goto Exit;
                }
              dphi0  = -gnorm2;
              dnorm2 = gnorm2;
              beta   = ZERO;
            }
          else if (!FirstFull) /* normal fullspace step*/
            {
              /* set x = xtemp */
              cg_copy(x, xtemp, n);

              /* set g = gtemp, compute gnorm = infinity norm of g,
                 ykyk = ||gtemp-g||_2^2, and ykgk = (gtemp-g) dot gnew */
              gnorm = cg_update_ykyk(g, gtemp, &ykyk, &ykgk, n);

              if (cg_tol(gnorm, &Com))
                {
                  status = 0;
                  goto Exit;
                }

              dkyk = dphi - dphi0;
              if (Parm->AdaptiveBeta)
                t = 2. - ONE / (0.1 * QuadTrust + ONE);
              else
                t = Parm->theta;
              beta = (ykgk - t * dphi * ykyk / dkyk) / dkyk;

              /* faster: initialize dnorm2 = gnorm2 at start, then
                 dnorm2 = gnorm2 + beta**2*dnorm2 - 2.*beta*dphi
                 gnorm2 = ||g_{k+1}||^2
                 dnorm2 = ||d_{k+1}||^2
                 dpi = g_{k+1}' d_k */

              /* lower bound for beta is BetaLower*d_k'g_k/ ||d_k||^2 */
              beta = MAX(beta, Parm->BetaLower * dphi0 / dnorm2);

              /* update search direction d = -g + beta*dold */
              if (UseMemory)
                {
                  /* update search direction d = -g + beta*dold, and
                     compute 2-norm of d, 2-norm of g computed above */
                  dnorm2 = cg_update_d(d, g, beta, NULL, n);
                }
              else
                {
                  /* update search direction d = -g + beta*dold, and
                     compute 2-norms of d and g */
                  dnorm2 = cg_update_d(d, g, beta, &gnorm2, n);
                }

              dphi0 = -gnorm2 + beta * dphi;
              if (Parm->debug) /* Check that dphi0 = d'g */
                {
                  t = ZERO;
                  for (i = 0; i < n; i++)
                    t = t + d[i] * g[i];
                  if (fabs(t - dphi0) > Parm->debugtol * fabs(dphi0))
                    {
                      printf("Warning, dphi0 != d'g!\n");
                      printf("dphi0:%13.6e, d'g:%13.6e\n", dphi0, t);
                    }
                }
            }
          else /* FirstFull = TRUE, precondition after leaving subspace */
            {
              /* set x = xtemp */
              cg_copy(x, xtemp, n);

              /* set g = gtemp, compute gnorm = infinity norm of g,
                 ykyk = ||gtemp-g||_2^2, and ykgk = (gtemp-g) dot gnew */
              gnorm = cg_update_ykyk(g, gtemp, &ykyk, &ykgk, n);

              if (cg_tol(gnorm, &Com))
                {
                  status = 0;
                  goto Exit;
                }

              mlast_sub = (mp_begin + IterSubRestart) % mem;
              /* save Sk */
              spp = mlast_sub * mem;
              cg_scale0(Sk + spp, dsub, alpha, nsub);
              /* calculate yty, save Yk, set gsub = gsubtemp */
              cg_Yk(Yk + spp, gsub, gsubtemp, &yty, nsub);
              ytg             = cg_dot0(Yk + spp, gsub, nsub);
              t               = alpha * (dphi - dphi0);
              SkYk[mlast_sub] = t;

              /* scale = t/ykyk ; */
              if (yty > ZERO)
                {
                  scale = t / yty;
                }

              /* calculate gsubtemp = H gsub */
              mp = mlast_sub;
              /* memk = size of the L-BFGS memory in subspace */
              memk = MIN(memk_begin + IterSubRestart, mem);
              l1   = MIN(IterSubRestart, memk);
              /* l2 = number of triangular columns in Yk with a zero */
              l2 = memk - l1;
              /* l1 = number of dense column in Yk (excluding first) */
              l1++;
              l1 = MIN(l1, memk);

              /* process dense columns */
              for (j = 0; j < l1; j++)
                {
                  mpp     = mp * mem;
                  t       = cg_dot0(Sk + mpp, gsubtemp, nsub) / SkYk[mp];
                  tau[mp] = t;
                  /* update gsubtemp -= t*Yk+mpp */
                  cg_daxpy0(gsubtemp, Yk + mpp, -t, nsub);
                  mp--;
                  if (mp < 0)
                    mp = mem - 1;
                }

              /* process columns from triangular (Hessenberg) matrix */
              for (j = 1; j < l2; j++)
                {
                  mpp     = mp * mem;
                  t       = cg_dot0(Sk + mpp, gsubtemp, mp + 1) / SkYk[mp];
                  tau[mp] = t;
                  /* update gsubtemp -= t*Yk+mpp */
                  if (mp == 0 && DenseCol1)
                    {
                      cg_daxpy0(gsubtemp, Yk + mpp, -t, nsub);
                    }
                  else
                    {
                      cg_daxpy0(gsubtemp, Yk + mpp, -t, MIN(mp + 2, nsub));
                    }
                  mp--;
                  if (mp < 0)
                    mp = mem - 1;
                }
              cg_scale0(gsubtemp, gsubtemp, scale, nsub);

              /* process columns from triangular (Hessenberg) matrix */
              for (j = 1; j < l2; j++)
                {
                  mp++;
                  if (mp == mem)
                    mp = 0;
                  mpp = mp * mem;
                  if (mp == 0 && DenseCol1)
                    {
                      t = cg_dot0(Yk + mpp, gsubtemp, nsub) / SkYk[mp];
                    }
                  else
                    {
                      t = cg_dot0(Yk + mpp, gsubtemp, MIN(mp + 2, nsub)) /
                          SkYk[mp];
                    }
                  /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                  cg_daxpy0(gsubtemp, Sk + mpp, tau[mp] - t, mp + 1);
                }

              /* process dense columns */
              for (j = 0; j < l1; j++)
                {
                  mp++;
                  if (mp == mem)
                    mp = 0;
                  mpp = mp * mem;
                  t   = cg_dot0(Yk + mpp, gsubtemp, nsub) / SkYk[mp];
                  /* update gsubtemp += (tau[mp]-t)*Sk+mpp */
                  cg_daxpy0(gsubtemp, Sk + mpp, tau[mp] - t, nsub);
                } /* done computing H gsubtemp */

              /* compute beta */
              dkyk = dphi - dphi0;
              if (Parm->AdaptiveBeta)
                t = 2. - ONE / (0.1 * QuadTrust + ONE);
              else
                t = Parm->theta;
              t1 = MAX(ykyk - yty, ZERO); /* Theoretically t1 = ykyk-yty */
              if (ykyk > ZERO)
                {
                  scale = (alpha * dkyk) / ykyk; /* = sigma */
                }
              beta = scale * ((ykgk - ytg) - t * dphi * t1 / dkyk) / dkyk;
              /* beta = MAX (beta, Parm->BetaLower*dphi0/dnorm2) ; */
              beta = MAX(beta, Parm->BetaLower * (dphi0 * alpha) / dkyk);

              /* compute search direction
                 d = -Zk (H - sigma)ghat - sigma g + beta d

      Note: d currently contains last 2 terms so only need
      to add the Zk term. Above gsubtemp = H ghat */

              /* form vsub = sigma ghat - H ghat = sigma ghat - gsubtemp */
              cg_scale0(vsub, gsubtemp, -ONE, nsub);
              cg_daxpy0(vsub, gsub, scale, nsub);
              cg_trisolve(vsub, Rk, mem, nsub, 1);

              /* rearrange vsub and store in wsub */
              mp = SkFlast;
              j  = nsub - (mp + 1);
              cg_copy0(wsub, vsub + j, mp + 1);
              cg_copy0(wsub + (mp + 1), vsub, j);


              /* save old direction d in gtemp */
              cg_copy(gtemp, d, n);

              /* d = Zk (sigma - H)ghat */
              cg_matvec(d, SkF, wsub, nsub, n, 1);

              /* incorporate the new g and old d terms in new d */
              cg_daxpy(d, g, -scale, n);
              cg_daxpy(d, gtemp, beta, n);

              gHg   = cg_dot0(gsubtemp, gsub, nsub);
              t1    = MAX(gnorm2 - gsubnorm2, ZERO);
              dphi0 = -gHg - scale * t1 + beta * dphi;
              /* dphi0 = cg_dot (d, g, n) could be inaccurate */
              dnorm2 = cg_dot(d, d, n);
            } /* end of preconditioned step */
        }     /* search direction has been computed */

      /* test for slow convergence */
      if ((f < fbest) || (gnorm2 < gbest))
        {
          nslow = 0;
          if (f < fbest)
            fbest = f;
          if (gnorm2 < gbest)
            gbest = gnorm2;
        }
      else
        nslow++;
      if (nslow > slowlimit)
        {
          status = 9;
          goto Exit;
        }

      if (PrintLevel >= 1)
        {
          printf("\niter: %5i f = %13.6e gnorm = %13.6e memk: %i "
                 "Subspace: %i\n",
                 (int)iter,
                 f,
                 gnorm,
                 memk,
                 Subspace);
        }

      if (Parm->debug)
        {
          if (f > Com.f0 + Parm->debugtol * Ck)
            {
              status = 8;
              goto Exit;
            }
        }

      if (dphi0 > ZERO)
        {
          status = 5;
          goto Exit;
        }

      if ((*user_test)(f, x, g, n, user_data))
        {
          status = 0;
          goto Exit;
        }
    }
  status = 2;
Exit:
  if (status == 11)
    {
      printf("RuntimeError: line %d ->", __LINE__);
      gnorm = INF; /* function is undefined */
    }
  if (Stat != NULL)
    {
      Stat->nfunc   = Com.nf;
      Stat->ngrad   = Com.ng;
      Stat->iter    = iter;
      Stat->NumSub  = NumSub;
      Stat->IterSub = IterSub;
      if (status < 10) /* function was evaluated */
        {
          Stat->f     = f;
          Stat->gnorm = gnorm;
        }
    }
  /* If there was an error, the function was evaluated, and its value
     is defined, then copy the most recent x value to the returned x
     array and evaluate the norm of the gradient at this point */
  if ((status > 0) && (status < 10))
    {
      cg_copy(x, xtemp, n);
      gnorm = ZERO;
      for (i = 0; i < n; i++)
        {
          g[i]  = gtemp[i];
          t     = fabs(g[i]);
          gnorm = MAX(gnorm, t);
        }
      if (Stat != NULL)
        Stat->gnorm = gnorm;
    }
  if (Parm->PrintFinal || PrintLevel >= 1)
    {
      const char mess1[] = "Possible causes of this error message:";
      const char mess2[] = "   - your tolerance may be too strict: "
                           "grad_tol = ";
      const char mess3[] = "Line search fails";
      const char mess4[] = "   - your gradient routine has an error";
      const char mess5[] = "   - the parameter epsilon is too small";

      printf("\nTermination status: %i\n", status);

      if (status && NegDiag)
        {
          printf("Parameter eta2 may be too small\n");
        }

      if (status == 0)
        {
          printf("Convergence tolerance for gradient satisfied\n\n");
        }
      else if (status == 1)
        {
          printf("Terminating since change in function value "
                 "<= feps*|f|\n\n");
        }
      else if (status == 2)
        {
          printf("Number of iterations exceed specified limit\n");
          printf("Iterations: %10.0f maxit: %10.0f\n",
                 (double)iter,
                 (double)maxit);
          printf("%s\n", mess1);
          printf("%s %e\n\n", mess2, grad_tol);
        }
      else if (status == 3)
        {
          printf("Slope always negative in line search\n");
          printf("%s\n", mess1);
          printf("   - your cost function has an error\n");
          printf("%s\n\n", mess4);
        }
      else if (status == 4)
        {
          printf("Line search fails, too many iterations\n");
          printf("%s\n", mess1);
          printf("%s %e\n\n", mess2, grad_tol);
        }
      else if (status == 5)
        {
          printf("Search direction not a descent direction\n\n");
        }
      else if (status == 6) /* line search fails, excessive eps updating */
        {
          printf("%s due to excessive updating of eps\n", mess3);
          printf("%s\n", mess1);
          printf("%s %e\n", mess2, grad_tol);
          printf("%s\n\n", mess4);
        }
      else if (status == 7) /* line search fails */
        {
          printf("%s\n%s\n", mess3, mess1);
          printf("%s %e\n", mess2, grad_tol);
          printf("%s\n%s\n\n", mess4, mess5);
        }
      else if (status == 8)
        {
          printf("Debugger is on, function value does not improve\n");
          printf("new value: %25.16e old value: %25.16e\n\n", f, Com.f0);
        }
      else if (status == 9)
        {
          printf("%i iterations without strict improvement in cost "
                 "or gradient\n\n",
                 nslow);
        }
      else if (status == 10)
        {
          printf("Insufficient memory for specified problem dimension %e"
                 " in cg_descent\n",
                 (double)n);
        }
      else if (status == 11)
        {
          printf("Function nan and could not be repaired\n\n");
        }
      else if (status == 12)
        {
          printf("memory = %i is an invalid choice for parameter memory\n",
                 Parm->memory);
          printf("memory should be either 0 or greater than 2\n\n");
        }

      printf("maximum norm for gradient: %13.6e\n", gnorm);
      printf("function value:            %13.6e\n\n", f);
      printf("iterations:              %10.0f\n", (double)iter);
      printf("function evaluations:    %10.0f\n", (double)Com.nf);
      printf("gradient evaluations:    %10.0f\n", (double)Com.ng);
      if (IterSub > 0)
        {
          printf("subspace iterations:     %10.0f\n", (double)IterSub);
          printf("number of subspaces:     %10.0f\n", (double)NumSub);
        }
      printf("===================================\n\n");
    }
  if (Work == NULL)
    free(work);
  return (status);
}

/* =========================================================================
   ==== cg_Wolfe ===========================================================
   =========================================================================
   Check whether the Wolfe or the approximate Wolfe conditions are satisfied
   ========================================================================= */
PRIVATE int
cg_Wolfe(double  alpha, /* stepsize */
         double  f,     /* function value associated with stepsize alpha */
         double  dphi,  /* derivative value associated with stepsize alpha */
         cg_com *Com    /* cg com */
)
{
  if (dphi >= Com->wolfe_lo)
    {
      /* test original Wolfe conditions */
      if (f - Com->f0 <= alpha * Com->wolfe_hi)
        {
          if (Com->Parm->PrintLevel >= 2)
            {
              printf("Wolfe conditions hold\n");
              /*              printf ("wolfe f: %25.15e f0: %25.15e df:
                 %25.15e\n", f, Com->f0, dphi) ;*/
            }
          return (1);
        }
      /* test approximate Wolfe conditions */
      else if (Com->AWolfe)
        {
          /*          if ( Com->Parm->PrintLevel >= 2 )
                {
                printf ("f:    %e fpert:    %e ", f, Com->fpert) ;
                if ( f > Com->fpert ) printf ("(fail)\n") ;
                else                  printf ("(OK)\n") ;
                printf ("dphi: %e hi bound: %e ", dphi, Com->awolfe_hi) ;
                if ( dphi > Com->awolfe_hi ) printf ("(fail)\n") ;
                else                         printf ("(OK)\n") ;
                }*/
          if ((f <= Com->fpert) && (dphi <= Com->awolfe_hi))
            {
              if (Com->Parm->PrintLevel >= 2)
                {
                  printf("Approximate Wolfe conditions hold\n");
                  /*                  printf ("f: %25.15e fpert: %25.15e dphi:
                     %25.15e awolf_hi: "
                          "%25.15e\n", f, Com->fpert, dphi, Com->awolfe_hi) ;*/
                }
              return (1);
            }
        }
    }
  /*  else if ( Com->Parm->PrintLevel >= 2 )
      {
      printf ("dphi: %e lo bound: %e (fail)\n", dphi, Com->wolfe_lo) ;
      }*/
  return (0);
}

/* =========================================================================
   ==== cg_tol =============================================================
   =========================================================================
   Check for convergence
   ========================================================================= */
PRIVATE int
cg_tol(double  gnorm, /* gradient sup-norm */
       cg_com *Com    /* cg com */
)
{
  /* StopRule = T => |grad|_infty <=max (tol, |grad|_infty*StopFact)
     F => |grad|_infty <= tol*(1+|f|)) */
  if (Com->Parm->StopRule)
    {
      if (gnorm <= Com->tol)
        return (1);
    }
  else if (gnorm <= Com->tol * (ONE + fabs(Com->f)))
    return (1);
  return (0);
}

/* =========================================================================
   ==== cg_line ============================================================
   =========================================================================
   Approximate Wolfe line search routine
Return:
-2 (function nan)
0 (Wolfe or approximate Wolfe conditions satisfied)
3 (slope always negative in line search)
4 (number line search iterations exceed nline)
6 (excessive updating of eps)
7 (Wolfe conditions never satisfied)
========================================================================= */
PRIVATE int
cg_line(cg_com *Com /* cg com structure */
)
{
  int    AWolfe, iter, ngrow, PrintLevel, qb, qb0, status, toggle;
  double alpha, a, a1, a2, b, bmin, B, da, db, d0, d1, d2, dB, df, f, fa, fb,
    fB, a0, b0, da0, db0, fa0, fb0, width, rho;
  char *        s1, *s2, *fmt1, *fmt2;
  cg_parameter *Parm;

  AWolfe     = Com->AWolfe;
  Parm       = Com->Parm;
  PrintLevel = Parm->PrintLevel;
  if (PrintLevel >= 1)
    {
      if (AWolfe)
        {
          printf("Approximate Wolfe line search\n");
          printf("=============================\n");
        }
      else
        {
          printf("Wolfe line search\n");
          printf("=================\n");
        }
    }

  /* evaluate function or gradient at Com->alpha (starting guess) */
  if (Com->QuadOK)
    {
      status = cg_evaluate("fg", "y", Com);
      fb     = Com->f;
      if (!AWolfe)
        fb -= Com->alpha * Com->wolfe_hi;
      qb = TRUE; /* function value at b known */
    }
  else
    {
      status = cg_evaluate("g", "y", Com);
      qb     = FALSE;
    }
  if (status)
    {
      printf("RuntimeError: line %d ->", __LINE__);
      return (status); /* function is undefined */
    }
  b = Com->alpha;

  if (AWolfe)
    {
      db = Com->df;
      d0 = da = Com->df0;
    }
  else
    {
      db = Com->df - Com->wolfe_hi;
      d0 = da = Com->df0 - Com->wolfe_hi;
    }
  a  = ZERO;
  a1 = ZERO;
  d1 = d0;
  fa = Com->f0;
  if (PrintLevel >= 1)
    {
      fmt1 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb: %13.6e "
             "da: %13.6e db: %13.6e\n";
      fmt2 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb:  x.xxxxxxxxxx "
             "da: %13.6e db: %13.6e\n";
      if (Com->QuadOK)
        s2 = "OK";
      else
        s2 = "";
      if (qb)
        printf(fmt1, "start    ", s2, a, b, fa, fb, da, db);
      else
        printf(fmt2, "start    ", s2, a, b, fa, da, db);
    }

  /* if a quadratic interpolation step performed, check Wolfe conditions */
  if ((Com->QuadOK) && (Com->f <= Com->f0))
    {
      if (cg_Wolfe(b, Com->f, Com->df, Com))
        return (0);
    }

  /* if a Wolfe line search and the Wolfe conditions have not been satisfied*/
  if (!AWolfe)
    Com->Wolfe = TRUE;

  /*Find initial interval [a,b] such that
    da <= 0, db >= 0, fa <= fpert = [(f0 + eps*fabs (f0)) or (f0 + eps)] */
  rho   = Com->rho;
  ngrow = 1;
  while (db < ZERO)
    {
      if (!qb)
        {
          status = cg_evaluate("f", "n", Com);
          if (status)
            {
              printf("RuntimeError: line %d ->", __LINE__);
              return (status);
            }
          if (AWolfe)
            fb = Com->f;
          else
            fb = Com->f - b * Com->wolfe_hi;
          qb = TRUE;
        }
      if (fb > Com->fpert) /* contract interval [a, b] */
        {
          status = cg_contract(&a, &fa, &da, &b, &fb, &db, Com);
          if (status == 0)
            return (0); /* Wolfe conditions hold */
          if (status == -2)
            goto Line; /* db >= 0 */
          if (Com->neps > Parm->neps)
            return (6);
        }

      /* expansion phase */
      ngrow++;
      if (ngrow > Parm->ntries)
        return (3);
      /* update interval (a replaced by b) */
      a  = b;
      fa = fb;
      da = db;
      /* store old values of a and corresponding derivative */
      d2 = d1;
      d1 = da;
      a2 = a1;
      a1 = a;

      bmin = rho * b;
      if ((ngrow == 2) || (ngrow == 3) || (ngrow == 6))
        {
          if (d1 > d2)
            {
              if (ngrow == 2)
                {
                  b = a1 - (a1 - a2) * (d1 / (d1 - d2));
                }
              else
                {
                  if ((d1 - d2) / (a1 - a2) >= (d2 - d0) / a2)
                    {
                      /* convex derivative, secant overestimates minimizer */
                      b = a1 - (a1 - a2) * (d1 / (d1 - d2));
                    }
                  else
                    {
                      /* concave derivative, secant underestimates minimizer*/
                      b = a1 - Parm->SecantAmp * (a1 - a2) * (d1 / (d1 - d2));
                    }
                }
              /* safeguard growth */
              b = MIN(b, Parm->ExpandSafe * a1);
            }
          else
            rho *= Parm->RhoGrow;
        }
      else
        rho *= Parm->RhoGrow;
      b             = MAX(bmin, b);
      Com->alphaold = Com->alpha;
      Com->alpha    = b;
      status        = cg_evaluate("g", "p", Com);
      if (status)
        {
          printf("RuntimeError: line %d ->", __LINE__);
          return (status);
        }
      b  = Com->alpha;
      qb = FALSE;
      if (AWolfe)
        db = Com->df;
      else
        db = Com->df - Com->wolfe_hi;
      if (PrintLevel >= 2)
        {
          if (Com->QuadOK)
            s2 = "OK";
          else
            s2 = "";
          printf(fmt2, "expand   ", s2, a, b, fa, da, db);
        }
    }

  /* we now have fa <= fpert, da >= 0, db <= 0 */
Line:
  toggle = 0;
  width  = b - a;
  qb0    = FALSE;
  for (iter = 0; iter < Parm->nline; iter++)
    {
      /* determine the next iterate */
      if ((toggle == 0) || ((toggle == 2) && ((b - a) <= width)))
        {
          Com->QuadOK = TRUE;
          if (Com->UseCubic && qb)
            {
              s1    = "cubic    ";
              alpha = cg_cubic(a, fa, da, b, fb, db);
              if (alpha < ZERO) /* use secant method */
                {
                  s1 = "secant   ";
                  if (-da < db)
                    alpha = a - (a - b) * (da / (da - db));
                  else if (da != db)
                    alpha = b - (a - b) * (db / (da - db));
                  else
                    alpha = -1.;
                }
            }
          else
            {
              s1 = "secant   ";
              if (-da < db)
                alpha = a - (a - b) * (da / (da - db));
              else if (da != db)
                alpha = b - (a - b) * (db / (da - db));
              else
                alpha = -1.;
            }
          width = Parm->gamma * (b - a);
        }
      else if (toggle == 1) /* iteration based on smallest value*/
        {
          Com->QuadOK = TRUE;
          if (Com->UseCubic)
            {
              s1 = "cubic    ";
              if (Com->alpha == a) /* a is most recent iterate */
                {
                  alpha = cg_cubic(a0, fa0, da0, a, fa, da);
                }
              else if (qb0) /* b is most recent iterate */
                {
                  alpha = cg_cubic(b, fb, db, b0, fb0, db0);
                }
              else
                alpha = -1.;

              /* if alpha no good, use cubic between a and b */
              if ((alpha <= a) || (alpha >= b))
                {
                  if (qb)
                    alpha = cg_cubic(a, fa, da, b, fb, db);
                  else
                    alpha = -1.;
                }

              /* if alpha still no good, use secant method */
              if (alpha < ZERO)
                {
                  s1 = "secant   ";
                  if (-da < db)
                    alpha = a - (a - b) * (da / (da - db));
                  else if (da != db)
                    alpha = b - (a - b) * (db / (da - db));
                  else
                    alpha = -1.;
                }
            }
          else /* ( use secant ) */
            {
              s1 = "secant   ";
              if ((Com->alpha == a) && (da > da0)) /* use a0 if possible */
                {
                  alpha = a - (a - a0) * (da / (da - da0));
                }
              else if (db < db0) /* use b0 if possible */
                {
                  alpha = b - (b - b0) * (db / (db - db0));
                }
              else /* secant based on a and b */
                {
                  if (-da < db)
                    alpha = a - (a - b) * (da / (da - db));
                  else if (da != db)
                    alpha = b - (a - b) * (db / (da - db));
                  else
                    alpha = -1.;
                }

              if ((alpha <= a) || (alpha >= b))
                {
                  if (-da < db)
                    alpha = a - (a - b) * (da / (da - db));
                  else if (da != db)
                    alpha = b - (a - b) * (db / (da - db));
                  else
                    alpha = -1.;
                }
            }
        }
      else
        {
          alpha       = .5 * (a + b); /* use bisection if b-a decays slowly */
          s1          = "bisection";
          Com->QuadOK = FALSE;
        }

      if ((alpha <= a) || (alpha >= b))
        {
          alpha = .5 * (a + b);
          s1    = "bisection";
          if ((alpha == a) || (alpha == b))
            return (7);
          Com->QuadOK = FALSE; /* bisection was used */
        }

      if (toggle == 0) /* save values for next iteration */
        {
          a0  = a;
          b0  = b;
          da0 = da;
          db0 = db;
          fa0 = fa;
          if (qb)
            {
              fb0 = fb;
              qb0 = TRUE;
            }
        }

      toggle++;
      if (toggle > 2)
        toggle = 0;

      Com->alpha = alpha;
      status     = cg_evaluate("fg", "n", Com);
      if (status)
        {
          printf("RuntimeError: line %d ->", __LINE__);
          return (status);
        }
      Com->alpha = alpha;
      f          = Com->f;
      df         = Com->df;
      if (Com->QuadOK)
        {
          if (cg_Wolfe(alpha, f, df, Com))
            {
              if (PrintLevel >= 2)
                {
                  printf("             a: %13.6e f: %13.6e df: %13.6e %1s\n",
                         alpha,
                         f,
                         df,
                         s1);
                }
              return (0);
            }
        }
      if (!AWolfe)
        {
          f -= alpha * Com->wolfe_hi;
          df -= Com->wolfe_hi;
        }
      if (df >= ZERO)
        {
          b  = alpha;
          fb = f;
          db = df;
          qb = TRUE;
        }
      else if (f <= Com->fpert)
        {
          a  = alpha;
          da = df;
          fa = f;
        }
      else
        {
          B = b;
          if (qb)
            fB = fb;
          dB = db;
          b  = alpha;
          fb = f;
          db = df;
          /* contract interval [a, alpha] */
          status = cg_contract(&a, &fa, &da, &b, &fb, &db, Com);
          if (status == 0)
            return (0);
          if (status == -1) /* eps reduced, use [a, b] = [alpha, b] */
            {
              if (Com->neps > Parm->neps)
                return (6);
              a  = b;
              fa = fb;
              da = db;
              b  = B;
              if (qb)
                fb = fB;
              db = dB;
            }
          else
            qb = TRUE;
        }
      if (PrintLevel >= 2)
        {
          if (Com->QuadOK)
            s2 = "OK";
          else
            s2 = "";
          if (!qb)
            printf(fmt2, s1, s2, a, b, fa, da, db);
          else
            printf(fmt1, s1, s2, a, b, fa, fb, da, db);
        }
    }
  return (4);
}

/* =========================================================================
   ==== cg_contract ========================================================
   =========================================================================
   The input for this routine is an interval [a, b] with the property that
   fa <= fpert, da >= 0, db >= 0, and fb >= fpert. The returned status is

   11  function or derivative not defined
   0  if the Wolfe conditions are satisfied
   -1  if a new value for eps is generated with the property that for the
   corresponding fpert, we have fb <= fpert
   -2  if a subinterval, also denoted [a, b], is generated with the property
   that fa <= fpert, da >= 0, and db <= 0

NOTE: The input arguments are unchanged when status = -1
========================================================================= */
PRIVATE int
cg_contract(double *A,  /* left side of bracketing interval */
            double *fA, /* function value at a */
            double *dA, /* derivative at a */
            double *B,  /* right side of bracketing interval */
            double *fB, /* function value at b */
            double *dB, /* derivative at b */
            cg_com *Com /* cg com structure */
)
{
  int    AWolfe, iter, PrintLevel, toggle, status;
  double a, alpha, b, old, da, db, df, d1, dold, f, fa, fb, f1, fold, t, width;
  char * s;
  cg_parameter *Parm;

  AWolfe     = Com->AWolfe;
  Parm       = Com->Parm;
  PrintLevel = Parm->PrintLevel;
  a          = *A;
  fa         = *fA;
  da         = *dA;
  b          = *B;
  fb         = *fB;
  db         = *dB;
  f1         = fb;
  d1         = db;
  toggle     = 0;
  width      = ZERO;
  for (iter = 0; iter < Parm->nshrink; iter++)
    {
      if ((toggle == 0) || ((toggle == 2) && ((b - a) <= width)))
        {
          /* cubic based on bracketing interval */
          alpha  = cg_cubic(a, fa, da, b, fb, db);
          toggle = 0;
          width  = Parm->gamma * (b - a);
          if (iter)
            Com->QuadOK = TRUE; /* at least 2 cubic iterations */
        }
      else if (toggle == 1)
        {
          Com->QuadOK = TRUE;
          /* cubic based on most recent iterate and smallest value */
          if (old < a) /* a is most recent iterate */
            {
              alpha = cg_cubic(a, fa, da, old, fold, dold);
            }
          else /* b is most recent iterate */
            {
              alpha = cg_cubic(a, fa, da, b, fb, db);
            }
        }
      else
        {
          alpha       = .5 * (a + b); /* use bisection if b-a decays slowly */
          Com->QuadOK = FALSE;
        }

      if ((alpha <= a) || (alpha >= b))
        {
          alpha       = .5 * (a + b);
          Com->QuadOK = FALSE; /* bisection was used */
        }

      toggle++;
      if (toggle > 2)
        toggle = 0;

      Com->alpha = alpha;
      status     = cg_evaluate("fg", "n", Com);
      if (status)
        {
          printf("RuntimeError: line %d ->", __LINE__);
          return (status);
        }
      f  = Com->f;
      df = Com->df;

      if (Com->QuadOK)
        {
          if (cg_Wolfe(alpha, f, df, Com))
            return (0);
        }
      if (!AWolfe)
        {
          f -= alpha * Com->wolfe_hi;
          df -= Com->wolfe_hi;
        }
      if (df >= ZERO)
        {
          *B  = alpha;
          *fB = f;
          *dB = df;
          *A  = a;
          *fA = fa;
          *dA = da;
          return (-2);
        }
      if (f <= Com->fpert) /* update a using alpha */
        {
          old  = a;
          a    = alpha;
          fold = fa;
          fa   = f;
          dold = da;
          da   = df;
        }
      else /* update b using alpha */
        {
          old = b;
          b   = alpha;
          fb  = f;
          db  = df;
        }
      if (PrintLevel >= 2)
        {
          if (Com->QuadOK)
            s = "OK";
          else
            s = "";
          printf("contract  %2s a: %13.6e b: %13.6e fa: %13.6e fb: "
                 "%13.6e da: %13.6e db: %13.6e\n",
                 s,
                 a,
                 b,
                 fa,
                 fb,
                 da,
                 db);
        }
    }

  /* see if the cost is small enough to change the PertRule */
  if (fabs(fb) <= Com->SmallCost)
    Com->PertRule = FALSE;

  /* increase eps if slope is negative after Parm->nshrink iterations */
  t = Com->f0;
  if (Com->PertRule)
    {
      if (t != ZERO)
        {
          Com->eps   = Parm->egrow * (f1 - t) / fabs(t);
          Com->fpert = t + fabs(t) * Com->eps;
        }
      else
        Com->fpert = 2. * f1;
    }
  else
    {
      Com->eps   = Parm->egrow * (f1 - t);
      Com->fpert = t + Com->eps;
    }
  if (PrintLevel >= 1)
    {
      printf("--increase eps: %e fpert: %e\n", Com->eps, Com->fpert);
    }
  Com->neps++;
  return (-1);
}

/* =========================================================================
   ==== cg_fg_evaluate =====================================================
   Evaluate the function and/or gradient.  Also, possibly check if either is nan
   and if so, then reduce the stepsize. Only used at the start of an iteration.
Return:
11 (function nan)
0 (successful evaluation)
=========================================================================*/

PRIVATE int
cg_evaluate(
  char *  what, /* fg = evaluate func and grad, g = grad only,f = func only*/
  char *  nan,  /* y means check function/derivative values for nan */
  cg_com *Com)
{
  INT           n;
  int           i;
  double        alpha, *d, *gtemp, *x, *xtemp;
  cg_parameter *Parm;
  Parm  = Com->Parm;
  n     = Com->n;
  x     = Com->x;
  d     = Com->d;
  xtemp = Com->xtemp;
  gtemp = Com->gtemp;
  alpha = Com->alpha;
  /* check to see if values are nan */
  if (!strcmp(nan, "y") || !strcmp(nan, "p"))
    {
      if (!strcmp(what, "f")) /* compute function */
        {
          cg_step(xtemp, x, d, alpha, n);
          /* provisional function value */
          Com->f = Com->cg_value(xtemp, n);
          Com->nf++;

          /* reduce stepsize if function value is nan */
          if ((Com->f != Com->f) || (Com->f >= INF) || (Com->f <= -INF))
            {
              for (i = 0; i < Parm->ntries; i++)
                {
                  if (!strcmp(nan, "p")) /* contract from good alpha */
                    {
                      alpha = Com->alphaold + .8 * (alpha - Com->alphaold);
                    }
                  else /* multiply by nan_decay */
                    {
                      alpha *= Parm->nan_decay;
                    }
                  cg_step(xtemp, x, d, alpha, n);
                  Com->f = Com->cg_value(xtemp, n);
                  Com->nf++;
                  if ((Com->f == Com->f) && (Com->f < INF) && (Com->f > -INF))
                    break;
                }
              if (i == Parm->ntries)
                {
                  printf("RuntimeError: line %d ->", __LINE__);
                  return (11);
                }
            }
          Com->alpha = alpha;
        }
      else if (!strcmp(what, "g")) /* compute gradient */
        {
          cg_step(xtemp, x, d, alpha, n);
          Com->cg_grad(gtemp, xtemp, n);
          Com->ng++;
          Com->df = cg_dot(gtemp, d, n);
          /* reduce stepsize if derivative is nan */
          if ((Com->df != Com->df) || (Com->df >= INF) || (Com->df <= -INF))
            {
              for (i = 0; i < Parm->ntries; i++)
                {
                  if (!strcmp(nan, "p")) /* contract from good alpha */
                    {
                      alpha = Com->alphaold + .8 * (alpha - Com->alphaold);
                    }
                  else /* multiply by nan_decay */
                    {
                      alpha *= Parm->nan_decay;
                    }
                  cg_step(xtemp, x, d, alpha, n);
                  Com->cg_grad(gtemp, xtemp, n);
                  Com->ng++;
                  Com->df = cg_dot(gtemp, d, n);
                  if ((Com->df == Com->df) && (Com->df < INF) &&
                      (Com->df > -INF))
                    break;
                }
              if (i == Parm->ntries)
                {
                  printf("RuntimeError: line %d ->", __LINE__);
                  return (11);
                }
              Com->rho = Parm->nan_rho;
            }
          else
            Com->rho = Parm->rho;
          Com->alpha = alpha;
        }
      else /* compute function and gradient */
        {
          cg_step(xtemp, x, d, alpha, n);
          if (Com->cg_valgrad != NULL)
            {
              Com->f = Com->cg_valgrad(gtemp, xtemp, n);
            }
          else
            {
              Com->cg_grad(gtemp, xtemp, n);
              Com->f = Com->cg_value(xtemp, n);
            }
          Com->df = cg_dot(gtemp, d, n);
          Com->nf++;
          Com->ng++;
          /* reduce stepsize if function value or derivative is nan */
          if ((Com->df != Com->df) || (Com->f != Com->f) || (Com->df >= INF) ||
              (Com->f >= INF) || (Com->df <= -INF) || (Com->f <= -INF))
            {
              for (i = 0; i < Parm->ntries; i++)
                {
                  if (!strcmp(nan, "p")) /* contract from good alpha */
                    {
                      alpha = Com->alphaold + .8 * (alpha - Com->alphaold);
                    }
                  else /* multiply by nan_decay */
                    {
                      alpha *= Parm->nan_decay;
                    }
                  cg_step(xtemp, x, d, alpha, n);
                  if (Com->cg_valgrad != NULL)
                    {
                      Com->f = Com->cg_valgrad(gtemp, xtemp, n);
                    }
                  else
                    {
                      Com->cg_grad(gtemp, xtemp, n);
                      Com->f = Com->cg_value(xtemp, n);
                    }
                  Com->df = cg_dot(gtemp, d, n);
                  Com->nf++;
                  Com->ng++;
                  if ((Com->df == Com->df) && (Com->f == Com->f) &&
                      (Com->df < INF) && (Com->f < INF) && (Com->df > -INF) &&
                      (Com->f > -INF))
                    break;
                }
              if (i == Parm->ntries)
                {
                  printf("RuntimeError: line %d ->", __LINE__);
                  return (11);
                }
              Com->rho = Parm->nan_rho;
            }
          else
            Com->rho = Parm->rho;
          Com->alpha = alpha;
        }
    }
  else /* evaluate without nan checking */
    {
      if (!strcmp(what, "fg")) /* compute function and gradient */
        {
          if (alpha == ZERO) /* evaluate at x */
            {
              /* the following copy is not needed except when the code
                 is run using the MATLAB mex interface */
              cg_copy(xtemp, x, n);
              if (Com->cg_valgrad != NULL)
                {
                  Com->f = Com->cg_valgrad(Com->g, xtemp, n);
                }
              else
                {
                  Com->cg_grad(Com->g, xtemp, n);
                  Com->f = Com->cg_value(xtemp, n);
                }
            }
          else
            {
              cg_step(xtemp, x, d, alpha, n);
              if (Com->cg_valgrad != NULL)
                {
                  Com->f = Com->cg_valgrad(gtemp, xtemp, n);
                }
              else
                {
                  Com->cg_grad(gtemp, xtemp, n);
                  Com->f = Com->cg_value(xtemp, n);
                }
              Com->df = cg_dot(gtemp, d, n);
            }
          Com->nf++;
          Com->ng++;
          if ((Com->df != Com->df) || (Com->f != Com->f) || (Com->df == INF) ||
              (Com->f == INF) || (Com->df == -INF) || (Com->f == -INF))
            {
              printf("RuntimeError: line %d ", __LINE__);
              printf("df %e ->", Com->df);
              return (11);
            }
        }
      else if (!strcmp(what, "f")) /* compute function */
        {
          cg_step(xtemp, x, d, alpha, n);
          Com->f = Com->cg_value(xtemp, n);
          Com->nf++;
          if ((Com->f != Com->f) || (Com->f == INF) || (Com->f == -INF))
            {
              printf("RuntimeError: line %d ->", __LINE__);
              return (11);
            }
        }
      else
        {
          cg_step(xtemp, x, d, alpha, n);
          Com->cg_grad(gtemp, xtemp, n);
          Com->df = cg_dot(gtemp, d, n);
          Com->ng++;
          if ((Com->df != Com->df) || (Com->df == INF) || (Com->df == -INF))
            {
              printf("RuntimeError: line %d ->", __LINE__);
              return (11);
            }
        }
    }
  return (0);
}

/* =========================================================================
   ==== cg_cubic ===========================================================
   =========================================================================
   Compute the minimizer of a Hermite cubic. If the computed minimizer
   outside [a, b], return -1 (it is assumed that a >= 0).
   ========================================================================= */
PRIVATE double
cg_cubic(double a,
         double fa, /* function value at a */
         double da, /* derivative at a */
         double b,
         double fb, /* function value at b */
         double db  /* derivative at b */
)
{
  double c, d1, d2, delta, t, v, w;
  delta = b - a;
  if (delta == ZERO)
    return (a);
  v = da + db - 3. * (fb - fa) / delta;
  t = v * v - da * db;
  if (t < ZERO) /* complex roots, use secant method */
    {
      if (fabs(da) < fabs(db))
        c = a - (a - b) * (da / (da - db));
      else if (da != db)
        c = b - (a - b) * (db / (da - db));
      else
        c = -1;
      return (c);
    }

  if (delta > ZERO)
    w = sqrt(t);
  else
    w = -sqrt(t);
  d1 = da + v - w;
  d2 = db + v + w;
  if ((d1 == ZERO) && (d2 == ZERO))
    return (-1.);
  if (fabs(d1) >= fabs(d2))
    c = a + delta * da / d1;
  else
    c = b - delta * db / d2;
  return (c);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   Start of routines that could use the BLAS
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* =========================================================================
   ==== cg_matvec ==========================================================
   =========================================================================
   Compute y = A*x or A'*x where A is a dense rectangular matrix
   ========================================================================= */
PRIVATE void
cg_matvec(double *y, /* product vector */
          double *A, /* dense matrix */
          double *x, /* input vector */
          int     n, /* number of columns of A */
          INT     m, /* number of rows of A */
          int     w  /* T => y = A*x, F => y = A'*x */
)
{
  /* if the blas have not been installed, then hand code the produce */
#ifdef NOBLAS
  INT j, l;
  l = 0;
  if (w)
    {
      cg_scale0(y, A, x[0], (int)m);
      for (j = 1; j < n; j++)
        {
          l += m;
          cg_daxpy0(y, A + l, x[j], (int)m);
        }
    }
  else
    {
      for (j = 0; j < n; j++)
        {
          y[j] = cg_dot0(A + l, x, (int)m);
          l += m;
        }
    }
#endif

    /* if the blas have been installed, then possibly call gdemv */
#ifndef NOBLAS
  INT      j, l;
  BLAS_INT M, N;
  if (w || (!w && (m * n < MATVEC_START)))
    {
      l = 0;
      if (w)
        {
          cg_scale(y, A, x[0], m);
          for (j = 1; j < n; j++)
            {
              l += m;
              cg_daxpy(y, A + l, x[j], m);
            }
        }
      else
        {
          for (j = 0; j < n; j++)
            {
              y[j] = cg_dot0(A + l, x, (int)m);
              l += m;
            }
        }
    }
  else
    {
      M = (BLAS_INT)m;
      N = (BLAS_INT)n;
      /* only use transpose mult with blas
         CG_DGEMV ("n", &M, &N, one, A, &M, x, blas_one, zero, y, blas_one) ;*/
      CG_DGEMV("t", &M, &N, one, A, &M, x, blas_one, zero, y, blas_one);
    }
#endif

  return;
}

/* =========================================================================
   ==== cg_trisolve ========================================================
   =========================================================================
   Solve Rx = y or R'x = y where R is a dense upper triangular matrix
   ========================================================================= */
PRIVATE void
cg_trisolve(double *x, /* right side on input, solution on output */
            double *R, /* dense matrix */
            int     m, /* leading dimension of R */
            int     n, /* dimension of triangular system */
            int     w  /* T => Rx = y, F => R'x = y */
)
{
  int i, l;
  if (w)
    {
      l = m * n;
      for (i = n; i > 0;)
        {
          i--;
          l -= (m - i);
          x[i] /= R[l];
          l -= i;
          cg_daxpy0(x, R + l, -x[i], i);
        }
    }
  else
    {
      l = 0;
      for (i = 0; i < n; i++)
        {
          x[i] = (x[i] - cg_dot0(x, R + l, i)) / R[l + i];
          l += m;
        }
    }

  /* equivalent to:
     BLAS_INT M, N ;
     M = (BLAS_INT) m ;
     N = (BLAS_INT) n ;
     if ( w ) CG_DTRSV ("u", "n", "n", &N, R, &M, x, blas_one) ;
     else     CG_DTRSV ("u", "t", "n", &N, R, &M, x, blas_one) ; */

  return;
}

/* =========================================================================
   ==== cg_inf =============================================================
   =========================================================================
   Compute infinity norm of vector
   ========================================================================= */
PRIVATE double
cg_inf(double *x, /* vector */
       INT     n  /* length of vector */
)
{
#ifdef NOBLAS
  INT    i, n5;
  double t;
  t  = ZERO;
  n5 = n % 5;

  for (i = 0; i < n5; i++)
    if (t < fabs(x[i]))
      t = fabs(x[i]);
  for (; i < n; i += 5)
    {
      if (t < fabs(x[i]))
        t = fabs(x[i]);
      if (t < fabs(x[i + 1]))
        t = fabs(x[i + 1]);
      if (t < fabs(x[i + 2]))
        t = fabs(x[i + 2]);
      if (t < fabs(x[i + 3]))
        t = fabs(x[i + 3]);
      if (t < fabs(x[i + 4]))
        t = fabs(x[i + 4]);
    }
  return (t);
#endif

#ifndef NOBLAS
  INT      i, n5;
  double   t;
  BLAS_INT N;
  if (n < IDAMAX_START)
    {
      t  = ZERO;
      n5 = n % 5;

      for (i = 0; i < n5; i++)
        if (t < fabs(x[i]))
          t = fabs(x[i]);
      for (; i < n; i += 5)
        {
          if (t < fabs(x[i]))
            t = fabs(x[i]);
          if (t < fabs(x[i + 1]))
            t = fabs(x[i + 1]);
          if (t < fabs(x[i + 2]))
            t = fabs(x[i + 2]);
          if (t < fabs(x[i + 3]))
            t = fabs(x[i + 3]);
          if (t < fabs(x[i + 4]))
            t = fabs(x[i + 4]);
        }
      return (t);
    }
  else
    {
      N = (BLAS_INT)n;
      i = (INT)CG_IDAMAX(&N, x, blas_one);
      return (fabs(x[i - 1])); /* adjust for fortran indexing */
    }
#endif
}

/* =========================================================================
   ==== cg_scale0 ===========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
PRIVATE void
cg_scale0(double *y, /* output vector */
          double *x, /* input vector */
          double  s, /* scalar */
          int     n  /* length of vector */
)
{
  int i, n5;
  n5 = n % 5;
  if (s == -ONE)
    {
      for (i = 0; i < n5; i++)
        y[i] = -x[i];
      for (; i < n;)
        {
          y[i] = -x[i];
          i++;
          y[i] = -x[i];
          i++;
          y[i] = -x[i];
          i++;
          y[i] = -x[i];
          i++;
          y[i] = -x[i];
          i++;
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        y[i] = s * x[i];
      for (; i < n;)
        {
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
        }
    }
  return;
}

/* =========================================================================
   ==== cg_scale ===========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
PRIVATE void
cg_scale(double *y, /* output vector */
         double *x, /* input vector */
         double  s, /* scalar */
         INT     n  /* length of vector */
)
{
  INT i, n5;
  n5 = n % 5;
  if (y == x)
    {
#ifdef NOBLAS
      for (i = 0; i < n5; i++)
        y[i] *= s;
      for (; i < n;)
        {
          y[i] *= s;
          i++;
          y[i] *= s;
          i++;
          y[i] *= s;
          i++;
          y[i] *= s;
          i++;
          y[i] *= s;
          i++;
        }
#endif
#ifndef NOBLAS
      if (n < DSCAL_START)
        {
          for (i = 0; i < n5; i++)
            y[i] *= s;
          for (; i < n;)
            {
              y[i] *= s;
              i++;
              y[i] *= s;
              i++;
              y[i] *= s;
              i++;
              y[i] *= s;
              i++;
              y[i] *= s;
              i++;
            }
        }
      else
        {
          BLAS_INT N;
          N = (BLAS_INT)n;
          CG_DSCAL(&N, &s, x, blas_one);
        }
#endif
    }
  else
    {
      for (i = 0; i < n5; i++)
        y[i] = s * x[i];
      for (; i < n;)
        {
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
          y[i] = s * x[i];
          i++;
        }
    }
  return;
}

/* =========================================================================
   ==== cg_daxpy0 ==========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
PRIVATE void
cg_daxpy0(double *x,     /* input and output vector */
          double *d,     /* direction */
          double  alpha, /* stepsize */
          int     n      /* length of the vectors */
)
{
  INT i, n5;
  n5 = n % 5;
  if (alpha == -ONE)
    {
      for (i = 0; i < n5; i++)
        x[i] -= d[i];
      for (; i < n; i += 5)
        {
          x[i] -= d[i];
          x[i + 1] -= d[i + 1];
          x[i + 2] -= d[i + 2];
          x[i + 3] -= d[i + 3];
          x[i + 4] -= d[i + 4];
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        x[i] += alpha * d[i];
      for (; i < n; i += 5)
        {
          x[i] += alpha * d[i];
          x[i + 1] += alpha * d[i + 1];
          x[i + 2] += alpha * d[i + 2];
          x[i + 3] += alpha * d[i + 3];
          x[i + 4] += alpha * d[i + 4];
        }
    }
  return;
}

/* =========================================================================
   ==== cg_daxpy ===========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
PRIVATE void
cg_daxpy(double *x,     /* input and output vector */
         double *d,     /* direction */
         double  alpha, /* stepsize */
         INT     n      /* length of the vectors */
)
{
#ifdef NOBLAS
  INT i, n5;
  n5 = n % 5;
  if (alpha == -ONE)
    {
      for (i = 0; i < n5; i++)
        x[i] -= d[i];
      for (; i < n; i += 5)
        {
          x[i] -= d[i];
          x[i + 1] -= d[i + 1];
          x[i + 2] -= d[i + 2];
          x[i + 3] -= d[i + 3];
          x[i + 4] -= d[i + 4];
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        x[i] += alpha * d[i];
      for (; i < n; i += 5)
        {
          x[i] += alpha * d[i];
          x[i + 1] += alpha * d[i + 1];
          x[i + 2] += alpha * d[i + 2];
          x[i + 3] += alpha * d[i + 3];
          x[i + 4] += alpha * d[i + 4];
        }
    }
#endif

#ifndef NOBLAS
  INT      i, n5;
  BLAS_INT N;
  if (n < DAXPY_START)
    {
      n5 = n % 5;
      if (alpha == -ONE)
        {
          for (i = 0; i < n5; i++)
            x[i] -= d[i];
          for (; i < n; i += 5)
            {
              x[i] -= d[i];
              x[i + 1] -= d[i + 1];
              x[i + 2] -= d[i + 2];
              x[i + 3] -= d[i + 3];
              x[i + 4] -= d[i + 4];
            }
        }
      else
        {
          for (i = 0; i < n5; i++)
            x[i] += alpha * d[i];
          for (; i < n; i += 5)
            {
              x[i] += alpha * d[i];
              x[i + 1] += alpha * d[i + 1];
              x[i + 2] += alpha * d[i + 2];
              x[i + 3] += alpha * d[i + 3];
              x[i + 4] += alpha * d[i + 4];
            }
        }
    }
  else
    {
      N = (BLAS_INT)n;
      CG_DAXPY(&N, &alpha, d, blas_one, x, blas_one);
    }
#endif

  return;
}

/* =========================================================================
   ==== cg_dot0 ============================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
PRIVATE double
cg_dot0(double *x, /* first vector */
        double *y, /* second vector */
        int     n  /* length of vectors */
)
{
  INT    i, n5;
  double t;
  t = ZERO;
  if (n <= 0)
    return (t);
  n5 = n % 5;
  for (i = 0; i < n5; i++)
    t += x[i] * y[i];
  for (; i < n; i += 5)
    {
      t += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] +
           x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
    }
  return (t);
}

/* =========================================================================
   ==== cg_dot =============================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
PRIVATE double
cg_dot(double *x, /* first vector */
       double *y, /* second vector */
       INT     n  /* length of vectors */
)
{
#ifdef NOBLAS
  INT    i, n5;
  double t;
  t = ZERO;
  if (n <= 0)
    return (t);
  n5 = n % 5;
  for (i = 0; i < n5; i++)
    t += x[i] * y[i];
  for (; i < n; i += 5)
    {
      t += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] +
           x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
    }
  return (t);
#endif

#ifndef NOBLAS
  INT      i, n5;
  double   t;
  BLAS_INT N;
  if (n < DDOT_START)
    {
      t = ZERO;
      if (n <= 0)
        return (t);
      n5 = n % 5;
      for (i = 0; i < n5; i++)
        t += x[i] * y[i];
      for (; i < n; i += 5)
        {
          t += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] +
               x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
        }
      return (t);
    }
  else
    {
      N = (BLAS_INT)n;
      return (CG_DDOT(&N, x, blas_one, y, blas_one));
    }
#endif
}

/* =========================================================================
   === cg_copy0 ============================================================
   =========================================================================
   Copy vector x into vector y
   ========================================================================= */
PRIVATE void
cg_copy0(double *y, /* output of copy */
         double *x, /* input of copy */
         int     n  /* length of vectors */
)
{
  int i, n5;
  n5 = n % 5;
  for (i = 0; i < n5; i++)
    y[i] = x[i];
  for (; i < n;)
    {
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
    }
  return;
}

/* =========================================================================
   === cg_copy =============================================================
   =========================================================================
   Copy vector x into vector y
   ========================================================================= */
PRIVATE void
cg_copy(double *y, /* output of copy */
        double *x, /* input of copy */
        INT     n  /* length of vectors */
)
{
#ifdef NOBLAS
  INT i, n5;
  n5 = n % 5;
  for (i = 0; i < n5; i++)
    y[i] = x[i];
  for (; i < n;)
    {
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
      y[i] = x[i];
      i++;
    }
#endif

#ifndef NOBLAS
  INT      i, n5;
  BLAS_INT N;
  if (n < DCOPY_START)
    {
      n5 = n % 5;
      for (i = 0; i < n5; i++)
        y[i] = x[i];
      for (; i < n;)
        {
          y[i] = x[i];
          i++;
          y[i] = x[i];
          i++;
          y[i] = x[i];
          i++;
          y[i] = x[i];
          i++;
          y[i] = x[i];
          i++;
        }
    }
  else
    {
      N = (BLAS_INT)n;
      CG_DCOPY(&N, x, blas_one, y, blas_one);
    }
#endif

  return;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   End of routines that could use the BLAS
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* =========================================================================
   ==== cg_step ============================================================
   =========================================================================
   Compute xtemp = x + alpha d
   ========================================================================= */
PRIVATE void
cg_step(double *xtemp, /*output vector */
        double *x,     /* initial vector */
        double *d,     /* search direction */
        double  alpha, /* stepsize */
        INT     n      /* length of the vectors */
)
{
  INT n5, i;
  n5 = n % 5;
  if (alpha == -ONE)
    {
      for (i = 0; i < n5; i++)
        xtemp[i] = x[i] - d[i];
      for (; i < n; i += 5)
        {
          xtemp[i]     = x[i] - d[i];
          xtemp[i + 1] = x[i + 1] - d[i + 1];
          xtemp[i + 2] = x[i + 2] - d[i + 2];
          xtemp[i + 3] = x[i + 3] - d[i + 3];
          xtemp[i + 4] = x[i + 4] - d[i + 4];
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        xtemp[i] = x[i] + alpha * d[i];
      for (; i < n; i += 5)
        {
          xtemp[i]     = x[i] + alpha * d[i];
          xtemp[i + 1] = x[i + 1] + alpha * d[i + 1];
          xtemp[i + 2] = x[i + 2] + alpha * d[i + 2];
          xtemp[i + 3] = x[i + 3] + alpha * d[i + 3];
          xtemp[i + 4] = x[i + 4] + alpha * d[i + 4];
        }
    }
  return;
}

/* =========================================================================
   ==== cg_init ============================================================
   =========================================================================
   initialize x to a given scalar value
   ========================================================================= */
PRIVATE void
cg_init(double *x, /* input and output vector */
        double  s, /* scalar */
        INT     n  /* length of vector */
)
{
  INT i, n5;
  n5 = n % 5;
  for (i = 0; i < n5; i++)
    x[i] = s;
  for (; i < n;)
    {
      x[i] = s;
      i++;
      x[i] = s;
      i++;
      x[i] = s;
      i++;
      x[i] = s;
      i++;
      x[i] = s;
      i++;
    }
  return;
}

/* =========================================================================
   ==== cg_update_2 ========================================================
   =========================================================================
   Set gold = gnew (if not equal), compute 2-norm^2 of gnew, and optionally
   set d = -gnew
   ========================================================================= */
PRIVATE double
cg_update_2(double *gold, /* old g */
            double *gnew, /* new g */
            double *d,    /* d */
            INT     n     /* length of vectors */
)
{
  INT    i, n5;
  double s, t;
  t  = ZERO;
  n5 = n % 5;

  if (d == NULL)
    {
      for (i = 0; i < n5; i++)
        {
          s = gnew[i];
          t += s * s;
          gold[i] = s;
        }
      for (; i < n;)
        {
          s = gnew[i];
          t += s * s;
          gold[i] = s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          i++;
        }
    }
  else if (gold != NULL)
    {
      for (i = 0; i < n5; i++)
        {
          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
        }
      for (; i < n;)
        {
          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
          i++;

          s = gnew[i];
          t += s * s;
          gold[i] = s;
          d[i]    = -s;
          i++;
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        {
          s = gnew[i];
          t += s * s;
          d[i] = -s;
        }
      for (; i < n;)
        {
          s = gnew[i];
          t += s * s;
          d[i] = -s;
          i++;

          s = gnew[i];
          t += s * s;
          d[i] = -s;
          i++;

          s = gnew[i];
          t += s * s;
          d[i] = -s;
          i++;

          s = gnew[i];
          t += s * s;
          d[i] = -s;
          i++;

          s = gnew[i];
          t += s * s;
          d[i] = -s;
          i++;
        }
    }
  return (t);
}

/* =========================================================================
   ==== cg_update_inf ======================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew, and optionally set d = -gnew
   ========================================================================= */
PRIVATE double
cg_update_inf(double *gold, /* old g */
              double *gnew, /* new g */
              double *d,    /* d */
              INT     n     /* length of vectors */
)
{
  INT    i, n5;
  double s, t;
  t  = ZERO;
  n5 = n % 5;

  if (d == NULL)
    {
      for (i = 0; i < n5; i++)
        {
          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
        }
      for (; i < n;)
        {
          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          if (t < fabs(s))
            t = fabs(s);
          i++;
        }
    }
  else
    {
      for (i = 0; i < n5; i++)
        {
          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
        }
      for (; i < n;)
        {
          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
          i++;

          s       = gnew[i];
          gold[i] = s;
          d[i]    = -s;
          if (t < fabs(s))
            t = fabs(s);
          i++;
        }
    }
  return (t);
}
/* =========================================================================
   ==== cg_update_ykyk =====================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew
   ykyk = 2-norm(gnew-gold)^2
   ykgk = (gnew-gold) dot gnew
   ========================================================================= */
PRIVATE double
cg_update_ykyk(double *gold, /* old g */
               double *gnew, /* new g */
               double *Ykyk,
               double *Ykgk,
               INT     n /* length of vectors */
)
{
  INT    i, n5;
  double t, gnorm, yk, ykyk, ykgk;
  gnorm = ZERO;
  ykyk  = ZERO;
  ykgk  = ZERO;
  n5    = n % 5;

  for (i = 0; i < n5; i++)
    {
      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
    }
  for (; i < n;)
    {
      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      yk      = t - gold[i];
      gold[i] = t;
      ykgk += yk * t;
      ykyk += yk * yk;
      i++;
    }
  *Ykyk = ykyk;
  *Ykgk = ykgk;
  return (gnorm);
}

/* =========================================================================
   ==== cg_update_inf2 =====================================================
   =========================================================================
   Set gold = gnew, compute inf-norm of gnew & 2-norm of gnew, set d = -gnew
   ========================================================================= */
PRIVATE double
cg_update_inf2(double *gold,   /* old g */
               double *gnew,   /* new g */
               double *d,      /* d */
               double *gnorm2, /* 2-norm of g */
               INT     n       /* length of vectors */
)
{
  INT    i, n5;
  double gnorm, s, t;
  gnorm = ZERO;
  s     = ZERO;
  n5    = n % 5;

  for (i = 0; i < n5; i++)
    {
      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
    }
  for (; i < n;)
    {
      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
      i++;

      t = gnew[i];
      if (gnorm < fabs(t))
        gnorm = fabs(t);
      s += t * t;
      gold[i] = t;
      d[i]    = -t;
      i++;
    }
  *gnorm2 = s;
  return (gnorm);
}

/* =========================================================================
   ==== cg_update_d ========================================================
   =========================================================================
   Set d = -g + beta*d, compute 2-norm of d, and optionally the 2-norm of g
   ========================================================================= */
PRIVATE double
cg_update_d(double *d,
            double *g,
            double  beta,
            double *gnorm2, /* 2-norm of g */
            INT     n       /* length of vectors */
)
{
  INT    i, n5;
  double dnorm2, s, t;
  s      = ZERO;
  dnorm2 = ZERO;
  n5     = n % 5;
  if (gnorm2 == NULL)
    {
      for (i = 0; i < n5; i++)
        {
          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
        }
      for (; i < n;)
        {
          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t    = g[i];
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;
        }
    }
  else
    {
      s = ZERO;
      for (i = 0; i < n5; i++)
        {
          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
        }
      for (; i < n;)
        {
          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;

          t = g[i];
          s += t * t;
          t    = -t + beta * d[i];
          d[i] = t;
          dnorm2 += t * t;
          i++;
        }
      *gnorm2 = s;
    }

  return (dnorm2);
}

/* =========================================================================
   ==== cg_Yk ==============================================================
   =========================================================================
   Compute y = gnew - gold, set gold = gnew, compute y'y
   ========================================================================= */
PRIVATE void
cg_Yk(double *y,    /*output vector */
      double *gold, /* initial vector */
      double *gnew, /* search direction */
      double *yty,  /* y'y */
      INT     n     /* length of the vectors */
)
{
  INT    n5, i;
  double s, t;
  n5 = n % 5;
  if ((y != NULL) && (yty == NULL))
    {
      for (i = 0; i < n5; i++)
        {
          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
        }
      for (; i < n;)
        {
          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
          i++;

          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
          i++;

          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
          i++;

          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
          i++;

          y[i]    = gnew[i] - gold[i];
          gold[i] = gnew[i];
          i++;
        }
    }
  else if ((y == NULL) && (yty != NULL))
    {
      s = ZERO;
      for (i = 0; i < n5; i++)
        {
          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
        }
      for (; i < n;)
        {
          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          s += t * t;
          i++;
        }
      *yty = s;
    }
  else
    {
      s = ZERO;
      for (i = 0; i < n5; i++)
        {
          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
        }
      for (; i < n;)
        {
          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
          i++;

          t       = gnew[i] - gold[i];
          gold[i] = gnew[i];
          y[i]    = t;
          s += t * t;
          i++;
        }
      *yty = s;
    }

  return;
}

/* =========================================================================
   === cg_default ==========================================================
   =========================================================================
   Set default conjugate gradient parameter values. If the parameter argument
   of cg_descent is NULL, this routine is called by cg_descent automatically.
   If the user wishes to set parameter values, then the cg_parameter structure
   should be allocated in the main program. The user could call cg_default
   to initialize the structure, and then individual elements in the structure
   could be changed, before passing the structure to cg_descent.
   =========================================================================*/
void
cg_default(cg_parameter *Parm)
{
  /* T => print final function value
     F => no printout of final function value */
  Parm->PrintFinal = FALSE;

  /* Level 0 = no printing, ... , Level 3 = maximum printing */
  Parm->PrintLevel = 0;

  /* T => print parameters values
     F => do not display parameter values */
  Parm->PrintParms = FALSE;

  /* T => use LBFGS
     F => only use L-BFGS when memory >= n */
  Parm->LBFGS = FALSE;

  /* number of vectors stored in memory (code breaks in the Yk update if
     memory = 1 or 2) */
  Parm->memory = 11;

  /* SubCheck and SubSkip control the frequency with which the subspace
     condition is checked. It it checked for SubCheck*mem iterations and
     if it is not activated, then it is skipped for Subskip*mem iterations
     and Subskip is doubled. Whenever the subspace condition is satisfied,
     SubSkip is returned to its original value. */
  Parm->SubCheck = 8;
  Parm->SubSkip  = 4;

  /* when relative distance from current gradient to subspace <= eta0,
     enter subspace if subspace dimension = mem (eta0 = 0 means gradient
     inside subspace) */
  Parm->eta0 = 0.001; /* corresponds to eta0*eta0 in the paper */

  /* when relative distance from current gradient to subspace >= eta1,
     leave subspace (eta1 = 1 means gradient orthogonal to subspace) */
  Parm->eta1 = 0.900; /* corresponds to eta1*eta1 in the paper */

  /* when relative distance from current gradient to subspace <= eta2,
     always enter subspace (invariant space) */
  Parm->eta2 = 1.e-10;

  /* T => use approximate Wolfe line search
     F => use ordinary Wolfe line search, switch to approximate Wolfe when
     |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost */
  Parm->AWolfe    = FALSE;
  Parm->AWolfeFac = 1.e-3;

  /* factor in [0, 1] used to compute average cost magnitude C_k as follows:
     Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
  Parm->Qdecay = .7;

  /* terminate after 2*n + nslow iterations without strict improvement in
     either function value or gradient */
  Parm->nslow = 1000;

  /* Stop Rules:
     T => ||grad||_infty <= max(grad_tol, initial |grad|_infty*StopFact)
     F => ||grad||_infty <= grad_tol*(1 + |f_k|) */
  Parm->StopRule = TRUE;
  Parm->StopFac  = 0.e-12;

  /* T => estimated error in function value is eps*Ck,
     F => estimated error in function value is eps */
  Parm->PertRule = TRUE;
  Parm->eps      = 1.e-6;

  /* factor by which eps grows when line search fails during contraction */
  Parm->egrow = 10.;

  /* T => attempt quadratic interpolation in line search when
     |f_k+1 - f_k|/|f_k| > QuadCutOff
     F => no quadratic interpolation step */
  Parm->QuadStep   = TRUE;
  Parm->QuadCutOff = 1.e-12;

  /* maximum factor by which a quad step can reduce the step size */
  Parm->QuadSafe = 1.e-10;

  /* T => when possible, use a cubic step in the line search */
  Parm->UseCubic = TRUE;

  /* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
  Parm->CubicCutOff = 1.e-12;

  /* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = FALSE*/
  Parm->SmallCost = 1.e-30;

  /* T => check that f_k+1 - f_k <= debugtol*C_k
     F => no checking of function values */
  Parm->debug    = FALSE;
  Parm->debugtol = 1.e-10;

  /* if step is nonzero, it is the initial step of the initial line search */
  Parm->step = ZERO;

  /* abort cg after maxit iterations */
  Parm->maxit = INT_INF;

  /* maximum number of times the bracketing interval grows during expansion */
  Parm->ntries = (int)50;

  /* maximum factor secant step increases stepsize in expansion phase */
  Parm->ExpandSafe = 200.;

  /* factor by which secant step is amplified during expansion phase
     where minimizer is bracketed */
  Parm->SecantAmp = 1.05;

  /* factor by which rho grows during expansion phase where minimizer is
     bracketed */
  Parm->RhoGrow = 2.0;

  /* maximum number of times that eps is updated */
  Parm->neps = (int)5;

  /* maximum number of times the bracketing interval shrinks */
  Parm->nshrink = (int)10;

  /* maximum number of secant iterations in line search is nline */
  Parm->nline = (int)50;

  /* conjugate gradient method restarts after (n*restart_fac) iterations */
  Parm->restart_fac = 6.0;

  /* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
  Parm->feps = ZERO;

  /* after encountering nan, growth factor when searching for
     a bracketing interval */
  Parm->nan_rho = 1.3;

  /* after encountering nan, decay factor for stepsize */
  Parm->nan_decay = 0.1;

  /* Wolfe line search parameter, range [0, .5]
     phi (a) - phi (0) <= delta phi'(0) */
  Parm->delta = .1;

  /* Wolfe line search parameter, range [delta, 1]
     phi' (a) >= sigma phi' (0) */
  Parm->sigma = .9;

  /* decay factor for bracket interval width in line search, range (0, 1) */
  Parm->gamma = .66;

  /* growth factor in search for initial bracket interval */
  Parm->rho = 5.;

  /* starting guess for line search =
     psi0 ||x_0||_infty over ||g_0||_infty if x_0 != 0
     psi0 |f(x_0)|/||g_0||_2               otherwise */
  Parm->psi0 = .01; /* factor used in starting guess for iteration 1 */

  /* for a QuadStep, function evaluated on interval
     [psi_lo, phi_hi]*psi2*previous step */
  Parm->psi_lo = 0.1;
  Parm->psi_hi = 10.;

  /* when the function is approximately quadratic, use gradient at
     psi1*psi2*previous step for estimating initial stepsize */
  Parm->psi1 = 1.0;

  /* when starting a new cg iteration, our initial guess for the line
     search stepsize is psi2*previous step */
  Parm->psi2 = 2.;

  /* choose theta adaptively if AdaptiveBeta = T */
  Parm->AdaptiveBeta = FALSE;

  /* lower bound for beta is BetaLower*d_k'g_k/ ||d_k||^2 */
  Parm->BetaLower = 0.4;

  /* value of the parameter theta in the cg_descent update formula:
     W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
     methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58. */
  Parm->theta = 1.0;

  /* parameter used in cost error estimate for quadratic restart criterion */
  Parm->qeps = 1.e-12;

  /* number of iterations the function is nearly quadratic before a restart */
  Parm->qrestart = 6;

  /* treat cost as quadratic if
     |1 - (cost change)/(quadratic cost change)| <= qrule */
  Parm->qrule = 1.e-8;
}

/* =========================================================================
   ==== cg_printParms ======================================================
   =========================================================================
   Print the contents of the cg_parameter structure
   ========================================================================= */
PRIVATE void
cg_printParms(cg_parameter *Parm)
{
  printf("PARAMETERS:\n");
  printf("\n");
  printf("Wolfe line search parameter ..................... delta: %e\n",
         Parm->delta);
  printf("Wolfe line search parameter ..................... sigma: %e\n",
         Parm->sigma);
  printf("decay factor for bracketing interval ............ gamma: %e\n",
         Parm->gamma);
  printf("growth factor for bracket interval ................ rho: %e\n",
         Parm->rho);
  printf("growth factor for bracket interval after nan .. nan_rho: %e\n",
         Parm->nan_rho);
  printf("decay factor for stepsize after nan ......... nan_decay: %e\n",
         Parm->nan_decay);
  printf("parameter in lower bound for beta ........... BetaLower: %e\n",
         Parm->BetaLower);
  printf("parameter describing cg_descent family .......... theta: %e\n",
         Parm->theta);
  printf("perturbation parameter for function value ......... eps: %e\n",
         Parm->eps);
  printf("factor by which eps grows if necessary .......... egrow: %e\n",
         Parm->egrow);
  printf("factor for computing average cost .............. Qdecay: %e\n",
         Parm->Qdecay);
  printf("relative change in cost to stop quadstep ... QuadCutOff: %e\n",
         Parm->QuadCutOff);
  printf("maximum factor quadstep reduces stepsize ..... QuadSafe: %e\n",
         Parm->QuadSafe);
  printf("skip quadstep if |f| <= SmallCost*start cost  SmallCost: %e\n",
         Parm->SmallCost);
  printf("relative change in cost to stop cubic step  CubicCutOff: %e\n",
         Parm->CubicCutOff);
  printf("terminate if no improvement over nslow iter ..... nslow: %i\n",
         Parm->nslow);
  printf("factor multiplying gradient in stop condition . StopFac: %e\n",
         Parm->StopFac);
  printf("cost change factor, approx Wolfe transition . AWolfeFac: %e\n",
         Parm->AWolfeFac);
  printf("restart cg every restart_fac*n iterations . restart_fac: %e\n",
         Parm->restart_fac);
  printf("cost error in quadratic restart is qeps*cost ..... qeps: %e\n",
         Parm->qeps);
  printf("number of quadratic iterations before restart  qrestart: %i\n",
         Parm->qrestart);
  printf("parameter used to decide if cost is quadratic ... qrule: %e\n",
         Parm->qrule);
  printf("stop when cost change <= feps*|f| ................ feps: %e\n",
         Parm->feps);
  printf("starting guess parameter in first iteration ...... psi0: %e\n",
         Parm->psi0);
  printf("starting step in first iteration if nonzero ...... step: %e\n",
         Parm->step);
  printf("lower bound factor in quad step ................ psi_lo: %e\n",
         Parm->psi_lo);
  printf("upper bound factor in quad step ................ psi_hi: %e\n",
         Parm->psi_hi);
  printf("initial guess factor for quadratic functions ..... psi1: %e\n",
         Parm->psi1);
  printf("initial guess factor for general iteration ....... psi2: %e\n",
         Parm->psi2);
  printf("max iterations .................................. maxit: %i\n",
         (int)Parm->maxit);
  printf("max number of contracts in the line search .... nshrink: %i\n",
         Parm->nshrink);
  printf("max expansions in line search .................. ntries: %i\n",
         Parm->ntries);
  printf("maximum growth of secant step in expansion . ExpandSafe: %e\n",
         Parm->ExpandSafe);
  printf("growth factor for secant step during expand . SecantAmp: %e\n",
         Parm->SecantAmp);
  printf("growth factor for rho during expansion phase .. RhoGrow: %e\n",
         Parm->RhoGrow);
  printf("distance threshhold for entering subspace ........ eta0: %e\n",
         Parm->eta0);
  printf("distance threshhold for leaving subspace ......... eta1: %e\n",
         Parm->eta1);
  printf("distance threshhold for invariant space .......... eta2: %e\n",
         Parm->eta2);
  printf("number of vectors stored in memory ............. memory: %i\n",
         Parm->memory);
  printf("check subspace condition mem*SubCheck its .... SubCheck: %i\n",
         Parm->SubCheck);
  printf("skip subspace checking for mem*SubSkip its .... SubSkip: %i\n",
         Parm->SubSkip);
  printf("max number of times that eps is updated .......... neps: %i\n",
         Parm->neps);
  printf("max number of iterations in line search ......... nline: %i\n",
         Parm->nline);
  printf("print level (0 = none, 3 = maximum) ........ PrintLevel: %i\n",
         Parm->PrintLevel);
  printf("Logical parameters:\n");
  if (Parm->PertRule)
    printf("    Error estimate for function value is eps*Ck\n");
  else
    printf("    Error estimate for function value is eps\n");
  if (Parm->QuadStep)
    printf("    Use quadratic interpolation step\n");
  else
    printf("    No quadratic interpolation step\n");
  if (Parm->UseCubic)
    printf("    Use cubic interpolation step when possible\n");
  else
    printf("    Avoid cubic interpolation steps\n");
  if (Parm->AdaptiveBeta)
    printf("    Adaptively adjust direction update parameter beta\n");
  else
    printf("    Use fixed parameter theta in direction update\n");
  if (Parm->PrintFinal)
    printf("    Print final cost and statistics\n");
  else
    printf("    Do not print final cost and statistics\n");
  if (Parm->PrintParms)
    printf("    Print the parameter structure\n");
  else
    printf("    Do not print parameter structure\n");
  if (Parm->AWolfe)
    printf("    Approximate Wolfe line search\n");
  else
    printf("    Wolfe line search");
  if (Parm->AWolfeFac > 0.)
    printf(" ... switching to approximate Wolfe\n");
  else
    printf("\n");
  if (Parm->StopRule)
    printf("    Stopping condition uses initial grad tolerance\n");
  else
    printf("    Stopping condition weighted by absolute cost\n");
  if (Parm->debug)
    printf("    Check for decay of cost, debugger is on\n");
  else
    printf("    Do not check for decay of cost, debugger is off\n");
}

/*
   Version 1.2 Change:
   1. The variable dpsi needs to be included in the argument list for
   subroutine cg_updateW (update of a Wolfe line search)

   Version 2.0 Changes:
   The user interface was redesigned. The parameters no longer need to
   be read from a file. For compatibility with earlier versions of the
   code, we include the routine cg_readParms to read parameters.
   In the simplest case, the user can use NULL for the
   parameter argument of cg_descent, and the code sets the default
   parameter values. If the user wishes to modify the parameters, call
   cg_default in the main program to initialize a cg_parameter
   structure. Individual elements of the structure could be modified.
   The header file cg_user.h contains the structures and prototypes
   that the user may need to reference or modify, while cg_descent.h
   contains header elements that only cg_descent will access.  Note
   that the arguments of cg_descent have changed.

   Version 3.0 Changes:
   Major overhaul

   Version 4.0 Changes:
   Modifications 1-3 were made based on results obtained by Yu-Hong Dai and
   Cai-Xia Kou in the paper "A nonlinear conjugate gradient algorithm with an
   optimal property and an improved Wolfe line search"
   1. Set theta = 1.0 by default in the cg_descent rule for beta_k (Dai and
   Kou showed both theoretical and practical advantages in this choice).
   2. Increase the default value of restart_fac to 6 (a value larger than 1 is
   more efficient when the problem dimension is small)
   3. Restart the CG iteration if the objective function is nearly quadratic
   for several iterations (qrestart).
   4. New lower bound for beta: BetaLower*d_k'g_k/ ||d_k||^2. This lower
   bound guarantees convergence and it seems to provide better
   performance than the original lower bound for beta in cg_descent;
   it also seems to give slightly better performance than the
   lower bound BetaLower*d_k'g_k+1/ ||d_k||^2 suggested by Dai and Kou.
   5. Evaluation of the objective function and gradient is now handled by
   the routine cg_evaluate.

   Version 4.1 Changes:
   1. Change cg_tol to be consistent with corresponding routine in asa_cg
   2. Compute dnorm2 when d is evaluated and make loops consistent with asa_cg

   Version 4.2 Changes:
   1. Modify the line search so that when there are too many contractions,
   the code will increase eps and switch to expansion of the search interval.
   This fixes some cases where the code terminates when eps is too small.
   When the estimated error in the cost function is too small, the algorithm
   could fail in cases where the slope is negative at both ends of the
   search interval and the objective function value on the right side of the
   interval is larger than the value at the left side (because the true
   objective function value on the right is not greater than the value on
   the left).
   2. Fix bug in cg_lineW

   Version 5.0 Changes:
   Revise the line search routines to exploit steps based on the
   minimizer of a Hermite interpolating cubic. Combine the approximate
   and the ordinary Wolfe line search into a single routine.
   Include safeguarded extrapolation during the expansion phase
   of the line search. Employ a quadratic interpolation step even
   when the requirement ftemp < f for a quadstep is not satisfied.

   Version 5.1 Changes:
   1. Shintaro Kaneko pointed out spelling error in line 738, change
   "stict" to "strict"
   2. Add MATLAB interface

   Version 5.2 Changes:
   1. Make QuadOK always TRUE in the quadratic interpolation step.
2. Change QuadSafe to 1.e-10 instead of 1.e-3 (less safe guarding).
3. Change psi0 to 2 in the routine for computing the stepsize
in the first iteration when x = 0.
4. In the quadratic step routine, the stepsize at the trial point
is the maximum of {psi_lo*psi2, prior df/(current df)}*previous step
5. In quadratic step routine, the function is evaluated on the safe-guarded
interval [psi_lo, phi_hi]*psi2*previous step.
6. Allow more rapid expansion in the line search routine.

Version 5.3 Changes:
1. Make changes so that cg_descent works with version R2012a of MATLAB.
This required allocating memory for the work array inside the MATLAB
mex routine and rearranging memory so that the xtemp pointer and the
work array pointer are the same.

Version 6.0 Changes:
Major revision of the code to implement the limited memory conjugate
gradient algorithm documented in reference [4] at the top of this file.
The code has doubled in length. The line search remains the same except
for small adjustments in the nan detection which allows it to
solve more problems that generate nan's. The direction routine, however,
      is completely new. Version 5.3 of the code is obtained by setting
      the memory parameter to be 0. The L-BFGS algorithm is obtained by setting
      the LBFGS parameter to TRUE. Otherwise, for memory > 0 and LBFGS = FALSE,
      the search directions are obtained by a limited memory version of the
      original cg_descent algorithm. The memory is used to detect when the
      gradients lose orthogonality locally. When orthogonality is lost,
      the algorithm solves a subspace problem until orthogonality is restored.
      It is now possible to utilize the BLAS, if they are available, by
      commenting out a line in the file cg_blas.h. See the README file for
      the details.

      Version 6.1 Changes:
      Fixed problems connected with memory handling in the MATLAB version
      of the code. These errors only arise when using some versions of MATLAB.
      Replaced "malloc" in the cg_descent mex routine with "mxMalloc".
      Thanks to Stephen Vavasis for reporting this error that occurred when
      using MATLAB version R2012a, 7.14.

      Version 6.2 Change:
      When using cg_descent in MATLAB, the input starting guess is no longer
      overwritten by the final solution. This makes the cg_descent mex function
      compliant with MATLAB's convention for the treatment of input arguments.
      Thanks to Dan Scholnik, Naval Research Laboratory, for pointing out this
      inconsistency with MATLAB convention in earlier versions of cg_descent.

      Version 6.3 Change:
      For problems of dimension <= Parm->memory (default 11), the final
      solution was not copied to the user's solution argument x. Instead x
      stored the the next-to-final iterate. This final copy is now inserted
      on line 969.  Thanks to Arnold Neumaier of the University of Vienna
      for pointing out this bug.

      Version 6.4 Change:
      In order to prevent a segmentation fault connected with MATLAB's memory
      handling, inserted a copy statement inside cg_evaluate. This copy is
      only needed when solving certain problems with MATLAB.  Many thanks
      to Arnold Neumaier for pointing out this problem

      Version 6.5 Change:
      When using the code in MATLAB, changed the x vector that is passed to
      the user's function and gradient routine to be column vectors instead
      of row vectors.

      Version 6.6 Change:
Expand the statistics structure to include the number of subspaces (NumSub)
  and the number of subspace iterations (IterSub).

  Version 6.7 Change:
  Add interface to CUTEst

  Version 6.8 Change:
  When the denominator of the variable "scale" vanishes, retain the
  previous value of scale. This correct an error pointed out by
  Zachary Blunden-Codd.

  Stefano's Change:
  set uninitialised values Com.df = 0;Com.df0 = 0; Com.f = 0; Com.f0 = 0;
  (this fixes a bug that tipically arises when calling the minimiser several
times in a row, strictly speaking uninitialising a variable leads to undefined
behaviour) The function now takes a pointer to a user_test for the user own
termination conditions and to user_data that allows to pass a pointer to a class
with methods
  */
