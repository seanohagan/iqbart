#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>

#include "info.h"
#include "tree.h"
#include "funs.h"
#include "bd.h"
#include "rrn.h"
#include "gig.h"

using std::cout;
using std::endl;

#include "cppmonbart.h"

MonBartResults cmonbart(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
    double tau,
    double nu,
    double lambda,
    double alpha,
    double mybeta,
    double phi,
    size_t nd,
    size_t burn,
    size_t m,
    size_t nm,
    size_t nkeeptrain,
    size_t nkeeptest,
    size_t nkeeptestme,
    size_t nkeeptreedraws,
    size_t printevery,
    std::vector<bool>& monotone_flags,
    std::vector<double>& tau_quantile_vec,
    bool data_aug,
    unsigned int seed
)
{
   //cout << "*****Into main of monotonic bart" << endl;
   //cout << "data_aug is" << data_aug << endl;
   //----------------------------------------------------------- 
   //random number generation
   rrn gen(seed);

   //--------------------------------------------------
   //process args
   size_t skiptr,skipte,skipteme,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;
   if(nkeeptestme) {skipteme=nd/nkeeptestme;}
   else skipteme=nd+1;
   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;

   //--------------------------------------------------
   //cout << "**********************" << endl;
   //cout << "n: " << n << endl;
   //cout << "p: " << p << endl;
   //cout << "first and last y: " << y[0] << ", " << y[n-1] << endl;
   //cout << "first row: " << x[0] << ", " << x[p-1] << endl;
   //cout << "second row: " << x[p] << ", " << x[2*p-1] << endl;
   //cout << "last row: " << x[(n-1)*p] << ", " << x[n*p-1] << endl;
   if(np) {
      //cout << "np: " << np << endl;
      //cout << "first row xp: " << xp[0] << ", " << xp[p-1] << endl;
      //cout << "second row xp: " << xp[p] << ", " << xp[2*p-1] << endl;
      //cout << "last row xp : " << xp[(np-1)*p] << ", " << xp[np*p-1] << endl;
   } else {
      //cout << "no test observations" << endl;
   }
   //cout << "tau: " << tau << endl;
   //cout << "nu: " << nu << endl;
   //cout << "lambda: " << lambda << endl;
   //cout << "tree prior base: " << alpha << endl;
   //cout << "tree prior power: " << mybeta << endl;
   //cout << "burn (nskip): " << burn << endl;
   //cout << "nd (ndpost): " << nd << endl;
   //cout << "m (ntree): " << m << endl;
   //cout << "nm (mu grid size): " << nm << endl;
   //cout << "*****nkeeptrain,nkeeptest,nkeeptestme, nkeeptreedraws: " <<
               //nkeeptrain << ", " << nkeeptest << ", " << nkeeptestme << ", " << nkeeptreedraws << endl;
   //cout << "*****printevery: " << printevery << endl;
   //cout << "*****skiptr,skipte,skipteme,skiptreedraws: " <<
               //skiptr << "," << skipte << "," << skipteme << "," << skiptreedraws << endl;
   //cout << "**********************" << endl;

   //--------------------------------------------------
   // main code
   //--------------------------------------------------
   //process train data
   double bigval = std::numeric_limits<double>::infinity();
   double miny = bigval; //use range of y to calibrate prior
   double maxy = -bigval;
   sinfo allys;
   for(size_t i=0;i<n;i++) {
      if(y[i]<miny) miny=y[i];
      if(y[i]>maxy) maxy=y[i];
      allys.sy += y[i]; // sum of y
      allys.sy2 += y[i]*y[i]; // sum of y^2
   }
   allys.n = n;
   double ybar = allys.sy/n; //sample mean
   double shat = sqrt((allys.sy2-n*ybar*ybar)/(n-1)); //sample standard deviation
   //cout << "ybar,shat: " << ybar << ", " << shat <<  endl;

   //--------------------------------------------------
   //process test data
   dinfo dip; //data information for prediction
   dip.n=np; dip.p=p; dip.x = xp; dip.y=0;
   //cout << "dip.n: " << dip.n << endl;

   //--------------------------------------------------
   // xinfo
   xinfo xi; //data information for prediction
   size_t nc=100; //100 equally spaced cutpoints from min to max.
   makexinfo(p,n,x,xi,nc);
   //cout << "x1 cuts: " << xi[0][0] << " ... " << xi[0][nc-1] << endl;
   if(p>1) {
      //cout << "xp cuts: " << xi[p-1][0] << " ... " << xi[p-1][nc-1] << endl;
   }

   //--------------------------------------------------
   //trees
   std::vector<tree> t(m);
   for(size_t i=0;i<m;i++) t[i].setm(ybar/m);

   //--------------------------------------------------
   //prior and mcmc
   pinfo pi;
   pi.pbd=1.0; //prob of birth/death move
   pi.pb=.5; //prob of birth given  birth/death

   pi.alpha=alpha; //prior prob a bot node splits is alpha/(1+d)^beta
   pi.mybeta=mybeta;
   pi.tau=tau;
   // pi.sigma=shat;
   pi.phi = phi;


   // assumes tau is already valid, between 0 and 1
   std::vector<double> theta1(tau_quantile_vec.size());
   std::transform(tau_quantile_vec.begin(), tau_quantile_vec.end(), theta1.begin(),
                  [](double t) -> double {return (1.0 - 2.0 * t) / (t * (1.0 - t));}
   );
   std::vector<double> theta2_sq(tau_quantile_vec.size());
   std::transform(tau_quantile_vec.begin(), tau_quantile_vec.end(), theta2_sq.begin(),
                  [](double t) -> double {return 2.0 / (t * (1.0 - t));}
   );

   pi.theta1 = &theta1;
   pi.theta2_sq = &theta2_sq;

   pi.tau_vec = &tau_quantile_vec;

   //initialize latent nu for quantile
   std::vector<double> latent_nu(n, 1.0);



   // end new quantile

   //***** discrete prior for constained model
   std::vector<double> mg(nm,0.0);  //grid for mu. 
   double pridel=3*pi.tau;
   for(size_t i=0;i!=mg.size();i++) mg[i] = -pridel + 2*pridel*(i+1)/(nm+1);
   std::vector<double> pm(nm,0.0);  //prior for mu. 

   double sum=0.0;
   for(size_t i=0;i!=mg.size();i++) {
      pm[i] = pn(mg[i],0.0,pi.tau*pi.tau);
      sum += pm[i];
   }
   for(size_t i=0;i!=mg.size();i++)  pm[i] /= sum;
   pi.mg = &mg;
   pi.pm = &pm;

   //--------------------------------------------------
   //dinfo
   double* allfit = new double[n]; //sum of fit of all trees
   for(size_t i=0;i<n;i++) allfit[i]=ybar;
   double* r = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp
   double* ftemp = new double[n]; //fit of current tree
   dinfo di;
   di.n=n; di.p=p; di.x = x; di.y=r; //the y will be the residual

   //--------------------------------------------------
   //storage for ouput
   double* ppredmean=0; //posterior mean for prediction
   double* fpredtemp=0; //temporary fit vector to compute prediction
   if(dip.n) {
      ppredmean = new double[dip.n];
      fpredtemp = new double[dip.n];
      for(size_t i=0;i<dip.n;i++) ppredmean[i]=0.0;
   }
   double rss;  //residual sum of squares
   double restemp; //a residual

   //--------------------------------------------------
   //return data structures
   MonBartResults results;
   // results.sigma_draws.resize(nd+burn);
   results.yhat_train_draws.resize(nkeeptrain, std::vector<double>(n));
   results.yhat_test_draws.resize(nkeeptest, std::vector<double>(np));
   results.yhat_train_mean.resize(n, 0.0);
   if (np > 0) {
       results.yhat_test_mean.resize(np, 0.0);
   }


   std::stringstream treess;
   treess.precision(10);
   treess << nkeeptreedraws << " " << m << " " << p << endl;

   //--------------------------------------------------
   //mcmc
   //cout << "\nMCMC:\n";
   time_t tp;
   int time1 = time(&tp);
   // gen.set_df(n+nu);
   size_t trcnt=0;
   size_t tecnt=0;
   size_t temecnt=0;
   size_t treedrawscnt=0;
   bool keeptest,keeptestme,keeptreedraw;

   double gig_chi;
   double gig_psi;

   for(size_t i=0;i<(nd+burn);i++) {

      //if(i%printevery==0) cout << "i: " << i << ", out of " << nd+burn << endl;
      // draw latent nu

      if(data_aug) {
         for(size_t k = 0; k < n; k++) {
            gig_chi = (y[k] - allfit[k])*(y[k] - allfit[k]) / ((*pi.theta2_sq)[k] * pi.phi);
            gig_psi = (2.0 + (((*pi.theta1)[k]*(*pi.theta1)[k])/((*pi.theta2_sq)[k])))/pi.phi;
            latent_nu[k] = do_rgig(0.5, gig_chi, gig_psi, gen);
         }
      }

      //draw trees
      for(size_t j=0;j<m;j++) {
         fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<n;k++) {
            allfit[k] = allfit[k]-ftemp[k];
            r[k] = y[k]-allfit[k];
            if(data_aug) {
              r[k] -= (*pi.theta1)[k] * latent_nu[k]; //modified
            }
         }
         bdc(t[j],xi,di,pi,gen, latent_nu, monotone_flags, data_aug);
         drmuc(t[j],xi,di,pi,gen, latent_nu, monotone_flags, data_aug);
         fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
      }
      //draw sigma
      // rss=0.0;
      // for(size_t k=0;k<n;k++) {restemp=y[k]-allfit[k]; rss += restemp*restemp;}
      // pi.sigma = sqrt((nu*lambda + rss)/gen.chi_square());
      // results.sigma_draws[i]=pi.sigma;
      if(i>=burn) {
         for(size_t k=0;k<n;k++) results.yhat_train_mean[k]+=allfit[k];

         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            for(size_t k=0;k<n;k++) results.yhat_train_draws[trcnt][k]=allfit[k];
            trcnt+=1;
         }

         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         keeptestme = nkeeptestme && (((i-burn+1) % skipteme) ==0) && np;
         if(keeptest || keeptestme) {
            for(size_t j=0;j<dip.n;j++) ppredmean[j]=0.0;
            for(size_t j=0;j<m;j++) {
               fit(t[j],xi,dip,fpredtemp);
               for(size_t k=0;k<dip.n;k++) ppredmean[k] += fpredtemp[k];
            }
         }
         if(keeptest) {
            for(size_t k=0;k<np;k++) results.yhat_test_draws[tecnt][k]=ppredmean[k];
            tecnt+=1;
         }
         if(keeptestme) {
            for(size_t k=0;k<np;k++) results.yhat_test_mean[k]+=ppredmean[k];
            temecnt+=1;
         }
         keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
         if(keeptreedraw) {
            for(size_t jj=0;jj<m;jj++) treess << t[jj];
            treedrawscnt +=1;
         }
      }
   }
   int time2 = time(&tp);
   //cout << "time for loop: " << time2-time1 << endl;
   //cout << "check counts" << endl;
   //cout << "trcnt,tecnt,temecnt,treedrawscnt: " << trcnt << "," << tecnt << "," << temecnt << ", " << treedrawscnt << endl;

   for(size_t k=0;k<n;k++) results.yhat_train_mean[k]/=nd;
   if (temecnt > 0) {
       for(size_t k=0;k<np;k++) results.yhat_test_mean[k]/=temecnt;
   }


   results.tree_draws.trees = treess.str();
   results.tree_draws.cutpoints = xi;

   delete[] allfit;
   delete[] r;
   delete[] ftemp;
   if(dip.n) {
      delete[] ppredmean;
      delete[] fpredtemp;
   }

   return results;
}

