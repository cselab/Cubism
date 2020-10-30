#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdint.h>

namespace cubism // AMR_CUBISM
{

class SpaceFillingCurve2D
{

 protected:
   unsigned int BX, BY;


   //convert (x,y) to d
   int AxestoTranspose(const unsigned int *X_in, int b) const
   {
      unsigned int x = X_in[0];
      unsigned int y = X_in[1];
      int n = 1 << b;
      int rx, ry, s, d=0;
      for (s=n/2; s>0; s/=2)
      {
          rx = (x & s) > 0;
          ry = (y & s) > 0;
          d += s * s * ((3 * rx) ^ ry);
          rot(n, &x, &y, rx, ry);
      }
      return d;
   }

   //convert d to (x,y)
   void TransposetoAxes(int index, unsigned int *X, int b) const 
   {
       // position, #bits, dimension
       int n = 1 << b;
       int rx, ry, s, t=index;
       X[0] = 0;
       X[1] = 0;
       for (s=1; s<n; s*=2) {
           rx = 1 & (t/2);
           ry = 1 & (t ^ rx);
           rot(s, &X[0], &X[1], rx, ry);
           X[0] += s * rx;
           X[1] += s * ry;
           t /= 4;
       }
   }

   //rotate/flip a quadrant appropriately
   void rot(int n, unsigned int *x, unsigned int *y, int rx, int ry) const 
   {
       if (ry == 0) {
           if (rx == 1) {
               *x = n-1 - *x;
               *y = n-1 - *y;
           }
   
           //Swap x and y
           int t  = *x;
           *x = *y;
           *y = t;
       }
   }



 public:
   int *SUBSTRACT;
   int base_level;
   int levelMax;


   std::vector< std::vector<int> > i_inverse;
   std::vector< std::vector<int> > j_inverse;
   std::vector < std::vector <int> > Zsave;

   SpaceFillingCurve2D(){};

   ~SpaceFillingCurve2D(){ delete[] SUBSTRACT; }

   void __setup(int nx, int ny)
   {
      BX = nx;
      BY = ny;
   }

   SpaceFillingCurve2D(unsigned int a_BX, unsigned int a_BY, int lmax) : BX(a_BX), BY(a_BY), levelMax(lmax)
   {
      std::cout << "Constructing Hilbert curve for " << BX << "x" << BY <<  " quadree with " << lmax << " levels..." << std::endl;
      i_inverse.resize(lmax);
      j_inverse.resize(lmax);
      for (int l = 0 ; l < lmax ; l++)
      {
        int aux = pow( pow(2,l) , 2);
        i_inverse[l].resize(BX*BY*aux,-1);
        j_inverse[l].resize(BX*BY*aux,-1);
        Zsave[l].resize(BX*BY*aux,-1);
      }

      SUBSTRACT = new int[BX * BY];

      int n_max  = max(BX, BY);
      base_level = (log(n_max) / log(2));
      if (base_level < (double)(log(n_max) / log(2))) base_level++;

      #pragma omp parallel for collapse(2)
      for (unsigned int j=0;j<BY;j++)
      for (unsigned int i=0;i<BX;i++)
      {
        const unsigned int c[2] = {i,j};
          
        int index = AxestoTranspose( c, base_level);
    
        int substract = 0;
        for (int h=0; h<index; h++)
        {
          unsigned int X[2] = {0,0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY) substract++;   
        }   
        SUBSTRACT[j*BX + i] = substract;
      }

      for (int l = 0 ; l < lmax ; l++)
      {
        int aux = pow(2,l);
        #pragma omp parallel for collapse(2)
        for (unsigned int j=0;j<BY*aux;j++)
        for (unsigned int i=0;i<BX*aux;i++)
        {
          int retval = forward(l,i,j);
        }      
      }

      std::cout << "Hilbert curve ready." << std::endl;
  }

   // space-filling curve (i,j,k) --> 1D index (given level l)
   int forward(const unsigned int l, const unsigned int i, const unsigned int j) //const
   {
      unsigned int aux = 1 << l;

      if (Zsave[l][ j*aux*BX + i ] != -1) return Zsave[l][  j*aux*BX + i ];
      unsigned int I   = i / aux;
      unsigned int J   = j / aux;
      
      if (I >= BX || J >= BY) return 0;
      if (l>=levelMax) return -1;

      const unsigned int c2_a[2] = {i, j};
      int s                      = SUBSTRACT[J * BX + I]  * aux * aux;

      int retval = AxestoTranspose(c2_a, l + base_level) - s;

      Zsave[l][  j*aux*BX + i ] = retval;
      return retval;
   }

   void inverse(int Z, int l, int &i, int &j)
   {
      i = i_inverse[l][Z];
      j = j_inverse[l][Z];
      assert(i_inverse[l][Z]>=0);
      assert(j_inverse[l][Z]>=0);
   }

   // return 1D index of CHILD of block (i,j,k) at level l (child is at level l+1)
   int child(int l, int i, int j) { return forward(l + 1, 2 * i, 2 * j); }

   int Encode(int level, int Z, int index[2])
   {
      int lmax   = levelMax;
      int retval = 0;

      int ix = index[0];
      int iy = index[1];
      for (int l = level; l >= 0; l--)
      {
         int Zp = forward(l, ix, iy);
         retval += Zp;
         ix /= 2;
         iy /= 2;
      }

      ix = 2 * index[0];
      iy = 2 * index[1];
      for (int l = level + 1; l < lmax; l++)
      {
         int Zc = forward(l, ix, iy);

         Zc -= Zc % 4;
         retval += Zc;

         int ix1, iy1;
         inverse(Zc, l, ix1, iy1);
         ix = 2 * ix1;
         iy = 2 * iy1;
      }

      retval += level;
  
      return retval;
   }
};

} // namespace cubism