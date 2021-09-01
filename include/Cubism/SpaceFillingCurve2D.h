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
   int BX, BY;
   bool isRegular;
   int base_level;
   std::vector < std::vector <long long> > Zsave;
   std::vector< std::vector<int> > i_inverse;
   std::vector< std::vector<int> > j_inverse;

   //convert (x,y) to index
   long long AxestoTranspose(const int *X_in, int b) const
   {
      int x = X_in[0];
      int y = X_in[1];
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

   //convert index to (x,y)
   void TransposetoAxes(long long index, int *X, int b) const 
   {
       // position, #bits, dimension
       int n = 1 << b;
       long long rx, ry, s, t=index;
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
   void rot(long long n, int *x, int *y, long long rx, long long ry) const
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
   int levelMax;

   SpaceFillingCurve2D(){};

   SpaceFillingCurve2D(int a_BX, int a_BY, int lmax) : BX(a_BX), BY(a_BY), levelMax(lmax)
   {
      const int n_max  = max(BX, BY);
      base_level = (log(n_max) / log(2));
      if (base_level < (double)(log(n_max) / log(2))) base_level++;

      i_inverse.resize(lmax);
      j_inverse.resize(lmax);
      Zsave.resize(lmax);
      #ifdef CUBISM_USE_MAP
      for (int l = 0 ; l < 1 ; l++)
      #else
      for (int l = 0 ; l < lmax ; l++)
      #endif
      {
        const int aux = pow( pow(2,l) , 2);
        i_inverse[l].resize(BX*BY*aux,-1);
        j_inverse[l].resize(BX*BY*aux,-1);
        Zsave[l].resize(BX*BY*aux,-1);
      }

      isRegular = true;
      #pragma omp parallel for collapse(2)
      for (int j=0;j<BY;j++)
      for (int i=0;i<BX;i++)
      {
        const int c[2] = {i,j};
        long long index = AxestoTranspose( c, base_level);
        long long substract = 0;
        for (long long h=0; h<index; h++)
        {
          int X[2] = {0,0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY) substract++;   
        }
        index -= substract;
        if (substract > 0) isRegular = false;
        i_inverse[0][index] = i;
        j_inverse[0][index] = j;
        Zsave[0][j*BX + i] = index;
      }
    }

   // space-filling curve (i,j,k) --> 1D index (given level l)
   long long forward(const int l, const int i, const int j) //const
   {
      const int aux = 1 << l;

      if (l>=levelMax) return 0;
      #ifndef CUBISM_USE_MAP
        if (Zsave[l][ j*aux*BX + i ] != -1) return Zsave[l][  j*aux*BX + i ];
      #endif
      long long retval;
      if (!isRegular)
      {
        const int I   = i / aux;
        const int J   = j / aux;
        const int c2_a[2] = {i-I*aux,j-J*aux};
        retval = AxestoTranspose(c2_a, l);
        retval += IJ_to_index(I,J)*aux*aux;
      }
      else
      {
        const int c2_a[2] = {i,j};
        retval = AxestoTranspose(c2_a, l + base_level);
      }
      #ifndef CUBISM_USE_MAP
        i_inverse[l][retval] = i;
        j_inverse[l][retval] = j;
        Zsave[l][j*aux*BX + i] = retval;
      #endif
      return retval;
   }

   void inverse(long long Z, int l, int &i, int &j)
   {
      #ifndef CUBISM_USE_MAP
        if (i_inverse[l][Z] != -1)
        {
          assert(i_inverse[l][Z] != -1 && j_inverse[l][Z] != -1);
          i = i_inverse[l][Z];
          j = j_inverse[l][Z];
          return;
        }
      #endif
      if (isRegular)
      {
        int X[2] = {0, 0};
        TransposetoAxes(Z, X, l + base_level);
        i = X[0];
        j = X[1];
        #ifndef CUBISM_USE_MAP
          int aux   = 1 << l;
          i_inverse[l][Z] = i;
          j_inverse[l][Z] = j;
          Zsave[l][j*aux*BX + i] = Z;
        #endif
      }
      else
      {
        int aux   = 1 << l;
        long long Zloc  = Z % (aux*aux);
        int X[2] = {0, 0};
        TransposetoAxes(Zloc, X, l);
        long long index = Z / (aux*aux);
        int I,J;
        index_to_IJ(index,I,J);
        i = X[0] + I*aux;
        j = X[1] + J*aux;
        #ifndef CUBISM_USE_MAP
          i_inverse[l][Z] = i;
          j_inverse[l][Z] = j;
          Zsave[l][j*aux*BX + i ] = Z;
        #endif
      }
      return;
   }

   long long IJ_to_index(int I, int J)
   {
     //int index = (J + K * BY) * BX + I;
     long long index = Zsave[0][J*BX + I];
     return index;
   }
   void index_to_IJ(long long index, int & I, int & J)
   {
      I = i_inverse[0][index];
      J = j_inverse[0][index];
      return;
   }

   // return 1D index of CHILD of block (i,j,k) at level l (child is at level l+1)
   long long child(int l, int i, int j) { return forward(l + 1, 2 * i, 2 * j); }

   long long Encode(int level, long long Z, int index[2])
   {
      int lmax   = levelMax;
      long long retval = 0;

      int ix = index[0];
      int iy = index[1];
      for (int l = level; l >= 0; l--)
      {
         long long Zp = forward(l, ix, iy);
         retval += Zp;
         ix /= 2;
         iy /= 2;
      }

      ix = 2 * index[0];
      iy = 2 * index[1];
      for (int l = level + 1; l < lmax; l++)
      {
         long long Zc = forward(l, ix, iy);

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
