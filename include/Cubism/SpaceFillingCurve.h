#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdint.h>

using namespace std;

namespace cubism // AMR_CUBISM
{

class SpaceFillingCurve
{

 protected:
  int BX, BY, BZ;


   int AxestoTranspose(const int *X_in, int b) const // position, #bits, dimension
   {
      if (b == 0)
      {
         assert(X_in[0] == 0);
         assert(X_in[1] == 0);
         assert(X_in[2] == 0);
         return 0;
      }

      const int n       = 3;
      int X[3] = {X_in[0], X_in[1], X_in[2]};

      assert(b - 1 >= 0);

      int M = 1 << (b - 1), P, Q, t;
      int i;

      // Inverse undo
      for (Q = M; Q > 1; Q >>= 1)
      {
         P = Q - 1;
         for (i = 0; i < n; i++)
            if (X[i] & Q) X[0] ^= P; // invert
            else
            {
               t = (X[0] ^ X[i]) & P;
               X[0] ^= t;
               X[i] ^= t;
            }
      } // exchange

      // Gray encode
      for (i = 1; i < n; i++) X[i] ^= X[i - 1];
      t = 0;
      for (Q = M; Q > 1; Q >>= 1)
         if (X[n - 1] & Q) t ^= Q - 1;
      for (i = 0; i < n; i++) X[i] ^= t;

      int retval = 0;
      int a      = 0;
      for (int level = 0; level < b; level++)
      {
         retval += ((1 << (a)) * (X[2] >> level & 1)) + ((1 << (a + 1)) * (X[1] >> level & 1)) +
                   ((1 << (a + 2)) * (X[0] >> level & 1));
         a += 3;
      }

      return retval;
   }

   void TransposetoAxes(int index, int *X, int b) const // position, #bits, dimension
   {
      const int n = 3;

      X[0] = 0;
      X[1] = 0;
      X[2] = 0;

      int aa = 0;
      for (int i = 0; index > 0; i++)
      {
         int x2 = index % 2;
         index  = index / 2;
         int x1 = index % 2;
         index  = index / 2;
         int x0 = index % 2;
         index  = index / 2;

         X[0] += x0 * (1 << aa);
         X[1] += x1 * (1 << aa);
         X[2] += x2 * (1 << aa);

         aa += 1;
      }

      int N = 2 << (b - 1), P, Q, t;
      int i;

      // Gray decode by H ^ (H/2)
      t = X[n - 1] >> 1;
      for (i = n - 1; i >= 1; i--) X[i] ^= X[i - 1];
      X[0] ^= t;

      // Undo excess work
      for (Q = 2; Q != N; Q <<= 1)
      {
         P = Q - 1;
         for (i = n - 1; i >= 0; i--)
            if (X[i] & Q) X[0] ^= P; // invert
            else
            {
               t = (X[0] ^ X[i]) & P;
               X[0] ^= t;
               X[i] ^= t;
            }
      } // exchange
   }

 public:
   int *SUBSTRACT;
   int base_level;
   int levelMax;


   SpaceFillingCurve(){};

   ~SpaceFillingCurve(){ delete[] SUBSTRACT; }

   void __setup(int nx, int ny, int nz)
   {
      BX = nx;
      BY = ny;
      BZ = nz;
   }

   std::vector < std::vector <int> > Zsave;
   std::vector< std::vector<int> > i_inverse;
   std::vector< std::vector<int> > j_inverse;
   std::vector< std::vector<int> > k_inverse;


   SpaceFillingCurve(int a_BX, int a_BY, int a_BZ, int lmax) : BX(a_BX), BY(a_BY), BZ(a_BZ), levelMax(lmax)
   {
      std::cout << "Constructing Hilbert curve for " << BX << "x" << BY << "x" << BZ << 
                   " octree with " << lmax << " levels..." << std::endl;

      SUBSTRACT = new int[BX * BY * BZ];

      int n_max  = max(max(BX, BY), BZ);
      base_level = (log(n_max) / log(2));
      if (base_level < (double)(log(n_max) / log(2))) base_level++;
      #pragma omp parallel
      {
        #pragma omp for collapse(3)
        for (int k=0;k<BZ;k++)
        for (int j=0;j<BY;j++)
        for (int i=0;i<BX;i++)
        {
          const int c[3] = {i,j,k};
          
          int index = AxestoTranspose( c, base_level);
    
          int substract = 0;
          for (int h=0; h<index; h++)
          {
            int X[3] = {0,0,0};
            TransposetoAxes(h, X, base_level);
            if (X[0] >= BX ||  
                X[1] >= BY ||  
                X[2] >= BZ) substract++;   
          }   
          SUBSTRACT[(j + k*BY)*BX + i] = substract;
        }
      }

      i_inverse.resize(lmax);
      j_inverse.resize(lmax);
      k_inverse.resize(lmax);
      Zsave.resize(lmax);
      for (int l = 0 ; l < lmax ; l++)
      {
        int aux = pow( pow(2,l) , 3);
        i_inverse[l].resize(BX*BY*BZ*aux,-1);
        j_inverse[l].resize(BX*BY*BZ*aux,-1);
        k_inverse[l].resize(BX*BY*BZ*aux,-1);
        Zsave[l].resize(BX*BY*BZ*aux,-1);
        for (int k = 0 ; k < BZ* (1<<l); k++)
        for (int j = 0 ; j < BY* (1<<l); j++)
        for (int i = 0 ; i < BX* (1<<l); i++)
        {
         int n = forward(l,i,j,k);
        }

      }

      std::cout << "Hilbert curve ready." << std::endl;
   }

   // space-filling curve (i,j,k) --> 1D index (given level l)
   int forward(const int l, const int i, const int j, const int k)
   {
      const int aux = 1 << l;

      if (l >= levelMax) return -1;
      if (Zsave[l][  k*aux*aux*BX*BY + j*aux*BX + i ] != -1) return Zsave[l][  k*aux*aux*BX*BY + j*aux*BX + i ];

      const int I   = i / aux;
      const int J   = j / aux;
      const int K   = k / aux;
      
      if (I >= BX || J >= BY || K >= BZ) return 0;

      const int c2_a[3] = {i, j, k};
      int s                      = SUBSTRACT[((J + K * BY) * BX + I)] * aux * aux * aux;

      int retval = AxestoTranspose(c2_a, l + base_level) - s;

      i_inverse[l][retval] = i;
      j_inverse[l][retval] = j;
      k_inverse[l][retval] = k;
      Zsave[l][  k*aux*aux*BX*BY + j*aux*BX + i ] = retval;
      
      return retval;
   }

   void inverse(int Z, int l, int &i, int &j, int &k)
   {
      assert ( i_inverse[l][Z] != -1);
      assert ( j_inverse[l][Z] != -1);
      assert ( k_inverse[l][Z] != -1);
      i = i_inverse[l][Z];
      j = j_inverse[l][Z];
      k = k_inverse[l][Z];
      return;
      //int X[3] = {0, 0, 0};
      //TransposetoAxes(Z, X, l + base_level);
      //i = X[0];
      //j = X[1];
      //k = X[2];
   }

   void inverse1(int Z, int l, int &i, int &j, int &k)
   {
      for (int iz = 0; iz <= 1; iz++)
      for (int iy = 0; iy <= 1; iy++)
      for (int ix = 0; ix <= 1; ix++)
      {
        int Ztest = forward(l, ix+i, iy+j, iz+k);
        if (Ztest == Z)
        {
          i = ix+i;
          j = iy+j;
          k = iz+k;
          return;
        }
      }      
      std::cout << "Didn't find inverse in SFC!" << std::endl;
      abort();
   }


   // return 1D index of CHILD of block (i,j,k) at level l (child is at level l+1)
   int child(int l, int i, int j, int k) { return forward(l + 1, 2 * i, 2 * j, 2 * k); }

   int Encode(int level, int Z, int index[3])
   {
      int lmax   = levelMax;
      int retval = 0;

      int ix = index[0];
      int iy = index[1];
      int iz = index[2];
      for (int l = level; l >= 0; l--)
      {
         int Zp = forward(l, ix, iy, iz);
         retval += Zp;
         ix /= 2;
         iy /= 2;
         iz /= 2;
      }

      ix = 2 * index[0];
      iy = 2 * index[1];
      iz = 2 * index[2];
      for (int l = level + 1; l < lmax; l++)
      {
         int Zc = forward(l, ix, iy, iz);

         Zc -= Zc % 8;
         retval += Zc;

         int ix1, iy1, iz1;
         ix1 = ix;
         iy1 = iy;
         iz1 = iz;

         inverse1(Zc, l, ix1, iy1, iz1);
         ix = 2 * ix1;
         iy = 2 * iy1;
         iz = 2 * iz1;
      }

      retval += level;
  
      return retval;
   }
};

} // namespace cubism