#pragma once

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <cassert>

using namespace std;


//#define MortonCurve
#define HilbertCurve 

namespace cubism //AMR_CUBISM
{



class SpaceFillingCurve
{

protected: 

  unsigned int BX,BY,BZ;


  //Copy-pasted from 
  //www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations

  // method to seperate bits from a given integer 3 positions apart
  inline uint64_t splitBy3(unsigned int a)const
  {
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f;// shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3;// shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
  }
        
  inline uint64_t mortonEncode_for(unsigned int x, unsigned int y, unsigned int z) const
  {
    uint64_t answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
  }

  //// DECODE 3D Morton code : For loop
  inline void m3D_d_for(int m, int & x, int & y, int & z)
  {
    x = 0; y = 0; z = 0;
    int checkbits = static_cast< int>(floor((sizeof(int) * 8.0f / 3.0f)));
    for (int i = 0; i <= checkbits; ++i)
    {
      int selector = 1;
      int shift_selector = 3 * i;
      int shiftback = 2 * i;
      x |= (m & (selector << shift_selector)) >> (shiftback);
      y |= (m & (selector << (shift_selector + 1))) >> (shiftback + 1);
      z |= (m & (selector << (shift_selector + 2))) >> (shiftback + 2);
    }
  }



  int AxestoTranspose( const unsigned int* X_in, int b) const // position, #bits, dimension
  { 
    const int n = 3;
    unsigned int X [3] = {X_in[0],X_in[1],X_in[2]};
    int M = 1 << (b-1), P, Q, t;
    int i;
    
    // Inverse undo
    for( Q = M; Q > 1; Q >>= 1 )
    {
      P = Q - 1;
      for( i = 0; i < n; i++ )
      if( X[i] & Q ) X[0] ^= P; // invert
      else
      { 
        t = (X[0]^X[i]) & P; 
        X[0] ^= t; X[i] ^= t; 
      } 
    } // exchange
    
    // Gray encode
    for( i = 1; i < n; i++ ) 
      X[i] ^= X[i-1];
    t = 0;
    for( Q = M; Q > 1; Q >>= 1 )
      if( X[n-1] & Q ) t ^= Q-1;
        for( i = 0; i < n; i++ ) X[i] ^= t;
     
       
    int retval = 0;
    int a = 0;
    for (int level = 0; level < b; level ++)
    {
      retval +=   ( (1<<(a  ))   *( X[2] >> level & 1 ) ) 
                + ( (1<<(a+1))   *( X[1] >> level & 1 ) ) 
                + ( (1<<(a+2))   *( X[0] >> level & 1 ) );
      a += 3;
    }
    return retval;
  }

  void TransposetoAxes(int index, unsigned int* X, int b) const // position, #bits, dimension
  { 
    const int n = 3;

    X[0] = 0;
    X[1] = 0;
    X[2] = 0;

    int aa = 0;
    for(int i=0; index>0; i++)    
    {    
      int x2 =index%2;    
      index= index/2;  
      int x1 =index%2;    
      index= index/2;  
      int x0 =index%2;    
      index= index/2;  
      
      X[0] += x0*( 1<<aa );
      X[1] += x1*( 1<<aa );
      X[2] += x2*( 1<<aa );
  
      aa += 1;
    }

    int N = 2 << (b-1), P, Q, t;
    int i;

    // Gray decode by H ^ (H/2)
    t = X[n-1] >> 1;
    for( i = n-1; i >= 1; i-- )
      X[i] ^= X[i-1];
    X[0] ^= t;

    // Undo excess work
    for( Q = 2; Q != N; Q <<= 1 )
    {
      P = Q - 1;
      for( i = n-1; i >= 0 ; i-- )
        if( X[i] & Q ) 
          X[0] ^= P; // invert
        else
        {
          t = (X[0]^X[i]) & P;
          X[0] ^= t;
          X[i] ^= t;
        }
    } // exchange
  }

public:

  int * Z_ORIGIN;

	SpaceFillingCurve(){};

  ~SpaceFillingCurve(){delete [] Z_ORIGIN;}

	void __setup(int nx,int ny,int nz)
  {
    BX = nx;
    BY = ny;
    BZ = nz;
  }

  SpaceFillingCurve(unsigned int a_BX, unsigned int a_BY, unsigned int a_BZ):BX(a_BX),BY(a_BY),BZ(a_BZ)
  {
    Z_ORIGIN = new int [BX*BY*BZ];
    
    int n_max = max(max(BX,BY),BZ);
    int lvl   =  ( log(n_max) / log(2) );
    if (lvl < (double) (log(n_max) / log(2)) ) lvl ++;
    
    for (unsigned int k=0;k<BZ;k++)
    for (unsigned int j=0;j<BY;j++)
    for (unsigned int i=0;i<BX;i++)
    {
      const unsigned int c[3] = {i,j,k};
      
      int index = AxestoTranspose( c, lvl);

      int substract = 0;
      for (int h=0; h<index; h++)
      {
        unsigned int X[3] = {0,0,0};
        TransposetoAxes(h, X, lvl);
        if (X[0] >= BX ||  
            X[1] >= BY ||  
            X[2] >= BZ) substract++;   
      }   
      index -= substract;
      Z_ORIGIN[(j + k*BY)*BX + i] =index;    
    }  
  }

  //space-filling curve (i,j,k) --> 1D index (given level l)
  int forward(const int l, const int i, const int j, const int k)  const
  {
    unsigned int aux =  1 << l; 
    unsigned int I1 = i % aux;
    unsigned int J1 = j % aux;
    unsigned int K1 = k % aux;
    unsigned int I = i / aux;
    unsigned int J = j / aux;
    unsigned int K = k / aux;

    #if defined(MortonCurve)
      int z = mortonEncode_for(I1,J1,K1);
      int z_origin = ((J + K*BY)*BX + I)*aux*aux*aux;
      return z+z_origin;
    #elif defined(HilbertCurve)
      const unsigned int c1[3] = {I1,J1,K1};
      return Z_ORIGIN[((J + K*BY)*BX + I)]*aux*aux*aux + AxestoTranspose(c1, l);
    #endif
  }

  //return 1D index of CHILD of block (i,j,k) at level l (child is at level l+1)
  int child(int l, int i, int j, int k)
  {
    return forward(l+1,2*i,2*j,2*k);
  }  
   
  int Encode(int level, int Z)
  {
    int retval;
    if (level == 0) 
    {
      retval = Z;
    }
    else 
    {
      int V = BX*BY*BZ * pow(pow(2,level-1),3);
      retval = V + Z; 
    }
    return retval;
  }
};



}//namespace AMR_CUBISM
