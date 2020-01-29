#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <cassert>

#include "MeshMap.h"

using namespace std;




#define MortonCurve
//#define HilbertCurve


#define HACK



namespace cubism //AMR_CUBISM
{




class SpaceFillingCurve
{
protected:
    const unsigned int BX,BY,BZ;

    #ifdef HilbertCurve
    #endif

    #if   defined(MortonCurve)
        //Copy-pasted from 
        //www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations
        inline uint64_t mortonEncode_for(unsigned int x, unsigned int y, unsigned int z) const
        {
            uint64_t answer = 0;
            for (uint64_t i = 0; i < (sizeof(uint64_t)* CHAR_BIT)/3; ++i) {
            answer |= ((x & ((uint64_t)1 << i)) << 2*i) | 
                      ((y & ((uint64_t)1 << i)) << (2*i + 1)) | 
                      ((z & ((uint64_t)1 << i)) << (2*i + 2));
            }
            return answer;
        }
        //// DECODE 3D Morton code : For loop
        //inline void m3D_d_for(int m, int & x,  int & y,  int & z)
        //{
        //    x = 0; y = 0; z = 0;
        //    int checkbits = static_cast< int>(floor((sizeof(int) * 8.0f / 3.0f)));
        //    for (int i = 0; i <= checkbits; ++i) {
        //        int selector = 1;
        //        int shift_selector = 3 * i;
        //        int shiftback = 2 * i;
        //        x |= (m & (selector << shift_selector)) >> (shiftback);
        //        y |= (m & (selector << (shift_selector + 1))) >> (shiftback + 1);
        //        z |= (m & (selector << (shift_selector + 2))) >> (shiftback + 2);
        //    }
        //}
    #elif defined(HilbertCurve)
        #define adjust_rotation(rotation,nDims,bits)                            \
        do {                                                                    \
              /* rotation = (rotation + 1 + ffs(bits)) % nDims; */              \
              bits &= -bits & nd1Ones;                                          \
              while (bits)                                                      \
                bits >>= 1, ++rotation;                                         \
              if ( ++rotation >= nDims )                                        \
                rotation -= nDims;                                              \
        } while (0)
        
        #define ones(T,k) ((((T)2) << (k-1)) - 1)
        
        #define rdbit(w,k) (((w) >> (k)) & 1)
             
        #define rotateRight(arg, nRots, nDims)                                  \
        ((((arg) >> (nRots)) | ((arg) << ((nDims)-(nRots)))) & ones(int,nDims))
        
        #define rotateLeft(arg, nRots, nDims)                                   \
        ((((arg) << (nRots)) | ((arg) >> ((nDims)-(nRots)))) & ones(int,nDims))
        
        #define DLOGB_BIT_TRANSPOSE
        static int
        bitTranspose(unsigned nDims, unsigned nBits, int inCoords)
        #if defined(DLOGB_BIT_TRANSPOSE)
        {
          unsigned const nDims1 = nDims-1;
          unsigned inB = nBits;
          unsigned utB;
          int inFieldEnds = 1;
          int inMask = ones(int,inB);
          int coords = 0;
        
          while ((utB = inB / 2))
            {
              unsigned const shiftAmt = nDims1 * utB;
              int const utFieldEnds =
            inFieldEnds | (inFieldEnds << (shiftAmt+utB));
              int const utMask =
            (utFieldEnds << utB) - utFieldEnds;
              int utCoords = 0;
              unsigned d;
              if (inB & 1)
            {
              int const inFieldStarts = inFieldEnds << (inB-1);
              unsigned oddShift = 2*shiftAmt;
              for (d = 0; d < nDims; ++d)
                {
                  int in = inCoords & inMask;
                  inCoords >>= inB;
                  coords |= (in & inFieldStarts) << oddShift++;
                  in &= ~inFieldStarts;
                  in = (in | (in << shiftAmt)) & utMask;
                  utCoords |= in << (d*utB);
                }
            }
              else
            {
              for (d = 0; d < nDims; ++d)
                {
                  int in = inCoords & inMask;
                  inCoords >>= inB;
                  in = (in | (in << shiftAmt)) & utMask;
                  utCoords |= in << (d*utB);
                }
            }
              inCoords = utCoords;
              inB = utB;
              inFieldEnds = utFieldEnds;
              inMask = utMask;
            }
          coords |= inCoords;
          return coords;
        }
        #else
        {
          int coords = 0;
          unsigned d;
          for (d = 0; d < nDims; ++d)
            {
              unsigned b;
              int in = inCoords & ones(int,nBits);
              int out = 0;
              inCoords >>= nBits;
              for (b = nBits; b--;)
            {
              out <<= nDims;
              out |= rdbit(in, b);
            }
              coords |= out << d;
            }
          return coords;
        }
        #endif
        
        /*****************************************************************
         * hilbert_i2c
         * 
         * Convert an index into a Hilbert curve to a set of coordinates.
         * Inputs:
         *  nDims:      Number of coordinate axes.
         *  nBits:      Number of bits per axis.
         *  index:      The index, contains nDims*nBits bits
         *              (so nDims*nBits must be <= 8*sizeof(int)).
         * Outputs:
         *  coord:      The list of nDims coordinates, each with nBits bits.
         * Assumptions:
         *      nDims*nBits <= (sizeof index) * (bits_per_byte)
         */
        void
        hilbert_i2c(unsigned nDims, int index, int coord[])
        {
          int nBits = sizeof (int);
          if (nDims > 1)
            {
              int coords;
              int const nbOnes = ones(int,nBits);
              unsigned d;
        
              if (nBits > 1)
            {
              unsigned const nDimsBits = nDims*nBits;
              int const ndOnes = ones(int,nDims);
              int const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
              unsigned b = nDimsBits;
              unsigned rotation = 0;
              int flipBit = 0;
              int const nthbits = ones(int,nDimsBits) / ndOnes;
              index ^= (index ^ nthbits) >> 1;
              coords = 0;
              do
                {
                  int bits = (index >> (b-=nDims)) & ndOnes;
                  coords <<= nDims;
                  coords |= rotateLeft(bits, rotation, nDims) ^ flipBit;
                  flipBit = (int)1 << rotation;
                  adjust_rotation(rotation,nDims,bits);
                } while (b);
              for (b = nDims; b < nDimsBits; b *= 2)
                coords ^= coords >> b;
              coords = bitTranspose(nBits, nDims, coords);
            }
              else
            coords = index ^ (index >> 1);
        
              for (d = 0; d < nDims; ++d)
            {
              coord[d] = coords & nbOnes;
              coords >>= nBits;
            }
            }
          else
            coord[0] = index;
        }
        
        /*****************************************************************
         * hilbert_c2i
         * 
         * Convert coordinates of a point on a Hilbert curve to its index.
         * Inputs:
         *  nDims:      Number of coordinates.
         *  nBits:      Number of bits/coordinate.
         *  coord:      Array of n nBits-bit coordinates.
         * Outputs:
         *  index:      Output index value.  nDims*nBits bits.
         * Assumptions:
         *      nDims*nBits <= (sizeof int) * (bits_per_byte)
         */
        int
        hilbert_c2i(unsigned nDims,int const coord[]) const 
        {
          int nBits = sizeof(int);
        
          if (nDims > 1)
            {
              unsigned const nDimsBits = nDims*nBits;
              int index;
              unsigned d;
              int coords = 0;
              for (d = nDims; d--; )
            {
              coords <<= nBits;
              coords |= coord[d];
            }
        
              if (nBits > 1)
            {
              int const ndOnes = ones(int,nDims);
              int const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
              unsigned b = nDimsBits;
              unsigned rotation = 0;
              int flipBit = 0;
              int const nthbits = ones(int,nDimsBits) / ndOnes;
              coords = bitTranspose(nDims, nBits, coords);
              coords ^= coords >> nDims;
              index = 0;
              do
                {
                  int bits = (coords >> (b-=nDims)) & ndOnes;
                  bits = rotateRight(flipBit ^ bits, rotation, nDims);
                  index <<= nDims;
                  index |= bits;
                  flipBit = (int)1 << rotation;
                  adjust_rotation(rotation,nDims,bits);
                } while (b);
              index ^= nthbits >> 1;
            }
              else
            index = coords;
              for (d = 1; d < nDimsBits; d *= 2)
            index ^= index >> d;
              return index;
            }
          else
            return coord[0];
        }         
    #endif

public:
    SpaceFillingCurve(unsigned int a_BX, unsigned int a_BY, unsigned int a_BZ):BX(a_BX),BY(a_BY),BZ(a_BZ){}

    //space-filling curve (i,j,k) --> 1D index (given level l)
    int forward(const int l, const int i, const int j, const int k)  const
    {
        #if defined(MortonCurve)
            int aux = pow(2,l);  
            int I1 = i % aux;
            int J1 = j % aux;
            int K1 = k % aux;
            int I = i / aux;
            int J = j / aux;
            int K = k / aux;
            int z;
            int z_origin;
            z = mortonEncode_for(I1,J1,K1);
            z_origin = ((J + K*BY)*BX + I)*aux*aux*aux;
            return z+z_origin;
        #elif defined(HilbertCurve)
            int aux = pow(2,l);   
            int I1 = i % aux;
            int J1 = j % aux;
            int K1 = k % aux;
            int I = i / aux;
            int J = j / aux;
            int K = k / aux;
            int z;
            int z_origin;
            int const c [3] = {I1,J1,K1};
            z = hilbert_c2i(3,c);
            z_origin = ((J + K*BY)*BX + I)*aux*aux*aux;          
            return z+z_origin; 
        #else
            int aux = pow(2,l);   
            int nx = aux*BX;
            int ny = aux*BY;
            int nz = aux*BZ;
            return k*nx*ny + j*nx + i ; 
        #endif
    }

    //return 1D index of CHILD of block (i,j,k) at level l (child is at level l+1)
    int child(int l, int i, int j, int k)
    {
        return forward(l+1,2*i,2*j,2*k);
    }  

    //    void inverse(int l, int z, int & i, int & j, int & k) 
    //    {
    //        int p = pow (2,l);  
    //        int ZcoorLoc = z % (p*p*p);
    //        m3D_d_for(ZcoorLoc,i,j,k); 
    //    
    //        int box = z / (p*p*p);
    //        
    //        int kbox =  box / (BY*BX);
    //        int jbox = (box - kbox*BY*BX)/BX; 
    //        int ibox = (box - kbox*BY*BX-jbox*BX)%BX;
    //      
    //        i += ibox*  p; 
    //        j += jbox*  p;
    //        k += kbox*  p;
    //        return;
    //    }
};



enum TreePosition
{
  Exists=0,
  CheckCoarser=-1,
  CheckFiner=1
};


enum State
{
  Leave  =0,
  Refine =1,
  Compress=-1
};









struct BlockInfo
{
    #ifdef HACK
        long long blockID;
    #endif
    int index[3];         //(i,j,k) coordinates of block at given refinement level
    void * ptrBlock;      //Pointer to data stored in user-defined Block
    int myrank;           //MPI rank to which the associated block currently belongs
    TreePosition TreePos; //Indicates if block (level,Zorder) actually Exists in the Octree or if one should look for its coarser (finer) parents (children)
    State state;          //Refine/Compress/Leave this block
    int Z,level;          //Z-order curve index of this block and refinement level
    int Znei[3][3][3];    //Z-order curve index of 26 neighboring boxes (Znei[1][1][1] = Z)
    double h;             //grid spacing
    void * auxiliary;     //Pointer to blockcase
    double origin[3];     //(x,y,z) of block's origin
    template <typename T>


    inline void pos(T p[3], int ix, int iy, int iz) const
    {
        p[0] = origin_(0) + h_()*(ix+0.5);
        p[1] = origin_(1) + h_()*(iy+0.5);
        p[2] = origin_(2) + h_()*(iz+0.5);
    }
   
    template <typename T>
    inline std::array<T, 3> pos(int ix, int iy, int iz) const
    {
        std::array<T, 3> result;
        pos(result.data(), ix, iy, iz);
        return result;
    }


    #ifdef HACK
        bool special;
        double h_gridpoint;
        double uniform_grid_spacing[3];
        double block_extent[3];
        double* ptr_grid_spacing[3];
        bool bUniform[3];
          
        template <typename T> // 3D
        inline void spacing(T dx[3], int ix, int iy, int iz) const
        {
            dx[0] = h_();
            dx[1] = h_();
            dx[2] = h_();
        }
    
        BlockInfo(long long ID, const int idx[3], const double _pos[3], const double _spacing, double h_gridpoint_, void * ptr=NULL, const bool _special=false):
        blockID(ID), ptrBlock(ptr), special(_special)
        {
            std::cout << "BlockInfo hacked constructor called!\n";
            abort();
        }
         
        template <typename TBlock>
        BlockInfo(long long ID, const int idx[3], MeshMap<TBlock>* const mapX, MeshMap<TBlock>* const mapY, MeshMap<TBlock>* const mapZ, void * ptr=NULL, const bool _special=false):
        blockID(ID), ptrBlock(ptr), special(_special)
        {      
            std::cout << "BlockInfo with MeshMap called in AMR setting. Are you sure?\n";abort();
            abort();
        }
    #endif 

    BlockInfo(){};

    void setup(const int a_level,const double a_h, const double a_origin[3],
               const int a_Bmin[3],int a_index[3], int a_myrank, 
               TreePosition a_TreePos) 
    {
        myrank    = a_myrank;
        TreePos   = a_TreePos;
        index [0] = a_index[0];
        index [1] = a_index[1];
        index [2] = a_index[2];
        state  = Leave;
        if (ptrBlock != NULL) h_gridpoint = h_();


        level     = a_level;
        h         = a_h;
        origin[0] = a_origin[0];
        origin[1] = a_origin[1];
        origin[2] = a_origin[2];

        block_extent[0] = h* _BLOCKSIZE_;
        block_extent[1] = h* _BLOCKSIZE_;
        block_extent[2] = h* _BLOCKSIZE_;
        
        const int TwoPower = pow(2,level);
        const int Bmax[3] = {a_Bmin[0]*TwoPower,a_Bmin[1]*TwoPower,a_Bmin[2]*TwoPower};

        
        SpaceFillingCurve Zcurve(a_Bmin[0],a_Bmin[1],a_Bmin[2]);
        Z = Zcurve.forward(level,index[0],index[1],index[2]);

        for (int i=-1; i<2; i++)
        for (int j=-1; j<2; j++)
        for (int k=-1; k<2; k++)
        {
            Znei[i+1][j+1][k+1] = Zcurve.forward(level,(index[0]+i+Bmax[0])%Bmax[0],
                                                       (index[1]+j+Bmax[1])%Bmax[1],
                                                       (index[2]+k+Bmax[2])%Bmax[2]);
        }
    }

    int level_() const 
    {
        assert (ptrBlock != NULL);
        return level; //static_cast<BaseBlock*>(ptrBlock)->level;
    }

    int Z_() const
    {
        assert (ptrBlock != NULL);
        return Z; //static_cast<BaseBlock*>(ptrBlock)->Z;
    }

    int index_(int i) const
    {
        assert (i>=0 && i<3);
        return index[i];
    }

    int Znei_(int i, int j, int k) const
    {
        assert(abs(i)<=1);
        assert(abs(j)<=1);
        assert(abs(k)<=1);
        //assert (ptrBlock != NULL);

        return Znei[1+i][1+j][1+k];//static_cast<BaseBlock*>(ptrBlock)->Znei[1+i][1+j][1+k];
    }

    double origin_(int i) const
    {
        assert (i>=0 && i<3);
        assert (ptrBlock != NULL);
        return origin[i];//static_cast<BaseBlock*>(ptrBlock)->origin[i];
    }

    double h_() const
    {
        assert (ptrBlock != NULL);
        return h;//static_cast<BaseBlock*>(ptrBlock)->h;
    }
};

}//namespace AMR_CUBISM