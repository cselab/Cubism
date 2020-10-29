#pragma once

#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <vector>

#include "SpaceFillingCurve.h"
#include "SpaceFillingCurve2D.h"

using namespace std;

#define DIMENSION 2
#include "MyClock.h"

namespace cubism // AMR_CUBISM
{

enum TreePosition
{
   Exists       = 0,
   CheckCoarser = -1,
   CheckFiner   = 1
};

enum State
{
   Leave    = 0,
   Refine   = 1,
   Compress = -1
};

struct BlockInfo
{
   static int levelMax(int l=0)
   {
      static int lmax = l;
      return lmax;
   }

#if DIMENSION == 3
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Warray-bounds" //ignore weird gcc warning
   static int blocks_per_dim(int i, int nx = 0, int ny = 0, int nz = 0)
   {
      static int a[3] = {nx, ny, nz};
      return a[i];
   }
 #pragma GCC diagnostic pop

   static SpaceFillingCurve * SFC()
   {
      static SpaceFillingCurve Zcurve(blocks_per_dim(0), blocks_per_dim(1), blocks_per_dim(2),levelMax());
      return &Zcurve;
   }

   static int forward(int level, int ix, int iy, int iz)
   {
      return (*SFC()).forward(level, ix, iy, iz);
   }

   static int child(int level, int ix, int iy, int iz)
   {
      return (*SFC()).child(level, ix, iy, iz);     
   }

   static int Encode(int level, int Z, int index[3])
   {
      return (*SFC()).Encode(level, Z, index);
   }

   static void inverse(int Z, int l, int &i, int &j, int &k)
   {
      (*SFC()).inverse(Z,l,i,j,k);   
   }
#endif

#if DIMENSION == 2

 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Warray-bounds" //ignore weird gcc warning
   static int blocks_per_dim(int i, int nx = 0, int ny = 0)
   {
      static int a[2] = {nx, ny};
      return a[i];
   }
 #pragma GCC diagnostic pop


   static SpaceFillingCurve2D * SFC()
   {
      static SpaceFillingCurve2D Zcurve(blocks_per_dim(0), blocks_per_dim(1),levelMax());
      return &Zcurve;
   }

   static int forward(int level, int ix, int iy)
   {
      return (*SFC()).forward(level, ix, iy);
   }

   static int child(int level, int ix, int iy)
   {
      return (*SFC()).child(level, ix, iy);     
   }

   static int Encode(int level, int Z, int index[2])
   {
      return (*SFC()).Encode(level, Z, index);
   }
   static void inverse(int Z, int l, int &i, int &j)
   {
      (*SFC()).inverse(Z,l,i,j);   
   }
#endif


   long long blockID,blockID_2;
   int index[3];         //(i,j,k) coordinates of block at given refinement level
   void *ptrBlock;       // Pointer to data stored in user-defined Block
   int myrank;           // MPI rank to which the associated block currently belongs
   TreePosition TreePos; // Indicates if block (level,Zorder) actually Exists in the Octree or if
                         // one should look for its coarser (finer) parents (children)
   State state;       // Refine/Compress/Leave this block
   int Z, level;      // Z-order curve index of this block and refinement level   
   int Znei[3][3][3]; // Z-order curve index of 26 neighboring boxes (Znei[1][1][1] = Z)

   double h,h_gridpoint;          // grid spacing
   void *auxiliary;   // Pointer to blockcase
   double origin[3];  //(x,y,z) of block's origin

   bool changed;
   bool changed2;

   int halo_block_id;
   int Zparent;

#if DIMENSION == 3
   int Zchild[2][2][2];
#endif
#if DIMENSION == 2
   int Zchild[2][2];
#endif

#if DIMENSION == 3
   template <typename T>
   inline void pos(T p[3], int ix, int iy, int iz) const
   {
      p[0] = origin[0] + h * (ix + 0.5);
      p[1] = origin[1] + h * (iy + 0.5);
      p[2] = origin[2] + h * (iz + 0.5);
   }
   template <typename T>
   inline std::array<T, 3> pos(int ix, int iy, int iz) const
   {
      std::array<T, 3> result;
      pos(result.data(), ix, iy, iz);
      return result;
   }
#endif

#if DIMENSION == 2
   template <typename T>
   inline void pos(T p[2], int ix, int iy) const
   {
      p[0] = origin[0] + h * (ix + 0.5);
      p[1] = origin[1] + h * (iy + 0.5);
   }
   template <typename T>
   inline std::array<T, 2> pos(int ix, int iy) const
   {
      std::array<T, 2> result;
      pos(result.data(), ix, iy);
      return result;
   }
#endif

   bool ready;
   BlockInfo(){ready = false;};

   bool operator<(const BlockInfo &other) const
   {
      return (blockID_2 < other.blockID_2);
   }

   void setup(const int a_level, const double a_h, const double a_origin[3], const int a_Z, int a_myrank, TreePosition a_TreePos)
   {
      ready = true;
      //myrank   = a_myrank;
      TreePos  = a_TreePos;

      level = a_level;
      Z     = a_Z;

      inverse(Z,level,index[0],index[1],index[2]);
      
      state    = Leave;
      h_gridpoint = a_h;

      level     = a_level;
      h         = a_h;
      origin[0] = a_origin[0];
      origin[1] = a_origin[1];
      origin[2] = a_origin[2];

      changed     = true;

      const int TwoPower = 1 << level;
#if DIMENSION == 3
      const int Bmax[3]  = {blocks_per_dim(0) * TwoPower, 
                            blocks_per_dim(1) * TwoPower,
                            blocks_per_dim(2) * TwoPower};

      for (int i = -1; i < 2; i++)
         for (int j = -1; j < 2; j++)
            for (int k = -1; k < 2; k++)
            {
               Znei[i + 1][j + 1][k + 1] =
                   forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                           (index[1] + j + Bmax[1]) % Bmax[1], (index[2] + k + Bmax[2]) % Bmax[2]);
            }
      if (level == 0)
      {
         Zparent = 0;
      }
      else
      {
         Zparent = forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                           (index[1] / 2 + Bmax[1]) % Bmax[1], (index[2] / 2 + Bmax[2]) % Bmax[2]);
      }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
               Zchild[i][j][k] =
                   forward(level + 1, 2 * index[0] + i, 2 * index[1] + j, 2 * index[2] + k);
            }


#endif
#if DIMENSION == 2
      const int Bmax[3]  = {blocks_per_dim(0) * TwoPower, 
                            blocks_per_dim(1) * TwoPower,
                            blocks_per_dim(2) * TwoPower};
      for (int i = -1; i < 2; i++)
         for (int j = -1; j < 2; j++)
            for (int k = -1; k < 2; k++)
            {
               Znei[i + 1][j + 1][k + 1] =
                   forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                                  (index[1] + j + Bmax[1]) % Bmax[1]);
            }
      if (level == 0)
      {
         Zparent = 0;
      }
      else
      {
         Zparent = forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                                      (index[1] / 2 + Bmax[1]) % Bmax[1]);
      }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            Zchild[i][j] =
                forward(level + 1, 2 * index[0] + i, 2 * index[1] + j);
         }
#endif

      blockID = index[0] + (1<<level)*blocks_per_dim(0)*index[1];
      blockID_2 = Encode(level, Z, index);
   }

   int Znei_(int i, int j, int k) const
   {
      assert(abs(i) <= 1);
      assert(abs(j) <= 1);
      assert(abs(k) <= 1);
      return Znei[1 + i][1 + j][1 + k];
   }
};
} // namespace cubism
