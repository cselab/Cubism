#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <vector>
#include <cassert>

#include "MeshMap.h"
#include "SpaceFillingCurve.h"

using namespace std;



//#define MortonCurve
#define HilbertCurve 

#define HACK

#include <mpi.h>

#include <string>

struct MyClock
{       
  int N;
  double t[100];
  double s[100];
  string name [100];

  void padTo(std::string &str, const size_t num, const char paddingChar = ' ')
  {
      if(num > str.size())
          str.insert(0, num - str.size(), paddingChar);
  }

  MyClock()
  {
    reset();
  }
  void reset()
  {
    N = 0;
    for (int i = 0; i < 100; i ++)
      t[i] = 0;
  }
  void start(int i, string _name)
  {
    name[i] = _name;
    s[i]    = MPI_Wtime();
    N = max(N,i);
  }
  void finish(int i)
  {
    t[i] += MPI_Wtime() - s[i];
  }
  void display()
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    double * mean    = new double [N];
    double * maximum = new double [N];
    MPI_Reduce(t,mean   ,N,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(t,maximum,N,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (rank != 0 )
    {
      delete [] mean;
      delete [] maximum;
      return;
    } 
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    for (int i=0; i<N; i++)
    {
      padTo(name[i],70);
      mean[i] /= size;
      printf("%s    :  %8.4f (max)     %8.4f (mean) \n", name[i].c_str(), maximum[i], mean[i]);
    }
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    
    delete [] mean;
    delete [] maximum;
  }
};
extern MyClock Clock;




namespace cubism //AMR_CUBISM
{


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
    static int blocks_per_dim (int i, int nx = 0,int ny = 0,int nz = 0)
    {
      static int a [3] = {nx,ny,nz};
      return a[i];
    }

    static int forward (int level, int ix, int iy, int iz)
    {
      static SpaceFillingCurve Zcurve (blocks_per_dim(0),blocks_per_dim(1),blocks_per_dim(2));
      return Zcurve.forward(level,ix,iy,iz);
    }

    static int Encode (int level, int Z, int index[3] )
    {
      static SpaceFillingCurve Zcurve (blocks_per_dim(0),blocks_per_dim(1),blocks_per_dim(2));
      return Zcurve.Encode(level,Z,index);
    }

    long long blockID;
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

    bool changed {true};

    int halo_block_id;
    int Zparent;

    int Zchild[2][2][2];

    template <typename T>
    inline void pos(T p[3], int ix, int iy, int iz) const
    {
        p[0] = origin[0] + h*(ix+0.5);
        p[1] = origin[1] + h*(iy+0.5);
        p[2] = origin[2] + h*(iz+0.5);
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
            dx[0] = h;
            dx[1] = h;
            dx[2] = h;
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

    bool operator<(const BlockInfo & other) const 
    { 
      return (blockID < other.blockID);

//      if (level == other.level)
//      {
//        assert ((blockID < other.blockID) == (Z < other.Z) );
//
//        return (Z < other.Z);
//      }
//      else if (level < other.level)
//      {
//        int aux = pow(2, other.level- level);
//        int i[3] = {other.index[0] / aux, other.index[1] / aux, other.index[2] / aux};
//        int zzz = forward(level,i[0],i[1],i[2]);
//
//
//        assert ((blockID < other.blockID) == (Z < zzz) );
//        return (Z < zzz);
//      }
//      else 
//      {
//        int aux = pow(2, level- other.level);
//        int i[3] = {index[0] / aux, index[1] / aux, index[2] / aux};
//        int zzz = forward(other.level,i[0],i[1],i[2]);
//
//
//        std::cout << "zzz='" << zzz <<"\n";
//        std::cout << index[0] << " " <<  index[1] << " " << index[2] <<"\n";
//
//        std::cout << other.index[0] << " " <<  other.index[1] << " " << other.index[2] <<"\n";
//        
//        std::cout << level << " " << Z << " " << blockID << "      compare  with         " << other.level << " " << other.Z << " " << other.blockID <<"\n";
//        
//        assert ((blockID < other.blockID) == (zzz < other.Z) );
//        
//        return (zzz < other.Z);
//      }
    }  


  	BlockInfo(const int a_level,const double a_h, const double a_origin[3],int a_index[3], int a_myrank,TreePosition a_TreePos)
  	{
  		setup(a_level,a_h,a_origin,a_index,a_myrank,a_TreePos);
  	};


    void setup(const int a_level,const double a_h, const double a_origin[3],int a_index[3], int a_myrank, TreePosition a_TreePos) 
    {
        myrank    = a_myrank;
        TreePos   = a_TreePos;
        index [0] = a_index[0];
        index [1] = a_index[1];
        index [2] = a_index[2];
        state  = Leave;
        if (ptrBlock != NULL) h_gridpoint = h;

        level     = a_level;
        h         = a_h;
        origin[0] = a_origin[0];
        origin[1] = a_origin[1];
        origin[2] = a_origin[2];

        block_extent[0] = h* _BLOCKSIZE_;
        block_extent[1] = h* _BLOCKSIZE_;
        block_extent[2] = h* _BLOCKSIZE_;
        
        const int TwoPower = pow(2,level);
        const int Bmax[3] = {blocks_per_dim(0)*TwoPower,blocks_per_dim(1)*TwoPower,blocks_per_dim(2)*TwoPower};
 
        Z = forward(level,index[0],index[1],index[2]);

        for (int i=-1; i<2; i++)
        for (int j=-1; j<2; j++)
        for (int k=-1; k<2; k++)
        {
            Znei[i+1][j+1][k+1] = forward(level,(index[0]+i+Bmax[0])%Bmax[0],(index[1]+j+Bmax[1])%Bmax[1],(index[2]+k+Bmax[2])%Bmax[2]);
        }

        blockID = Encode(level,Z,index);

        if (level == 0)
        {
          Zparent = 0;
        }
        else
        {
          Zparent = forward(level-1,(index[0]/2+Bmax[0])%Bmax[0],(index[1]/2+Bmax[1])%Bmax[1],(index[2]/2+Bmax[2])%Bmax[2]);
        }


        for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
        for (int k=0; k<2; k++)
        {
          Zchild[i][j][k] = forward(level+1, 2*index[0]+i,2*index[1]+j,2*index[2]+k);
        }



    }

    int Znei_(int i, int j, int k) const
    {
      assert(abs(i)<=1);
      assert(abs(j)<=1);
      assert(abs(k)<=1);
      return Znei[1+i][1+j][1+k];
    }

};

}//namespace AMR_CUBISM
