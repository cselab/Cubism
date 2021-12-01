#pragma once

#include "Grid.h"
#include <map>
#include <omp.h>

namespace cubism
{

template <typename TBlock, typename ElementTypeT = typename TBlock::ElementType>
struct BlockCase
{
   typedef ElementTypeT ElementType;
   typedef TBlock BlockType;

   std::vector<std::vector<ElementType>> m_pData;
   unsigned int m_vSize[3];
   bool storedFace[6];
   int level;
   long long Z;

   BlockCase(bool a_storedFace[6], unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
   {
      m_vSize[0] = nSizeX;
      m_vSize[1] = nSizeY;
      m_vSize[2] = nSizeZ;

      storedFace[0] = a_storedFace[0];
      storedFace[1] = a_storedFace[1];
      storedFace[2] = a_storedFace[2];
      storedFace[3] = a_storedFace[3];
      storedFace[4] = a_storedFace[4];
      storedFace[5] = a_storedFace[5];

      m_pData.resize(6);

      for (int d = 0; d < 3; d++)
      {
         int d1 = (d + 1) % 3;
         int d2 = (d + 2) % 3;

         // assume everything is initialized to 0!!!!!!!!
         if (storedFace[2 * d]) m_pData[2 * d].resize(m_vSize[d1] * m_vSize[d2]);
         if (storedFace[2 * d + 1]) m_pData[2 * d + 1].resize(m_vSize[d1] * m_vSize[d2]);
      }
   }

   void SetupMetaData(int m, long long n)
   {
      level = m;
      Z     = n;
   }

   ~BlockCase() {}
};

template <typename TGrid, typename TBlock>
class FluxCorrection
{
 public:
   typedef TBlock BlockType;
   typedef typename BlockType::ElementType ElementType;
   typedef typename ElementType::RealType Real;
   typedef BlockCase<BlockType> Case;
   bool TimeIntegration;
   int rank{0};

 protected:
   std::map<std::array<long long, 2>, Case *> MapOfCases;
   TGrid *m_refGrid;
   std::vector<Case> Cases;
   bool xperiodic;
   bool yperiodic;
   bool zperiodic;
   std::array<int, 3> blocksPerDim;

 public:
   virtual void prepare(TGrid &grid)
   {
      if (grid.UpdateFluxCorrection == false) return;
      grid.UpdateFluxCorrection = false;

      Cases.clear();
      MapOfCases.clear();
      m_refGrid = &grid;
      std::vector<BlockInfo> & B = (*m_refGrid).getBlocksInfo();

      xperiodic = grid.xperiodic;
      yperiodic = grid.yperiodic;
      zperiodic = grid.zperiodic;
      blocksPerDim = (*m_refGrid).getMaxBlocks();
      std::array<int,6> icode = {1*2 + 3*1 + 9*1, 1*0 + 3*1 + 9*1, 1*1 + 3*2 + 9*1, 1*1 + 3*0 + 9*1, 1*1 + 3*1 + 9*2, 1*1 + 3*1 + 9*0};

      for (auto & info: B)
      {
	      m_refGrid->getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
        const int aux = 1<<info.level;

        const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
        const int  xskip = info.index[0]==0 ? -1 : 1;
        const int  yskip = info.index[1]==0 ? -1 : 1;
        const int  zskip = info.index[2]==0 ? -1 : 1;

        bool storeFace[6] = {false,false,false,false,false,false};
        bool stored = false;

        for (int f=0; f<6; f++)
        {
          const int code[3] = { icode[f]%3-1, (icode[f]/3)%3-1, (icode[f]/9)%3-1};
          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue;
          #if DIMENSION == 2
          if (code[2] != 0) continue;
          #endif

          BlockInfo infoNei =m_refGrid->getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));
          if (!m_refGrid->Tree(infoNei).Exists())
          {
            storeFace[ abs(code[0]) *  max(0,code[0]) + abs(code[1]) * (max(0,code[1])+2) + abs(code[2]) * (max(0,code[2])+4)  ] = true;
            stored = true;
          }
        }
        if (stored)
        {
          Cases.push_back(Case(storeFace,BlockType::sizeX,BlockType::sizeY,BlockType::sizeZ));
          Cases.back().SetupMetaData(info.level,info.Z);
        }
      }
      for (size_t i = 0; i < Cases.size() ; i ++ )
      {
        MapOfCases.insert(  std::pair<std::array <long long,2>,Case *> ({(long long)Cases[i].level,Cases[i].Z}, &Cases[i]) );
        m_refGrid->getBlockInfoAll(Cases[i].level,Cases[i].Z).auxiliary = &Cases[i];
      }
      m_refGrid->FillPos();
   }

   Case *GetCase(int level, long long Z)
   {
      std::array<long long, 2> tmp = {(long long)level, Z};

      auto search = MapOfCases.find(tmp);

      if (search == MapOfCases.end())
      {
         return nullptr;
      }
      assert((*search->second).level == level);
      assert((*search->second).Z == Z);
      return (search->second);
   }

   virtual void FillBlockCases(bool Integrate = true)
   {
      TimeIntegration = Integrate;
      // This assumes that the BlockCases have been filled by the user somehow...
      std::vector<BlockInfo> &B = (*m_refGrid).getBlocksInfo();

      std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1, 1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1, 1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

      #pragma omp parallel for
      for (size_t i = 0 ; i < B.size() ; i++)
      {
        BlockInfo & info = B[i];
        const int aux = 1 << info.level;
        const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
        const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
        const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
        const int  xskip = info.index[0] == 0 ? -1 : 1;
        const int  yskip = info.index[1] == 0 ? -1 : 1;
        const int  zskip = info.index[2] == 0 ? -1 : 1;

        for (int f = 0; f < 6; f++)
        {
          const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1, (icode[f] / 9) % 3 - 1};

          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue;
          #if DIMENSION == 2
          if (code[2] != 0) continue;
          #endif

          BlockInfo & infoNei = m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

          if (m_refGrid->Tree(infoNei).CheckFiner())
          {
            FillCase(info, code);

            const int myFace = abs(code[0]) * max(0, code[0]) + abs(code[1]) * (max(0, code[1]) + 2) + abs(code[2]) * (max(0, code[2]) + 4);
            std::array<long long, 2> temp = {(long long)info.level, info.Z};
            auto search             = MapOfCases.find(temp);
            assert(search != MapOfCases.end());
            Case &CoarseCase                     = (*search->second);
            std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
            const int d  = myFace / 2;
            const int d2 = min((d + 1) % 3, (d + 2) % 3);
            const int N2 = CoarseCase.m_vSize[d2];
            BlockType &block = *(BlockType *)info.ptrBlock;

            #if DIMENSION == 3
              if (TimeIntegration)
              {
                abort();
                #if 0
                // WARNING: tmp indices are tmp[z][y][x][Flow Quantity]!
                const double V = 1.0 / (info.h*info.h*info.h);
                const int d1 = max((d + 1) % 3, (d + 2) % 3);
                const int N1 = CoarseCase.m_vSize[d1];
                if (d == 0)
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeX - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    for (int e = 0 ; e < ElementType::DIM; e++)
                      block.tmp[i1][i2][j][e] += V*CoarseFace[i2 + i1 * N2].member(e);
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }
                else if (d == 1)
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeY - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    for (int e = 0 ; e < ElementType::DIM; e++)
                      block.tmp[i1][j][i2][e] += V*CoarseFace[i2 + i1 * N2].member(e);
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }
                else
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeZ - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    for (int e = 0 ; e < ElementType::DIM; e++)
                      block.tmp[j][i1][i2][e] += V*CoarseFace[i2 + i1 * N2].member(e);
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }
                #endif
              }
              else
              {
                // WARNING: tmp indices are tmp[z][y][x][Flow Quantity]!
                const int d1 = max((d + 1) % 3, (d + 2) % 3);
                const int N1 = CoarseCase.m_vSize[d1];
                if (d == 0)
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeX - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    block(j,i2,i1) += CoarseFace[i2 + i1 * N2];
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }
                else if (d == 1)
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeY - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    block(i2,j,i1) += CoarseFace[i2 + i1 * N2];
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }
                else
                {
                  const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeZ - 1;
                  for (int i1 = 0; i1 < N1; i1 ++)
                  for (int i2 = 0; i2 < N2; i2 ++)
                  {
                    block(i2,i1,j) += CoarseFace[i2 + i1 * N2];
                    CoarseFace[i2 + i1 * N2].clear();
                  }
                }               
              }
            #else
              assert(d!=2);
              if (d == 0)
              {
                const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeX - 1;
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(j,i2) += CoarseFace[i2];
                  CoarseFace[i2].clear();
                }
              }
              else //if (d == 1)
              {
                const int j = (myFace % 2 == 0) ? 0 : TBlock::sizeY - 1;
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(i2,j) += CoarseFace[i2];
                  CoarseFace[i2].clear();
                }
              }
            #endif
          }
        }
      }
   }

   void FillCase(BlockInfo & info, const int *const code)
   {
      const int myFace    = abs( code[0]) * max(0,  code[0]) + abs( code[1]) * (max(0,  code[1]) + 2) + abs( code[2]) * (max(0,  code[2]) + 4);
      const int otherFace = abs(-code[0]) * max(0, -code[0]) + abs(-code[1]) * (max(0, -code[1]) + 2) + abs(-code[2]) * (max(0, -code[2]) + 4);

      std::array<long long, 2> temp = {(long long)info.level, info.Z};
      auto search             = MapOfCases.find(temp);

      Case &CoarseCase = (*search->second);
      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

      assert(myFace / 2 == otherFace / 2);
      assert(search != MapOfCases.end());
      assert(CoarseCase.Z == info.Z);
      assert(CoarseCase.level == info.level);

      #if DIMENSION == 3
      for (int B = 0; B <= 3; B++) // loop over fine blocks that make up coarse face
      #else
      for (int B = 0; B <= 1; B++) // loop over fine blocks that make up coarse face
      #endif
      {
        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        #if DIMENSION == 3
        const long long Z = (*m_refGrid).getZforward(info.level + 1,
                  2 * info.index[0] + max(code[0], 0) + code[0] +(B % 2) * max(0, 1 - abs(code[0])),
                  2 * info.index[1] + max(code[1], 0) + code[1] +    aux * max(0, 1 - abs(code[1])),
                  2 * info.index[2] + max(code[2], 0) + code[2] +(B / 2) * max(0, 1 - abs(code[2])));
        #else
        const long long Z = (*m_refGrid).getZforward(info.level + 1,
                  2 * info.index[0] + max(code[0], 0) + code[0] +(B % 2) * max(0, 1 - abs(code[0])),
                  2 * info.index[1] + max(code[1], 0) + code[1] +    aux * max(0, 1 - abs(code[1])));
        #endif
        if (m_refGrid->Tree(info.level + 1, Z).rank() != rank) continue;
        auto search1 = MapOfCases.find({info.level + 1, Z});

        Case &FineCase                     = (*search1->second);
        std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];
        const int d   = myFace / 2;
        const int d1  = max((d + 1) % 3, (d + 2) % 3);
        const int d2  = min((d + 1) % 3, (d + 2) % 3);
        const int N1F = FineCase.m_vSize[d1];
        const int N2F = FineCase.m_vSize[d2];
        const int N1  = N1F;
        const int N2  = N2F;
        int base = 0;
        if      (B == 1)  base = (N2 / 2) + (0) * N2;
        else if (B == 2)  base = (0) + (N1 / 2) * N2;
        else if (B == 3)  base = (N2 / 2) + (N1 / 2) * N2;
        assert(search1 != MapOfCases.end());
        assert(N1F == (int)CoarseCase.m_vSize[d1]);
        assert(N2F == (int)CoarseCase.m_vSize[d2]);
        assert(FineFace.size() == CoarseFace.size());
        #if DIMENSION == 3
          for (int i1 = 0; i1 < N1; i1 += 2)
          for (int i2 = 0; i2 < N2; i2 += 2)
          {
            CoarseFace[base + (i2 / 2) + (i1 / 2) * N2] += FineFace[i2 +  i1    * N2] + FineFace[i2+1 +  i1    * N2] +
                                                           FineFace[i2 + (i1+1) * N2] + FineFace[i2+1 + (i1+1) * N2];
            FineFace[i2     +  i1      * N2].clear();
            FineFace[i2 + 1 +  i1      * N2].clear();
            FineFace[i2     + (i1 + 1) * N2].clear();
            FineFace[i2 + 1 + (i1 + 1) * N2].clear();
          }
        #else
          for (int i2 = 0; i2 < N2; i2 += 2)
          {
            CoarseFace[base + i2/2] += FineFace[i2] + FineFace[i2 + 1];
            FineFace[i2    ].clear();
            FineFace[i2 + 1].clear();
          }
        #endif
      }
   }
};

} // namespace cubism
