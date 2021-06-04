#pragma once

#include "Common.h"

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>

CUBISM_NAMESPACE_BEGIN

struct MyClock
{
   int N;
   double t[100];
   double s[100];
   std::string name[100];

   double total_s;
   double total_f;

   void padTo(std::string &str, const size_t num, const char paddingChar = ' ')
   {
      if (num > str.size()) str.insert(0, num - str.size(), paddingChar);
   }

   MyClock() { reset(); }
   void reset()
   {
      N = 0;
      for (int i = 0; i < 100; i++) t[i] = 0;
   }
   void start(int i, std::string _name)
   {
      name[i] = std::move(_name);
      s[i]    = MPI_Wtime();
      N       = N > i ? N : i;
   }
   void finish(int i) { t[i] += MPI_Wtime() - s[i]; }

   void start_total() { total_s = MPI_Wtime(); }
   void finish_total() { total_f = MPI_Wtime() - total_s; }

   void display()
   {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      double m1, m2;
      MPI_Reduce(&total_f, &m1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&total_f, &m2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      double *mean    = new double[N];
      double *maximum = new double[N];
      MPI_Reduce(t, mean, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(t, maximum, N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (rank != 0)
      {
         delete[] mean;
         delete[] maximum;
         return;
      }
      std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      for (int i = 0; i < N; i++)
      {
         padTo(name[i], 40);
         mean[i] /= size;
         printf("%s    :  %8.4f (max)     %8.4f (mean) \n", name[i].c_str(), maximum[i], mean[i]);
      }
      std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      std::cout << "TOTAL TIME = " << m1 << " " << m2 << "\n";
      std::ofstream outfile;
      outfile.open("TIMES.txt", std::ios_base::app); // append instead of overwrite
      outfile << size << " " << m1 << " " << m2 << "\n";
      delete[] mean;
      delete[] maximum;
   }
};
extern MyClock Clock;

CUBISM_NAMESPACE_END
