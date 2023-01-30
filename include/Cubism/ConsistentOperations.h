#pragma once

//Functions suggested in "Numerical symmetry-preserving techniques for low-dissipation shock-capturing schemes" equation (19) for summation that does not favor a spatial direction

namespace cubism {
#ifdef PRESERVE_SYMMETRY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function" //silence annoying warnings
static Real ConsistentSum(const Real a, const Real b, const Real c)
{
   const Real s1 = a + (b+c);
   const Real s2 = b + (c+a);
   const Real s3 = c + (a+b);
   return 0.5*(std::min({s1,s2,s3})+std::max({s1,s2,s3}));
}
static Real ConsistentSum(const Real a, const Real b, const Real c, const Real d)
{
   const Real s1 = (a+b)+(c+d);
   const Real s2 = (c+a)+(b+d);
   const Real s3 = (b+c)+(a+d);
   return 0.5*(std::min({s1,s2,s3})+std::max({s1,s2,s3}));
}

static Real ConsistentAverage(const Real e000, const Real e001, const Real e010, const Real e011, const Real e100, const Real e101, const Real e110, const Real e111)
{
   const Real a = e000 + e111;
   const Real b = e001 + e110;
   const Real c = e010 + e101;
   const Real d = e100 + e011;
   return 0.125*ConsistentSum(a,b,c,d);
}

template <typename T>
static T ConsistentAverage(const T e000, const T e001, const T e010, const T e011, const T e100, const T e101, const T e110, const T e111)
{
   const T a = e000 + e111;
   const T b = e001 + e110;
   const T c = e010 + e101;
   const T d = e100 + e011;
   return 0.125*ConsistentSum(a,b,c,d);
}

template <typename T>
static T ConsistentSum(const T a, const T b, const T c)
{
   T s1 = a + (b+c);
   T s2 = b + (c+a);
   T s3 = c + (a+b);
   T result;
   for (int i = 0 ; i < T::DIM ; i ++) result.member(i) = 0.5*(std::min({s1.member(i),s2.member(i),s3.member(i)})+std::max({s1.member(i),s2.member(i),s3.member(i)}));
   return result;
}

template <typename T>
static T ConsistentSum(const T a, const T b, const T c, const T d)
{
   T s1 = (a+b)+(c+d);
   T s2 = (c+a)+(b+d);
   T s3 = (b+c)+(a+d);
   T result;
   for (int i = 0 ; i < T::DIM ; i ++) result.member(i) = 0.5*(std::min({s1.member(i),s2.member(i),s3.member(i)})+std::max({s1.member(i),s2.member(i),s3.member(i)}));
   return result;
}
#pragma GCC diagnostic pop
#endif
}
