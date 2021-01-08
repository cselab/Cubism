#pragma once

#include <cmath>
#include <map>
#include <vector>

namespace cubism
{

template <typename T>
class GrowingVector
{

   size_t pos;
   size_t s;

 public:
   std::vector<T> v;
   GrowingVector() { pos = 0; }
   GrowingVector(size_t size) { resize(size); }
   GrowingVector(size_t size, T value) { resize(size, value); }

   void resize(size_t new_size, T value)
   {
      v.resize(new_size, value);
      s = new_size;
   }
   void resize(size_t new_size)
   {
      v.resize(new_size);
      s = new_size;
   }

   size_t size() { return s; }

   void clear()
   {
      pos = 0;
      s   = 0;
   }

   void push_back(T value)
   {
      if (pos < v.size()) v[pos] = value;
      else
         v.push_back(value);
      pos++;
      s++;
   }

   T *data() { return v.data(); }

   T &operator[](size_t i) { return v[i]; }

   T &back() { return v[pos - 1]; }

   void EraseAll()
   {
      v.clear();
      pos = 0;
      s   = 0;
   }

   ~GrowingVector() { v.clear(); }
};

} // namespace cubism