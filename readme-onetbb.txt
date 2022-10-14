To use thread-safe unordered_map from oneTBB:
1. Run the script: ./install_oneTBB.sh
2. 
   export ONETBBROOT="[path to Cubism]/Cubism/oneTBB/build/installed_onetbb"
   example:
       export ONETBBROOT="/users/chatzima/3D_CUP/Cubism/oneTBB/build/installed_onetbb"
3. export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ONETBBROOT}
4. define the variable CUBISM_USE_ONETBB. For an ordinary Makefile, include these lines:
      LIBS     += -L$(ONETBBROOT)/lib64 -ltbb
      CPPFLAGS += -I$(ONETBBROOT)include
      CPPFLAGS += -DCUBISM_USE_ONETBB
