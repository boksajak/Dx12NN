set currrentDir=%CD%
mkdir vsbuild
cd vsbuild
cmake -G "Visual Studio 17 2022" -D CMAKE_INSTALL_PREFIX="%currrentDir%/build/"  ../src/ 
cd %currrentDir%
