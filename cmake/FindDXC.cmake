find_path(DXC_INCLUDE_DIR "dxc/dxcapi.h" PATH_SUFFIXES include)
find_path(DXC_DLL_DIR "dxcompiler.dll" PATH_SUFFIXES bin)
      
set(DXC_LIBRARIES     )
set(DXC_INCLUDE_DIRS ${DXC_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DXC  DEFAULT_MSG DXC_INCLUDE_DIR DXC_DLL_DIR)

mark_as_advanced(DXC_INCLUDE_DIR DXC_DLL_DIR)
