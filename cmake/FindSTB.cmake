find_path(STB_INCLUDE_DIR "stb_image.h")

set(STB_INCLUDE_DIRS ${STB_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STB DEFAULT_MSG STB_INCLUDE_DIR)

mark_as_advanced(STB_INCLUDE_DIR)
