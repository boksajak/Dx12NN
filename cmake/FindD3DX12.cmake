find_path(D3DX12_INCLUDE_DIR "d3dx12.h")

set(D3DX12_LIBRARIES     )
set(D3DX12_INCLUDE_DIRS ${D3DX12_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(D3DX12  DEFAULT_MSG D3DX12_INCLUDE_DIR)

mark_as_advanced(D3DX12_INCLUDE_DIR)
