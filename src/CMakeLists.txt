cmake_minimum_required (VERSION 3.10)
project (Dx12NN)

# Include macros
include(${CMAKE_SOURCE_DIR}/../cmake/macros.cmake)
 
# Additional places where to look for Find*.cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")
  
# Get subdirectories in sdk folder
SUBDIRLIST(SDK_SUBDIRS "${CMAKE_SOURCE_DIR}/../sdk")

# Create a list of sdk subdirs (including sdk folder)
SET(SDK_SUBDIRS_CONCAT "${CMAKE_SOURCE_DIR}/../sdk")
FOREACH(subdir ${SDK_SUBDIRS})
    set(SDK_SUBDIRS_CONCAT ${SDK_SUBDIRS_CONCAT} "${CMAKE_SOURCE_DIR}/../sdk/${subdir}")
ENDFOREACH()
    
print_list_contents("SDK Subdirectories:" SDK_SUBDIRS_CONCAT)

# Additional places where to look for libraries - all our SDK folders
set(CMAKE_PREFIX_PATH ${SDK_SUBDIRS_CONCAT} CACHE PATH "Paths to dependencies")
 
 # Add project subdirs
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})                    
FOREACH(subdir ${SUBDIRS})
    ADD_SUBDIRECTORY(${subdir})
ENDFOREACH()
              
file(GLOB_RECURSE INCS "*.h")
file(GLOB_RECURSE CPPS "*.cpp")
file(GLOB_RECURSE SHADERS "*.hlsl")

# Create folders in VS Project
FOREACH(subdir ${SUBDIRS})
  FILE(GLOB_RECURSE TEMP_SRCS 
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
      ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/*.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/*.h) 
  
  SOURCE_GROUP(${subdir} FILES ${TEMP_SRCS}) 
ENDFOREACH()  
       
find_package(GLM REQUIRED)    
find_package(STB REQUIRED)                             
find_package(ImGui REQUIRED Sources)
find_package(D3DX12 REQUIRED)
find_package(DXC REQUIRED)

include_directories(${GLM_INCLUDE_DIRS}) 
include_directories(${STB_INCLUDE_DIRS}) 
include_directories(${ImGui_INCLUDE_DIR}) 
include_directories(${D3DX12_INCLUDE_DIR}) 
include_directories(${DXC_INCLUDE_DIR}) 

# Put imgui files into separate VS folder
FILE(GLOB_RECURSE IMGUI_SRCS 
      ${ImGui_INCLUDE_DIR}/*.cpp 
      ${ImGui_INCLUDE_DIR}/*.h) 

SOURCE_GROUP(imgui FILES ${IMGUI_SRCS})

# Set Unicode support for MSVC project
if(MSVC)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

add_definitions (-DXC_DONT_COMPILE_FXC)

set(CMAKE_CONFIGURATION_TYPES Debug Release)

add_executable(Dx12NN WIN32 ${CPPS} ${INCS})

# Set default VS project 
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Dx12NN)

# Add custom target to include shaders in Visual Studio project (SOURCE_GROUP only works when source files are actually used in some target)
add_custom_target(shaders SOURCES ${SHADERS})

target_link_libraries(Dx12NN
    PUBLIC ImGui::ImGui
    PRIVATE ImGui::Sources)
    
# Set target directory for copying. Can refer to target which hasn't been added yet.
set(COPY_TARGET_DIR $<TARGET_FILE_DIR:Dx12NN>)
post_build_make_dir_link(Dx12NN ${PROJECT_SOURCE_DIR}/shaders  ${COPY_TARGET_DIR}/shaders)
post_build_make_dir_link(Dx12NN ${PROJECT_SOURCE_DIR}/assets  ${COPY_TARGET_DIR}/assets)
if(MSVC)
    post_build_make_dir_link(Dx12NN ${PROJECT_SOURCE_DIR}/shaders  ${PROJECT_BINARY_DIR}/shaders)
    post_build_make_dir_link(Dx12NN ${PROJECT_SOURCE_DIR}/assets  ${PROJECT_BINARY_DIR}/assets)
endif(MSVC)

# Copy dlls to target dir
add_custom_command(TARGET Dx12NN POST_BUILD   
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${DXC_DLL_DIR}/dxcompiler.dll"    
        $<TARGET_FILE_DIR:Dx12NN>)     

add_custom_command(TARGET Dx12NN POST_BUILD   
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${DXC_DLL_DIR}/dxil.dll"    
        $<TARGET_FILE_DIR:Dx12NN>)    
