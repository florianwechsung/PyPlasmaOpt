cmake_minimum_required(VERSION 3.5)
project(cppplasmaopt)

find_package (Python3 REQUIRED COMPONENTS NumPy)

set(CMAKE_CXX_FLAGS "-O3 -march=native")
find_package(OpenMP)


add_subdirectory(pybind11)



pybind11_add_module(${PROJECT_NAME} 
    cppplasmaopt/main.cpp cppplasmaopt/biot_savart_B.cpp cppplasmaopt/biot_savart_dB_by_dX.cpp
    cppplasmaopt/biot_savart_dB_by_dcoilcoeff.cpp cppplasmaopt/biot_savart_dB_by_dcoilcoeff_via_chainrule.cpp
    cppplasmaopt/biot_savart_d2B_by_dXdcoilcoeff.cpp)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_include_directories(${PROJECT_NAME} PRIVATE "xtensor/include" "xtensor-python/include" "xtl/include" "blaze" "cppplasmaopt" ${Python3_NumPy_INCLUDE_DIRS})


add_executable(profiling_biot_savart
    profiling/profile_biot_savart.cpp cppplasmaopt/biot_savart_B.cpp cppplasmaopt/biot_savart_dB_by_dX.cpp
    cppplasmaopt/biot_savart_dB_by_dcoilcoeff.cpp cppplasmaopt/biot_savart_dB_by_dcoilcoeff_via_chainrule.cpp
    cppplasmaopt/biot_savart_d2B_by_dXdcoilcoeff.cpp)
set_target_properties(profiling_biot_savart
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_include_directories(profiling_biot_savart PRIVATE  "xtensor/include" "xtensor-python/include" "xtl/include" "blaze" "cppplasmaopt" ${PYBIND11_INCLUDE_DIR} ${Python3_NumPy_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(profiling_biot_savart PUBLIC ${PYTHON_LIBRARIES})


if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(profiling_biot_savart PUBLIC OpenMP::OpenMP_CXX)
endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
