cmake_minimum_required(VERSION 3.5) # actually is 3.10

# set the project name
project(vessel_synthesis)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add all libraries needed for compilation
set(VTK_8_1_libs vtkCommonCore-8.1 vtkCommonDataModel-8.1 vtkCommonExecutionModel-8.1 vtkFiltersModeling-8.1 vtkIOCore-8.1 vtkIOLegacy-8.1 vtkIOXML-8.1 vtkIOGeometry-8.1 vtkInfovisCore-8.1 vtkFiltersGeneral-8.1 vtkFiltersCore-8.1 vtkCommonTransforms-8.1 vtkIOXMLParser-8.1)

find_path(VITA_INCLUDE_DIRS core/GeneratorData.h
          HINTS .. ENV VITA_PATH
          PATH_SUFFIXES include/vita_source )
find_path(VITA_LIBRARY_DIRS libVItA.a libVItA.so
          HINTS .. ENV VITA_PATH
          PATH_SUFFIXES lib )
find_path(VTK_INCLUDE_DIRS vtkAbstractArray.h
          HINTS .. ENV VITA_PATH /usr/local/
          PATH_SUFFIXES vita_build/include/vtk-8.1 include/vtk-8.1)
find_path(VTK_LIBRARY_DIRS libvtkCommonCore-8.1.so
          HINTS .. ENV VITA_PATH /usr/local/
          PATH_SUFFIXES vita_build/lib lib)

link_directories(${VITA_LIBRARY_DIRS} ${VTK_LIBRARY_DIRS})

# add the executable
add_executable(vessel_synthesis vessel_synthesis.cpp)

# link libraries
target_link_libraries(vessel_synthesis VItA ${VTK_8_1_libs})

# add all include directories needed for compilation
target_include_directories(vessel_synthesis PUBLIC
                           "${VITA_INCLUDE_DIRS}"
                           "${VTK_INCLUDE_DIRS}"
                           )

