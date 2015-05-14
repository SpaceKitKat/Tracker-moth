cmake_minimum_required(VERSION 2.8.3)
project(moth_tracker)

find_package( OpenCV REQUIRED )
find_package( Boost 1.40 COMPONENTS program_options REQUIRED ) 

# Perform static build
#set(BUILD_SHARED_LIBS OFF)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(CMAKE_EXE_LINKER_FLAGS "-static")

#set( tracker_SRC progressbar.cpp moth_tracker.cpp )
#set( tracker_HDR progressbar.h )

## Build ##
include_directories(
  ${OpenCV_INCLUDE_LIBRARIES}
  ${Boost_INCLUDE_DIR}
)
#INCLUDE( ${QT_USE_FILE} )
#ADD_DEFINITIONS(${QT_DEFINITIONS}
# This macro invokes moc: processes headers and stores src files in _HDR_MOC 
#QT4_WRAP_CPP( tracker_HDR_MOC ${tracker_HDR} )

## Declare a cpp executable
add_executable(moth_tracker moth_tracker.cpp)
#add_executable(moth_tracker ${tracker_SRC} ${tracker_HRD_MOC})
#add_executable(test Test/test.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(moth_tracker
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
#  ${QT_LIBRARIES}
)
#target_link_libraries(test
#  ${OpenCV_LIBRARIES}
#)
# Set Linker flags
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-opencv")

## Install ##
#install(TARGETS moth_tracker test
#  RUNTIME DESTINATION bin
#)

