cmake_minimum_required(VERSION 2.8.3)
project(moth_tracker)

find_package( OpenCV REQUIRED )

# Perform static build
#set(BUILD_SHARED_LIBS OFF)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(CMAKE_EXE_LINKER_FLAGS "-static")

## Build ##
include_directories(
  ${OpenCV_INCLUDE_LIBRARIES}
)

## Declare a cpp executable
#add_executable(backsub backsub.cpp)
add_executable(moth_tracker moth_tracker.cpp)
add_executable(test Test/test.cpp)

## Specify libraries to link a library or executable target against
#target_link_libraries(backsub
#  ${OpenCV_LIBRARIES}
#)
target_link_libraries(moth_tracker
  ${OpenCV_LIBRARIES}
)
target_link_libraries(test
  ${OpenCV_LIBRARIES}
)
# Set Linker flags
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-opencv")

## Install ##
install(TARGETS moth_tracker test
  RUNTIME DESTINATION bin
)

