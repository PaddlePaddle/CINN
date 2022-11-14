IF(NOT WITH_CUDA)
  SET(JITIFY_FOUND OFF)
  RETURN()
ENDIF()

include(ExternalProject)

set(JITIFY_SOURCE_PATH ${THIRD_PARTY_PATH}/install/jitify)
set(JITIFY_STL_HEADERS ${THIRD_PARTY_PATH}/install/jitify/stl_headers)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${JITIFY_STL_HEADERS})
if (NOT EXISTS ${JITIFY_STL_HEADERS})
  message(FATAL_ERROR "Directory creation failed: ${JITIFY_STL_HEADERS}")
endif()

ExternalProject_Add(
  external_jitify
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/NVIDIA/jitify.git"
  GIT_TAG master
  PREFIX ${THIRD_PARTY_PATH}/jitify
  SOURCE_DIR ${JITIFY_SOURCE_PATH}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)

include_directories(${JITIFY_SOURCE_PATH})
add_definitions(-DNVRTC_STL_PATH="${JITIFY_STL_HEADERS}")
message(STATUS "Jitify header files path: ${JITIFY_STL_HEADERS}")

add_library(extern_jitify INTERFACE)
add_dependencies(extern_jitify external_jitify)
set(jitify_deps extern_jitify)
