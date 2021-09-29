INCLUDE(ExternalProject)

SET(ABSL_SOURCES_DIR ${THIRD_PARTY_PATH}/absl)
SET(ABSL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/absl)
SET(ABSL_INCLUDE_DIR "${ABSL_INSTALL_DIR}/include" CACHE PATH "absl include directory." FORCE)

SET(ABSL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

INCLUDE_DIRECTORIES(${ABSL_INCLUDE_DIR})

SET(ABSL_REPOSITORY "https://github.com/abseil/abseil-cpp.git")
SET(ABSL_TAG "20210324.2")

SET(OPTIONAL_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
        "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

ExternalProject_Add(
        extern_absl
        ${EXTERNAL_PROJECT_LOG_ARGS}
        DEPENDS gflags
        GIT_REPOSITORY  ${ABSL_REPOSITORY}
        GIT_TAG         ${ABSL_TAG}
        PREFIX          ${ABSL_SOURCES_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS      ${OPTIONAL_ARGS}
        -DCMAKE_INSTALL_PREFIX=${ABSL_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=${ABSL_INSTALL_DIR}/lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DWITH_GFLAGS=ON
        -Dgflags_DIR=${GFLAGS_INSTALL_DIR}/lib/cmake/gflags
        -DBUILD_TESTING=OFF
        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
        ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ABSL_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR:PATH=${ABSL_INSTALL_DIR}/lib
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)


function(ABSL_IMPORT_LIB lib_name)
    ADD_LIBRARY("absl_${lib_name}" STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET "absl_${lib_name}" PROPERTY IMPORTED_LOCATION ${ABSL_INSTALL_DIR}/lib/libabsl_${lib_name}.a)
    ADD_DEPENDENCIES("absl_${lib_name}" extern_absl)
endfunction(ABSL_IMPORT_LIB)

# It may be more convinent if we just include all absl libs
set(ABSL_LIB_NAMES
  base
  hash
  wyhash
  city
  strings
  throw_delegate
  bad_any_cast_impl
  bad_optional_access
  bad_variant_access
  raw_hash_set
  )
set(ABSL_LIBS "")
foreach(lib_name ${ABSL_LIB_NAMES})
  ABSL_IMPORT_LIB(${lib_name})
  list(APPEND ABSL_LIBS "absl_${lib_name}")
endforeach()

