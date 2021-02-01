set(srcs
        basic_kernels.cc
        tensor_shape_kernels.cc
        tensor_kernels.cc
        control_flow_kernels.cc
        )

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinnrt/kernel/${cpp}"
            CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
