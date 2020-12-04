set(srcs
        basic_kernels.cc
        tensor_shape_kernels.cc
        tensor_kernels.cc
        )

foreach(cpp ${srcs})
    set(core_src
            "${core_src};cinnrt/kernel/${cpp}"
            CACHE INTERNAL "")
endforeach()

set(includes
    basic_kernels.h
    tensor_kernels.h
    tensor_shape_kernels.h
    )

foreach(header ${includes})
  set(core_includes "${core_includes};cinnrt/kernel/${header}" CACHE INTERNAL "")
endforeach()
