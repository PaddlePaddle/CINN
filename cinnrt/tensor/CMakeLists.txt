set(srcs
  tensor_map.cc
  tensor_shape.cc
  tensor_metadata.cc
  dense_host_tensor.cc
  dense_tensor_view.cc
  )

foreach(cpp ${srcs})
  set(core_src
    "${core_src};cinnrt/tensor/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
