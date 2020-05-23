set(srcs
        dot.cc
        binary.cc
        elementwise.cc
        )

cc_test(test_hlir_primitive_dot SRCS dot_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};${CMAKE_SOURCE_DIR}/cinn/hlir/instruction/primitive/${cpp}"
            CACHE INTERNAL "")
endforeach()
