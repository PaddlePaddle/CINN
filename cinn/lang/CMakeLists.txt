set(srcs buffer.cc compute.cc placeholder.cc tensor.cc module.cc lower.cc builtin.cc
        lower_impl.cc
        )

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/lang/${cpp}"
      CACHE INTERNAL "")
endforeach()

cc_test(test_compute SRCS compute_test.cc DEPS core)
cc_test(test_placeholder SRCS placeholder_test.cc DEPS core)
cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cc_test(test_lower SRCS lower_test.cc DEPS core)
