# local-inference-batched-bench-impl: batched-bench logic, reusable by app

set(TARGET local-inference-batched-bench-impl)

add_library(${TARGET} batched-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# local-inference-batched-bench executable

set(TARGET local-inference-batched-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE local-inference-batched-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
