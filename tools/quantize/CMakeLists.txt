# local-inference-quantize-impl: quantize logic, reusable by app

set(TARGET local-inference-quantize-impl)

add_library(${TARGET} quantize.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# local-inference-quantize executable

set(TARGET local-inference-quantize)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE local-inference-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
