# local-inference-cli-impl: CLI logic, reusable by app

set(TARGET local-inference-cli-impl)

add_library(${TARGET} cli.cpp
                      cli-client.cpp
                      cli-context.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_link_libraries(${TARGET} PUBLIC local-inference-server-impl llama-common ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# local-inference-cli executable

set(TARGET local-inference-cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE local-inference-cli-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
