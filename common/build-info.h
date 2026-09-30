#pragma once

#include <cstdio>

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);

// the version of inmoloc, and the version with the commit (e.g. "0.1.0-dev-46cafc393")
const char * inmoloc_version(void);
const char * llama_build_info(void);

void llama_print_build_info(const char *, FILE * = stderr);
