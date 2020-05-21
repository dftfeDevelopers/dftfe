// Copy of elpa/elpa.h that redefines complex so that
// the c99 headers of elpa can be included within C++
#ifndef ELPA_H
#define ELPA_H

#include <limits.h>
#include <complex.h>

#include <elpa/elpa_version.h>

struct elpa_struct;
typedef struct elpa_struct *elpa_t;

struct elpa_autotune_struct;
typedef struct elpa_autotune_struct *elpa_autotune_t;

#ifdef __cplusplus
#define complex _Complex
#endif
#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated_c_api.h>
#include <elpa/elpa_generated.h>
#include <elpa/elpa_generic.h>
#ifdef __cplusplus
#undef complex
#endif

const char *elpa_strerr(int elpa_error);

#endif
