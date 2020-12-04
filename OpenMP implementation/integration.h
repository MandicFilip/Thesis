#ifndef INTEGRATION_HEADER

#define INTEGRATION_HEADER

#include"funtion.h"

#ifdef INTEGRATION_FUNCTION

struct OptFunctionParameters;

double opt_function(double* positions, struct OptFunctionParameters* opt_fun_params);

double calc_error(double result, struct OptFunctionParameters* opt_fun_params);

struct OptFunctionParameters* create_opt_fin_params(int m, int n);

void alter_opt_fin_params(struct OptFunctionParameters* ptr, int m, int n);

void free_opt_fin_params(struct OptFunctionParameters* ptr);

#endif

#endif // !INTEGRATION_HEADER
