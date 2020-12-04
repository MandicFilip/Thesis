#ifndef SUM_SQUARES_HEADER

#define SUM_SQUARES_HEADER

#include"funtion.h"

#ifdef SUM_SQUARES_FUNCTION

double opt_function(double* positions, struct OptFunctionParameters* opt_fun_params);

struct OptFunctionParameters;

double calc_error(double result, struct OptFunctionParameters* opt_fun_params);

struct OptFunctionParameters* create_opt_fin_params(int n);

void free_opt_fin_params(struct OptFunctionParameters* ptr);


#endif

#endif // !SUM_SQUARES_HEADER
