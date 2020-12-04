#include<math.h>
#include<stdlib.h>
#include<stdio.h>

#include<omp.h>

#include"funtion.h"

#ifdef SUM_SQUARES_FUNCTION

struct OptFunctionParameters
{
	int dimension;
};

struct OptFunctionParameters* create_opt_fin_params(int n)
{
	struct OptFunctionParameters* ptr = (OptFunctionParameters*) malloc(sizeof(struct OptFunctionParameters));

	if (!ptr)
	{
		fprintf(stderr, "Error allocating memory for parameters for optimization function");
		return NULL;
	}

	ptr->dimension = n;
	return ptr;
}

void free_opt_fin_params(struct OptFunctionParameters* ptr)
{
	free(ptr);
}

double opt_function(double* positions, struct OptFunctionParameters* opt_fun_params)
{
	double result = 0;
	for (int i = 0; i < opt_fun_params->dimension; i++)
	{
		result += positions[i] * positions[i];
	}
	return result;
}

double calc_error(double result, struct OptFunctionParameters* opt_fun_params)
{
	return result;
}

#endif