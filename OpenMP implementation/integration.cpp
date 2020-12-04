#include<math.h>
#include<stdlib.h>
#include<stdio.h>

#include<omp.h>

#include"funtion.h"

#ifdef INTEGRATION_FUNCTION

struct OptFunctionParameters
{
	int Mmax;
	int N;
};

struct OptFunctionParameters* create_opt_fin_params(int m, int n)
{
	struct OptFunctionParameters* ptr = (OptFunctionParameters*)malloc(sizeof(struct OptFunctionParameters));

	if (!ptr)
	{
		fprintf(stderr, "Error allocating memory for parameters for optimization function");
		return NULL;
	}

	ptr->Mmax = m;
	ptr->N = n;
	return ptr;
}

void free_opt_fin_params(struct OptFunctionParameters* ptr)
{
	free(ptr);
}

void alter_opt_fin_params(struct OptFunctionParameters* ptr, int m, int n)
{
	ptr->Mmax = m;
	ptr->N = n;
}

double opt_function(double* positions, struct OptFunctionParameters* opt_fun_params)
{
	double outer_loop = 0;
	for (int m = 1; m < opt_fun_params->Mmax + 1; m++)
	{
		double correct = 1 / (double)(m + 1) / (double)(m + 1);
		
		double inner_sum = 0;
		for (int k = 0; k < opt_fun_params->N; k++)
		{
			//printf("pos[%d] = %lf\n", k, positions[k]);
			inner_sum += positions[k + opt_fun_params->N] * pow(positions[k], m) * log(positions[k]);
		}
		
		double up = fabs(correct + inner_sum);
		outer_loop += up / correct;
	}
	return outer_loop / opt_fun_params->Mmax;
}

double calc_error(double result, struct OptFunctionParameters* opt_fun_params)
{
	return result;
}

#endif