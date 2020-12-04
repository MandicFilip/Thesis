#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<string.h>

#include"funtion.h"
#include"sum_squares.h"
#include"integration.h"
#include"util.h"
#include"pso.h"

#define MIN_NUMBER_OF_ARGUMENTS_SUM_SQUARES 6
#define MIN_NUMBER_OF_ARGUMENTS_INTEGRATION 7

int main(int argc, char** args)
{
#ifdef SUM_SQUARES_FUNCTION
	if (argc < MIN_NUMBER_OF_ARGUMENTS_SUM_SQUARES)
	{
		printf("Error, not enough parameters!");
		return -1;
	}

	int swarm_size = atoi(args[1]);

	int dimension = atoi(args[2]);

	int iterations_number = atoi(args[3]);
	int log_density = atoi(args[4]);

	struct OptFunctionParameters* opt_params = create_opt_fin_params(dimension);

	char* filename = args[5];
#endif 

#ifdef INTEGRATION_FUNCTION
	if (argc < MIN_NUMBER_OF_ARGUMENTS_INTEGRATION)
	{
		printf("Error, not enough parameters!");
		return -1;
	}

	int swarm_size = atoi(args[1]);

	int Mmax = atoi(args[2]);
	int N = atoi(args[3]);
	
	int dimension = N * 2;

	int iterations_number = atoi(args[4]);
	int log_density = atoi(args[5]);

	struct OptFunctionParameters* opt_params = create_opt_fin_params(Mmax, N);

	char* filename = args[6];
#endif

	struct PerformanceData* performance_data = create_performance_struct(iterations_number);

	double* result = (double*)malloc((dimension + 1) * sizeof(double));

	if (!opt_params || !performance_data || !result)
	{
		printf("Error, failed allocation for performance struct, parameters for optimization function or result!\n");
		return -1;
	}

	setup_log_file(filename);

	printf("The right answer is: 0\n\n");
	report_algorithm_parameters(swarm_size, dimension, iterations_number, log_density);
	
	//ALGORITHM CALL
	int error = exec_pso(swarm_size, dimension, iterations_number, log_density, opt_params, result, performance_data);

	if (error)
	{
		printf("Error running pso! Terminating the program!\n");
	}
	else if (result)
	{
		generate_final_report(result, swarm_size, dimension, iterations_number, log_density, performance_data);
	}

	close_log_file();

	free_performance_struct(performance_data);
	free_opt_fin_params(opt_params);
	free(result);

	return 0;
}
