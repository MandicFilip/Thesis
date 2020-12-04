#include<stdlib.h>
#include<stdio.h>
#include<float.h>
#include<time.h>
#include<string.h>
#include<omp.h>

#include"funtion.h"
#include"pso.h"
#include"util.h"
#include"sum_squares.h"
#include"integration.h"

#define C1 1.494
#define C2 1.494

#define DT 1.0
#define W 0.729

#define MAX_VELOCITY 0.2

#define PSO_OK 0
#define PSO_ERROR -1

//limits for parallel implementation on GPU, due to Curand Mersenne Twister
#define THREAD_BLOCK_SIZE_IN_CUDA 256
#define MAX_NUMBER_OF_BLOCKS 200

#define CHECK_ALLOC(x) \
	if (!(x)) \
	{ \
		fprintf(stderr, "Bad memory allocation on position: %s:%d", __FILE__, __LINE__); \
		exit(-1); \
	}

struct Agent
{
	double* position;
	double* velocity;
	double value;
	double error;
	double* local_best;
};

struct Parameters
{
	int dimension;

	int swarm_size;
	struct Agent* agents;
	double* global_best_position;
	double global_best_value;
	double min_error;

	double max_velocity; //domen scaled max velocity
};

void free_parameters(struct Parameters* parameters)
{
	for (int i = 0; i < parameters->swarm_size; i++)
	{
		free(parameters->agents[i].position);
		free(parameters->agents[i].velocity);
		free(parameters->agents[i].local_best);
	}

	free(parameters->agents);
	free(parameters->global_best_position);

	free(parameters);
}

struct Parameters* create_parameters(int swarm_size, int dimension)
{
	struct Parameters* parameters; 	
	CHECK_ALLOC(parameters = (struct Parameters*) malloc(sizeof(struct Parameters)));

	//init_random(); //SET RANDOM SEED FOR RANDOM GEN

	parameters->max_velocity = MAX_VELOCITY * (DOMAIN_MAX - DOMAIN_MIN);

	parameters->swarm_size = swarm_size;
	parameters->dimension = dimension;

	CHECK_ALLOC(parameters->global_best_position = (double*)malloc(dimension * sizeof(double)));

	for (int i = 0; i < dimension; i++)
	{
#ifdef SUM_SQUARES_FUNCTION
	parameters->global_best_position[i] = (double)((get_random_double(0) - 0.5) * (DOMAIN_MAX - DOMAIN_MIN));

#endif // SUM_SQUARES_FUNCTION

#ifdef INTEGRATION_FUNCTION
	parameters->global_best_position[i] = get_random_double(0) * ((double)DOMAIN_MAX - (double)DOMAIN_MIN);
#endif // INTEGRATION_FUNCTION
	}

	parameters->global_best_value = INT_MAX; //not important
	parameters->min_error = INT_MAX; // big number

	CHECK_ALLOC(parameters->agents = (struct Agent*)malloc(swarm_size * sizeof(struct Agent)));

	//init parameters
	for (int i = 0; i < swarm_size; i++)
	{
		CHECK_ALLOC(parameters->agents[i].position = (double*)malloc(dimension * sizeof(double)));
		for (int j = 0; j < dimension; j++)
		{
#ifdef SUM_SQUARES_FUNCTION
			parameters->agents[i].position[j] = (get_random_double(0) - 0.5) * ((double)DOMAIN_MAX - (double)DOMAIN_MIN);
#endif // SUM_SQUARES_FUNCTION

#ifdef INTEGRATION_FUNCTION
			parameters->agents[i].position[j] = get_random_double(0) * ((double)DOMAIN_MAX - (double)DOMAIN_MIN);
#endif // INTEGRATION_FUNCTION
		}

		CHECK_ALLOC(parameters->agents[i].velocity = (double*)malloc(dimension * sizeof(double)));
		for (int j = 0; j < dimension; j++)
		{
			parameters->agents[i].velocity[j] =
				(get_random_double(0) - 0.5) * parameters->max_velocity;
		}

		parameters->agents[i].value = 0; //not important
		parameters->agents[i].error = INT_MAX; // big number

		CHECK_ALLOC(parameters->agents[i].local_best = (double*)malloc(dimension * sizeof(double)));
		
		memcpy(parameters->agents[i].local_best, parameters->agents[i].position, dimension * sizeof(double));
	}

	return parameters;
}

double calc_velocity(struct Parameters* parameters, struct Agent* agent, int k, int id_thread)
{
	double rand_local = get_random_double(id_thread);
	double rand_global = get_random_double(id_thread);

	double inertion = W * agent->velocity[k];
	double local_effect = C1 * rand_local * (agent->local_best[k] - agent->position[k]);
	double sworm_effect = C2 * rand_global * (parameters->global_best_position[k] - agent->position[k]);

	double velocity = inertion + local_effect + sworm_effect;

	if (velocity > parameters->max_velocity)
	{
		velocity = parameters->max_velocity;
	}
	else if (velocity < -1 * parameters->max_velocity)
	{
		velocity = -1 * parameters->max_velocity;
	}

	return velocity;
}

double calc_position(struct Parameters* parameters, struct Agent* agent, int k)
{
	double new_position = agent->position[k] + DT * agent->velocity[k];

	if (new_position > DOMAIN_MAX)
	{
		new_position = DOMAIN_MAX - (new_position - DOMAIN_MAX);
	}
	else if (new_position < DOMAIN_MIN)
	{
		new_position = DOMAIN_MIN + (DOMAIN_MIN - new_position);
	}

	return new_position;
}

void call_iteration(struct Parameters* parameters, int index, struct OptFunctionParameters* opt_fun_params, int tid)
{
	struct Agent* agent = &parameters->agents[index];

	for (int i = 0; i < parameters->dimension; i++)
	{
		agent->velocity[i] = calc_velocity(parameters, agent, i, tid);
		agent->position[i] = calc_position(parameters, agent, i);
	}

	agent->value = opt_function(agent->position, opt_fun_params);
	//double error = calc_error(agent->value, opt_fun_params);
	double error = agent->value;

	if (error < agent->error)
	{
		agent->error = error;
		memcpy(agent->local_best, agent->position, parameters->dimension * sizeof(double));
	}
}

int find_best_agent(struct Parameters* parameters)
{
	double global_best_value = parameters->agents[0].error;
	int global_best_agent = 0;

	double local_best_value;
	int local_best_agent;
	int i = 0;

	#pragma omp parallel shared(parameters, global_best_agent, global_best_value) private(i, local_best_agent, local_best_value)
	{
		local_best_value = parameters->agents[0].error;
		local_best_agent = 0;

		#pragma omp for private(i)
		for (i = 0; i < parameters->swarm_size; i++)
		{
			if (parameters->agents[i].error < local_best_value)
			{
				local_best_value = parameters->agents[i].error;
				local_best_agent = i;
			}
		}

		#pragma omp critical
		{
			if (local_best_value < global_best_value)
			{
				global_best_value = local_best_value;
				global_best_agent = local_best_agent;
			}
		}

	}

	return global_best_agent;
}

int exec_pso(unsigned swarm_size, unsigned dimension, unsigned turns_num, unsigned log_density,
	struct OptFunctionParameters* opt_fun_params, double* result, struct PerformanceData* performance_data)
{
	clock_t pso_begin_time = clock();

	unsigned long long max_swarm_size = MAX_NUMBER_OF_BLOCKS * THREAD_BLOCK_SIZE_IN_CUDA;

	if (!swarm_size || swarm_size >= max_swarm_size)
	{
		fprintf(stderr, "Error! Swarm size is out of range!\n");
		return PSO_ERROR;
	}
	if (!opt_fun_params || !result)
	{
		fprintf(stderr, "Error! Nullptr passed as one of the parameters!\n");
		return PSO_ERROR;
	}
	if (!dimension || !turns_num || !log_density)
	{
		fprintf(stderr, "Error! Dimension, number of iterations and log_density must be > 0!\n");
		return PSO_ERROR;
	}

	struct Parameters* parameters = create_parameters(swarm_size, dimension);
	if (!parameters)
	{
		printf("Error creating parameters, returned value is not correct! Algorithm has't been executed!");
		return PSO_ERROR;
	}

	clock_t pso_begin_iterations = clock();

	for (int i = 0; i < turns_num; i++)
	{
		clock_t pso_begin_current_iteration = clock();

		int j = 0;

		//swarm update
		#pragma omp parallel for shared(parameters, opt_fun_params) private(j)
		for (j = 0; j < parameters->swarm_size; j++)
		{
			int tid = omp_get_thread_num();
			call_iteration(parameters, j, opt_fun_params, tid);
		}

		clock_t pso_current_iteration_sync_done = clock();

		int best_agent = find_best_agent(parameters);

		int is_new_best = 0;
		if (parameters->agents[best_agent].error < parameters->min_error)
		{
			parameters->min_error = parameters->agents[best_agent].error;
			parameters->global_best_value = parameters->agents[best_agent].value;
			memcpy(parameters->global_best_position, parameters->agents[best_agent].local_best, parameters->dimension * sizeof(double));

			is_new_best = 1;
		}

		clock_t pso_current_iteration_best_done = clock();

		log_positions(parameters->global_best_position, dimension, i);
		log_results(parameters->global_best_value, i, log_density);

		clock_t pso_end_current_iteration = clock();

		calc_turn_times(performance_data, i, is_new_best, pso_begin_current_iteration,
			pso_current_iteration_sync_done, pso_current_iteration_best_done, pso_end_current_iteration); // PERFORMANCE CALC FUNCTION

	}
	clock_t pso_end_iterations = clock();

	result[0] = parameters->global_best_value;

#ifdef INTEGRATION_FUNCTION
	for (int i = 0; i < dimension; i++)
	{
		result[i + 1] = parameters->global_best_position[i];
	}
#endif // INTEGRATION_FUNCTION


	free_parameters(parameters);

	clock_t pso_end_time = clock();

	calc_algorithm_time(performance_data, turns_num, pso_begin_time, pso_begin_iterations, pso_end_iterations, pso_end_time);

	return PSO_OK;
}
