#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include<stdlib.h>
#include<stdio.h>
#include<float.h>
#include<time.h>
#include<math.h>

#include"util.h"

#define ARRAYS_IN_AGENT 5

#define FULL_MASK 0xffffffff

//curand thread safe limit
#define SWARM_UPDATE_BLOCK_SIZE 256
#define FIND_BEST_BLOCK_SIZE 512
#define COPY_BEST_BLOCK_SIZE 512
#define WARP_SIZE 32

//Swarm size limit - can be extanded to (2^31) - 1
#define MAX_NUMBER_OF_BLOCKS 1024

#define CURAND_SEED 1234

#define PSO_OK 0
#define PSO_ERROR -1

#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

//PSO CONSTANTS
#define DEFAULT_C1 1.494
#define DEFAULT_C2 1.494

#define DEFAULT_DT 1
#define DEFAULT_W 0.729

#define MAX_VELOCITY 0.2

//#define SUM_SQUARES_FUNCTION 1
#define INTEGRATION_FUNCTION 2

__constant__ double DOMAIN_MAX;
__constant__ double DOMAIN_MIN;

__constant__ double C1; //cognitive coefitient
__constant__ double C2; //social coefitient

__constant__ double DT; //time interval - will be 1 for easier calculation
__constant__ double MAX_AGENT_VELOCITY; //domen scaled max velocity
__constant__ double W; //inertion coefitient

//---------------------------------------------------------------------------------------------------------------------
//------------------------------------------------CUDA ERROR CHECK-----------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

inline void check_error(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}
#define CUDA_CALL(ans) { check_error((ans), __FILE__, __LINE__); }

//---------------------------------------------------------------------------------------------------------------------
//------------------------------------------------SUM SQUARES----------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

#ifdef SUM_SQUARES_FUNCTION

#define UP_DOMAIN_BORDER 1
#define DOWN_DOMAIN_BORDER -1

struct OptFunctionParameters
{
	int swarm_size;
	int dimension;
};

__device__ double opt_function
(double** positions, int agent, struct OptFunctionParameters* opt_fun_params, const double* cache)
{
	double result = 0;
	for (int i = 0; i < opt_fun_params->dimension; i++)
	{
		result += positions[i][agent] * positions[i][agent];
	}
	return result;
}

__device__ double calc_error(double result, struct OptFunctionParameters* opt_fun_params)
{
	return result;
}

#endif // SUM_SQUARES_FUNCTION

#ifdef INTEGRATION_FUNCTION

#define UP_DOMAIN_BORDER 1
#define DOWN_DOMAIN_BORDER 0

__constant__ int Mmax;
__constant__ int N;

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

__device__ double opt_function
(double** positions, int agent, struct OptFunctionParameters* opt_fun_params, double* const cache)
{
	double* const thread_cache = &cache[threadIdx.x * N * 2];
	for (int i = 0; i < N * 2; i++)
	{
		thread_cache[i] = positions[i][agent];
	}

	double outer_loop = 0;
	for (int m = 1; m < Mmax + 1; m++)
	{
		double nextM_square = (m + 1) * (m + 1);

		double inner_sum = 0;
		for (int k = 0; k < N; k++)
		{
			inner_sum += thread_cache[k + N] * pow(thread_cache[k], m) * log(thread_cache[k]);
		}

		outer_loop += fabs(1 + inner_sum * nextM_square);
	}
	return outer_loop / Mmax;
}

__device__ double calc_error(double result, struct OptFunctionParameters* opt_fun_params)
{
	return result;
}

void populate_command_arguments_to_cuda(int m, int n)
{
	CUDA_CALL(cudaMemcpyToSymbol(Mmax, &m, sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(N, &n, sizeof(int)));
}

#endif // INTEGRATION_FUNCTION


//---------------------------------------------------------------------------------------------------------------------
//------------------------------------------------PSO structs----------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

struct Agent
{
	double** position;
	double** velocity;
	double* value;
	double* error;
	double** local_best;
};

struct Parameters
{
	int swarm_size;
	int dimension;
	struct Agent agents;
	double* global_best_position;
	double global_best_value;
	double min_error;

	//reduction implementation data
	double* temp_error;
	double* results;

	int* temp_index;
	int* results_index;
};

struct CudaRunParameters
{
	int swarm_update_blocks_number;
	int swarm_update_threads_per_block;
	int swarm_update_shared_memory_per_block;

	int find_best_agent_blocks_number;
	int find_best_agent_threads_per_block;
	int find_best_agent_shared_memory_per_block;

	int copy_best_agent_blocks_number;
	int copy_best_agent_threads_per_block;
};

void populate_constant_memory()
{
	double domain_max = UP_DOMAIN_BORDER;
	double domain_min = DOWN_DOMAIN_BORDER;

	double c1 = DEFAULT_C1; //cognitive coefitient
	double c2 = DEFAULT_C2; //social coefitient

	double dt = DEFAULT_DT; //time interval - will be 1 for easier calculation
	double max_velocity = MAX_VELOCITY * (UP_DOMAIN_BORDER - DOWN_DOMAIN_BORDER); //domen scaled max velocity
	double w = DEFAULT_W; //inertion coefitient

	CUDA_CALL(cudaMemcpyToSymbol(C1, &c1, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(C2, &c2, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(DT, &dt, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(MAX_AGENT_VELOCITY, &max_velocity, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(W, &w, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(DOMAIN_MAX, &domain_max, sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(DOMAIN_MIN, &domain_min, sizeof(double)));
}

void free_parameters(struct Parameters* parameters)
{
	for (int i = 0; i < parameters->dimension; i++)
	{
		CUDA_CALL(cudaFree(parameters->agents.position[i]));
		CUDA_CALL(cudaFree(parameters->agents.velocity[i]));
		CUDA_CALL(cudaFree(parameters->agents.local_best[i]));
	}

	CUDA_CALL(cudaFree(parameters->agents.position));
	CUDA_CALL(cudaFree(parameters->agents.velocity));
	CUDA_CALL(cudaFree(parameters->agents.error));
	CUDA_CALL(cudaFree(parameters->agents.value));
	CUDA_CALL(cudaFree(parameters->agents.local_best));

	CUDA_CALL(cudaFree(parameters->global_best_position));
	
	CUDA_CALL(cudaFree(parameters->temp_error));
	CUDA_CALL(cudaFree(parameters->temp_index));
	CUDA_CALL(cudaFree(parameters->results));
	CUDA_CALL(cudaFree(parameters->results_index));

	CUDA_CALL(cudaFree(parameters));
}

struct Parameters* create_parameters(int swarm_size, int dimension)
{
	struct Parameters* parameters;
	
	CUDA_CALL(cudaMallocManaged(&parameters, sizeof(struct Parameters)));

	parameters->swarm_size = swarm_size;
	parameters->dimension = dimension;

	parameters->global_best_value = INT_MAX; //not important
	parameters->min_error = INT_MAX; // big number

	populate_constant_memory();

	CUDA_CALL(cudaMallocManaged(&parameters->global_best_position, dimension * sizeof(double)));

	for (int i = 0; i < dimension; i++)
	{
#ifdef SUM_SQUARES_FUNCTION
		parameters->global_best_position[i] = (get_random_double() - 0.5) * ((double)UP_DOMAIN_BORDER - (double)DOWN_DOMAIN_BORDER);
#endif // DEBUG

#ifdef INTEGRATION_FUNCTION
		parameters->global_best_position[i] = get_random_double() * ((double)UP_DOMAIN_BORDER - (double)DOWN_DOMAIN_BORDER);
#endif // INTEGRATION_FUNCTION
	}

	int number_of_blocks = (int)ceil(swarm_size / (double)FIND_BEST_BLOCK_SIZE);
	int blocks_for_reduction = (int)ceil((double)number_of_blocks / 4.0);

	CUDA_CALL(cudaMallocManaged(&parameters->agents.position, dimension * sizeof(double)));
	CUDA_CALL(cudaMallocManaged(&parameters->agents.velocity, dimension * sizeof(double)));
	CUDA_CALL(cudaMallocManaged(&parameters->agents.local_best, dimension * sizeof(double)));

	CUDA_CALL(cudaMallocManaged(&parameters->agents.error, swarm_size * sizeof(double)));
	CUDA_CALL(cudaMallocManaged(&parameters->agents.value, swarm_size * sizeof(double)));

	CUDA_CALL(cudaMallocManaged(&parameters->temp_error, swarm_size * sizeof(double)));
	CUDA_CALL(cudaMallocManaged(&parameters->temp_index, swarm_size * sizeof(int)));
	CUDA_CALL(cudaMallocManaged(&parameters->results, blocks_for_reduction * sizeof(double)));
	CUDA_CALL(cudaMallocManaged(&parameters->results_index, blocks_for_reduction * sizeof(int)));

	for (int i = 0; i < dimension; i++)
	{
		CUDA_CALL(cudaMallocManaged(&parameters->agents.position[i], swarm_size * sizeof(double)));
		for (int j = 0; j < swarm_size; j++)
		{
#ifdef SUM_SQUARES_FUNCTION
			parameters->agents.position[i][j] =
				(get_random_double() - 0.5) * ((double)UP_DOMAIN_BORDER - (double)DOWN_DOMAIN_BORDER);

#endif // SUM_SQUARES_FUNCTION

#ifdef INTEGRATION_FUNCTION
			parameters->agents.position[i][j] =
				get_random_double() * ((double)UP_DOMAIN_BORDER - (double)DOWN_DOMAIN_BORDER);

#endif // INTEGRATION_FUNCTION
		}

		CUDA_CALL(cudaMallocManaged(&parameters->agents.velocity[i], swarm_size * sizeof(double)));
		for (int j = 0; j < swarm_size; j++)
		{
			parameters->agents.velocity[i][j] = 
				(get_random_double() - 0.5) * MAX_VELOCITY * (UP_DOMAIN_BORDER - DOWN_DOMAIN_BORDER);
		}

		CUDA_CALL(cudaMallocManaged(&parameters->agents.local_best[i], swarm_size * sizeof(double)));
		for (int j = 0; j < swarm_size; j++)
		{
			parameters->agents.local_best[i][j] = parameters->agents.position[i][j];
		}
	}

	for (int i = 0; i < swarm_size; i++)
	{
		parameters->agents.value[i] = DBL_MAX; //not important
		parameters->agents.error[i] = DBL_MAX; // big number

		parameters->temp_error[i] = DBL_MAX;
		parameters->temp_index[i] = i;
	}

	for (int i = 0; i < blocks_for_reduction; i++)
	{
		parameters->results_index[i] = i;
		parameters->results[i] = DBL_MAX;
	}

	return parameters;
}

struct CudaRunParameters* create_cuda_run_parameters(int swarm_size, int dimension)
{
	struct CudaRunParameters* cuda_parameters = (CudaRunParameters*)malloc(sizeof(CudaRunParameters));

	cuda_parameters->swarm_update_blocks_number = (int)ceil(swarm_size / (double)SWARM_UPDATE_BLOCK_SIZE);
	cuda_parameters->swarm_update_threads_per_block = SWARM_UPDATE_BLOCK_SIZE;
	cuda_parameters->swarm_update_shared_memory_per_block = SWARM_UPDATE_BLOCK_SIZE * dimension * sizeof(double);

	//printf("Running swarm update blocks: %d\n", cuda_parameters->swarm_update_blocks_number);
	//printf("Running swarm update threads: %d\n\n", cuda_parameters->swarm_update_threads_per_block);

	int number_of_blocks = (int)ceil(swarm_size / (double)FIND_BEST_BLOCK_SIZE);
	
	cuda_parameters->find_best_agent_blocks_number = (int)ceil((double)number_of_blocks / 4.0);
	cuda_parameters->find_best_agent_threads_per_block = FIND_BEST_BLOCK_SIZE;
	cuda_parameters->find_best_agent_shared_memory_per_block = 
		cuda_parameters->find_best_agent_threads_per_block * sizeof(double);

	//printf("Running find best agent blocks: %d\n", cuda_parameters->find_best_agent_blocks_number);
	//printf("Running find best agent threads: %d\n", cuda_parameters->find_best_agent_threads_per_block);
	//printf("Running find best agent shared memory: %d\n\n", cuda_parameters->find_best_agent_shared_memory_per_block);

	cuda_parameters->copy_best_agent_blocks_number = (int)ceil(dimension / (double)COPY_BEST_BLOCK_SIZE);
	cuda_parameters->copy_best_agent_threads_per_block = COPY_BEST_BLOCK_SIZE;

	//printf("Running copy best agent blocks: %d\n", cuda_parameters->copy_best_agent_blocks_number);
	//printf("Running copy best agent threads: %d\n\n", cuda_parameters->copy_best_agent_threads_per_block);

	return cuda_parameters;
}

__device__ double calc_velocity(struct Parameters* parameters, int agent, int k, curandStateMtgp32* state)
{
	double rand_local = curand_uniform_double(state);
	double rand_global = curand_uniform_double(state);

	double inertion = W * parameters->agents.velocity[k][agent];
	double local_effect = C1 * rand_local * (parameters->agents.local_best[k][agent] - parameters->agents.position[k][agent]);
	double swarm_effect = C2 * rand_global * (parameters->global_best_position[k] - parameters->agents.position[k][agent]);

	double velocity = inertion + local_effect + swarm_effect;

	if (velocity > MAX_AGENT_VELOCITY)
	{
		velocity = MAX_AGENT_VELOCITY;
	}
	else if (velocity < -1 * MAX_AGENT_VELOCITY)
	{
		velocity = -1 * MAX_AGENT_VELOCITY;
	}

	return velocity;
}

__device__ double calc_position(struct Parameters* parameters, int agent, int k)
{
	double new_position = parameters->agents.position[k][agent] + DT * parameters->agents.velocity[k][agent];

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

//CUDA function
__global__ void call_iteration(struct Parameters* parameters, curandStateMtgp32* states, struct OptFunctionParameters* funParams)
{	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int dimension = parameters->dimension;

	extern __shared__ double cache[];

	if (id < parameters->swarm_size)
	{
		struct Agent* agents = &parameters->agents;

		for (int k = 0; k < dimension; k++)
		{
			agents->velocity[k][id] = calc_velocity(parameters, id, k, &states[blockIdx.x]);
			agents->position[k][id] = calc_position(parameters, id, k);
		}

		agents->value[id] = opt_function(agents->position, id, funParams, cache);
		double error = calc_error(agents->value[id], funParams);

		if (error < agents->error[id])
		{
			agents->error[id] = error;
			for (int k = 0; k < dimension; k++)
			{
				agents->local_best[k][id] = agents->position[k][id];
			}
		}

		parameters->temp_error[id] = agents->error[id];
		parameters->temp_index[id] = id;
	}
}

__global__ void copy_best(struct Parameters* parameters, int agent)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	//int best_agent = parameters->results_index[0];
	int best_agent = agent;

	if (id < parameters->dimension)
	{
		parameters->global_best_position[id] = parameters->agents.local_best[id][best_agent];
	}
}

int find_best_agent_cpu(double* values, int swarm_size)
{
	double best = values[0];
	int agent = 0;
	for (int i = 1; i < swarm_size; i++)
	{
		if (values[i] < best)
		{
			best = values[i];
			agent = i;
		}
	}
	return agent;
}

__global__ void find_gpu_min_unroll_4
(double* values, int* temp_index, double* result, int* result_index, int len)
{
	extern __shared__ double shared_block[];

	int threadID = threadIdx.x;
	int id = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int total_len = len;

	int* index_block = &temp_index[blockIdx.x * blockDim.x * 4];

	double val;
	int pair;
	int index;
	int index_buffer;

	if (id < total_len)
	{
		val = values[id];
		index = temp_index[id];

		pair = id + blockDim.x;
		if ((pair < total_len) && (val > values[pair]))
		{
			val = values[pair];
			index = temp_index[pair];
		}

		pair += blockDim.x;
		if ((pair < total_len) && (val > values[pair]))
		{
			val = values[pair];
			index = temp_index[pair];
		}

		pair += blockDim.x;
		if ((pair < total_len) && (val > values[pair]))
		{
			val = values[pair];
			index = temp_index[pair];
		}

		shared_block[threadID] = val;
		temp_index[id] = index;
	}
	__syncthreads();

	if ((blockDim.x >= 256) && (threadID < 256) && (id + 256 < total_len))
	{
		pair = threadID + 256;
		val = shared_block[pair];
		if (shared_block[threadID] > val)
		{
			shared_block[threadID] = val;
			index_block[threadID] = index_block[pair];
		}
	}
	__syncthreads();


	if ((blockDim.x >= 128) && (threadID < 128) && (id + 128 < total_len))
	{
		pair = threadID + 128;
		val = shared_block[pair];
		if (shared_block[threadID] > val)
		{
			shared_block[threadID] = val;
			index_block[threadID] = index_block[pair];
		}
	}
	__syncthreads();

	if ((blockDim.x >= 64) && (threadID < 64) && (id + 64 < total_len))
	{
		pair = threadID + 64;
		val = shared_block[pair];
		if (shared_block[threadID] > val)
		{
			shared_block[threadID] = val;
			index_block[threadID] = index_block[pair];
		}
	}
	__syncthreads();

	if (id < total_len)
	{
		index = index_block[threadID];
	}

	if (threadID < 32)
	{
		volatile double* start = shared_block;

		pair = threadID + 32;
		if ((id + 32 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_block[pair];
		}

		pair = threadID + 16;
		index_buffer = __shfl_down_sync(FULL_MASK, index, 16);
		if ((id + 16 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_buffer;
		}

		pair = threadID + 8;
		index_buffer = __shfl_down_sync(FULL_MASK, index, 8);
		if ((id + 8 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_buffer;
		}

		pair = threadID + 4;
		index_buffer = __shfl_down_sync(FULL_MASK, index, 4);
		if ((id + 4 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_buffer;
		}

		pair = threadID + 2;
		index_buffer = __shfl_down_sync(FULL_MASK, index, 2);
		if ((id + 2 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_buffer;
		}

		pair = threadID + 1;
		index_buffer = __shfl_down_sync(FULL_MASK, index, 1);
		if ((id + 1 < total_len) && (start[threadID] > start[pair]))
		{
			start[threadID] = start[pair];
			index = index_buffer;
		}
	}

	if (!threadID)
	{
		result[blockIdx.x] = shared_block[0];
		result_index[blockIdx.x] = index;
	}
}

int exec_pso(int swarm_size, int dimension, int turns_num, int log_density,
	struct OptFunctionParameters* host_opt_fun_params, double* result, struct PerformanceData* performance_data)
{
	clock_t pso_begin_time = clock();

	//-----------------------------------------------CHECK INPUT PARAMETERS--------------------------------------------
	int max_swarm_size = MAX_NUMBER_OF_BLOCKS * SWARM_UPDATE_BLOCK_SIZE;

	if ((swarm_size <= 0) || (swarm_size >= max_swarm_size))
	{
		fprintf(stderr, "Error! Swarm size is out of range!\n");
		return PSO_ERROR;
	}
	
	if ((turns_num <= 0) || (log_density <= 0) || (dimension <= 0))
	{
		fprintf(stderr, "Error! Number of dimension, iterations, log_density must be > 0!\n");
		return PSO_ERROR;
	}
	
	if (!host_opt_fun_params || !result)
	{
		fprintf(stderr, "Error! Nullptr passed as one of the parameters!\n");
		return PSO_ERROR;
	}
	

	//-----------------------------------------------INITIALIZE CUDA DEVICE--------------------------------------------
	int deviceCount = 0;
	CUDA_CALL(cudaGetDeviceCount(&deviceCount));
	if (deviceCount <= 0)
	{
		return PSO_ERROR;
	}
	CUDA_CALL(cudaSetDevice(0));

	//-----------------------------------------------INITIALIZE STRUCTURES VARIABLES-----------------------------------
	struct Parameters* parameters = create_parameters(swarm_size, dimension);
	if (!parameters)
	{
		return PSO_ERROR;
	}
	
	struct CudaRunParameters* cuda_launch = create_cuda_run_parameters(swarm_size, dimension);
	if (!cuda_launch)
	{
		return PSO_ERROR;
	}

	OptFunctionParameters* dev_opt_fun_params;
	CUDA_CALL(cudaMalloc(&dev_opt_fun_params, sizeof(OptFunctionParameters)));	
	CUDA_CALL(cudaMemcpy(dev_opt_fun_params, host_opt_fun_params, sizeof(OptFunctionParameters), cudaMemcpyHostToDevice));
	
	//-----------------------------------------------PREPARE CURAND----------------------------------------------------
	mtgp32_kernel_params* dev_kernel_params;
	curandStateMtgp32* states;

	CUDA_CALL(cudaMalloc(&states, sizeof(curandStateMtgp32) * cuda_launch->swarm_update_blocks_number));
	CUDA_CALL(cudaMalloc(&dev_kernel_params, sizeof(mtgp32_kernel_params)));

	//prepare and copy device Mersenne Twister random numbers generator parameters
	curandStatus_t curand_error = curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, dev_kernel_params);
	if (curand_error != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		fprintf(stderr, "Error preparing parameters for device Mersenne Twister generator!\n");
		return PSO_ERROR;
	}

	//initialize generator
	curand_error = curandMakeMTGP32KernelState
	(states, mtgp32dc_params_fast_11213, dev_kernel_params, cuda_launch->swarm_update_blocks_number, CURAND_SEED);
	if (curand_error != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		fprintf(stderr, "Error initializing device Mersenne Twister generator!\n");
		return PSO_ERROR;
	}
	//-----------------------------------------------EXECUTE PSO-------------------------------------------------------

	int is_new_best;
	int best_value_index;
	int best_agent;

	clock_t pso_begin_iterations = clock();

	for (int i = 0; i < turns_num; i++)
	{
		clock_t pso_begin_current_iteration = clock();
		
		//Calculate optimization function and update swarm
		call_iteration << < cuda_launch->swarm_update_blocks_number, 
							cuda_launch->swarm_update_threads_per_block,
							cuda_launch->swarm_update_shared_memory_per_block >> >
			(parameters, states, dev_opt_fun_params);

		CUDA_CALL(cudaGetLastError());		
		CUDA_CALL(cudaDeviceSynchronize());

		clock_t pso_current_iteration_sync_done = clock();
		
		//--------------------------------find best----------------------------
		is_new_best = 0;

		find_gpu_min_unroll_4 << <	cuda_launch->find_best_agent_blocks_number, 
									cuda_launch->find_best_agent_threads_per_block, 
									cuda_launch->find_best_agent_shared_memory_per_block >>>
			(parameters->temp_error, parameters->temp_index, parameters->results, parameters->results_index, swarm_size);

		CUDA_CALL(cudaPeekAtLastError());
		CUDA_CALL(cudaDeviceSynchronize());

		best_value_index = find_best_agent_cpu(parameters->results, cuda_launch->find_best_agent_blocks_number);
		best_agent = parameters->results_index[best_value_index];

		if (parameters->agents.error[best_agent] < parameters->min_error)
		{
			parameters->min_error = parameters->agents.error[best_agent];
			parameters->global_best_value = parameters->agents.value[best_agent];
			
			copy_best <<<   cuda_launch->copy_best_agent_blocks_number, 
							cuda_launch->copy_best_agent_threads_per_block >>>
				(parameters, best_agent);
			
			CUDA_CALL(cudaGetLastError());
			CUDA_CALL(cudaDeviceSynchronize());
			
			is_new_best = 1;
		}

		clock_t pso_current_iteration_best_done = clock();

		log_results(parameters->global_best_value, i, log_density);
		log_positions(parameters->global_best_position, dimension, i);

		clock_t pso_end_current_iteration = clock();

		calc_turn_times(performance_data, i, is_new_best, pso_begin_current_iteration,
			pso_current_iteration_sync_done, pso_current_iteration_best_done, pso_end_current_iteration);
	}

	clock_t pso_end_iterations = clock();

	*result = parameters->global_best_value;

	for (int i = 0; i < dimension; i++)
	{
		result[i + 1] = parameters->global_best_position[i];
	}

	//-----------------------------------------------RELEASE MEMORY----------------------------------------------------
	
	free_parameters(parameters);

	CUDA_CALL(cudaFree(dev_opt_fun_params));
	
	CUDA_CALL(cudaFree(states));
	CUDA_CALL(cudaFree(dev_kernel_params));

	CUDA_CALL(cudaDeviceReset());
	
	free(cuda_launch);

	clock_t pso_end_time = clock();

	calc_algorithm_time(performance_data, turns_num, pso_begin_time, pso_begin_iterations, pso_end_iterations, pso_end_time);

	return PSO_OK;
}

#define MIN_NUMBER_OF_ARGUMENTS_SUM_SQUARES 6
#define MIN_NUMBER_OF_ARGUMENTS_INTEGRATION 7
#define MIN_NUMBER_OF_ARGUMENTS_SIMULATION 5

//args - sum squares: (executable name), sworm size, dimension, number of iterations, log_density, file for result
//args - integration: (executable name), sworm size, mmax, Nint, number of iterations, log_density, file for result
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

	int turns_number = atoi(args[3]);
	int log_density = atoi(args[4]);

	struct OptFunctionParameters* opt_params = (struct OptFunctionParameters*) malloc(sizeof(struct OptFunctionParameters));
	opt_params->dimension = dimension;
	opt_params->swarm_size = swarm_size;

	char* filename = args[5];
#endif // SUM_SQUARES_FUNCTION

#ifdef INTEGRATION_FUNCTION
	if (argc < MIN_NUMBER_OF_ARGUMENTS_INTEGRATION)
	{
		printf("Error, not enough parameters!");
		return -1;
	}

	int swarm_size = atoi(args[1]);
	
	int mmax = atoi(args[2]);
	int n = atoi(args[3]);
	
	int turns_number = atoi(args[4]);
	int log_density = atoi(args[5]);

	struct OptFunctionParameters* opt_params = (struct OptFunctionParameters*) malloc(sizeof(struct OptFunctionParameters));
	opt_params->Mmax = mmax;
	opt_params->N = n;

	populate_command_arguments_to_cuda(mmax, n);

	int dimension = n * 2;

	char* filename = args[6];
#endif // INTEGRATION_FUNCTION

	double* result = (double*)malloc((dimension + 1) * sizeof(double));
	struct PerformanceData* performance_data = create_performance_struct(turns_number);

	if (!opt_params || !performance_data)
	{
		printf("Error, failed allocation for performance struct or parameters for optimization function!\n");
		return -1;
	}

	setup_log_file(filename);

	printf("The right answer is: 0\n\n");

	report_algorithm_parameters(swarm_size, dimension, turns_number, log_density);

	int error = 0;
	
	error = exec_pso(swarm_size, dimension, turns_number, log_density, opt_params, result, performance_data);
	
	if (error)
	{
		printf("Error running pso! Terminating the program!\n");
	}
	else if (result)
	{
		generate_final_report(result, swarm_size, dimension, turns_number, log_density, performance_data);
	}
	
	close_log_file();

	free_performance_struct(performance_data);
	free(opt_params);
	free(result);
	
	return error;
}
