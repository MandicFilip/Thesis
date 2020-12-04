#include<stdlib.h>
#include<stdio.h>
#include<float.h>
#include<time.h>
#include<string.h>

#include"util.h"
#include"pso.h"
#include<random>

#define _CRT_SECURE_NO_DEPRECATE 
#define _CRT_SECURE_NO_WARNINGS

#define OPEN_MP_NUMBER_OF_THREADS 12

static FILE* log_file = NULL;

static std::mt19937 generators[OPEN_MP_NUMBER_OF_THREADS] = {
	std::mt19937(time(NULL)),
	std::mt19937(time(NULL) + 1),
	std::mt19937(time(NULL) + 2),
	std::mt19937(time(NULL) + 3),
	std::mt19937(time(NULL) + 4),
	std::mt19937(time(NULL) + 5),
	std::mt19937(time(NULL) + 6),
	std::mt19937(time(NULL) + 7),
	std::mt19937(time(NULL) + 8),
	std::mt19937(time(NULL) + 9),
	std::mt19937(time(NULL) + 10),
	std::mt19937(time(NULL) + 11),
	};

static std::uniform_real_distribution<double> dis[OPEN_MP_NUMBER_OF_THREADS] = {
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	std::uniform_real_distribution<double>(0.0, 1.0),
	};

//generates random [0..1)
double get_random_double(int id_thread)
{
	return dis[id_thread](generators[id_thread]);
	//return rand() / (double)RAND_MAX;
}

void setup_log_file(char* filename)
{
	log_file = fopen(filename, "w");
	if (!log_file)
	{
		printf("Error file used for log info, with name %s can't be open!", filename);
		return;
	}
}

void close_log_file()
{
	if (!log_file)
	{
		return;
	}

	fclose(log_file);
	log_file = NULL;
}

void log_results(double best_value, int turn, int log_density)
{
	if ((turn + 1) % log_density == 0)
	{
		printf("Swarm update: # %d, Best value: %.15lf\n", turn + 1, best_value);

		fprintf(log_file, "Swarm update: #%d, Best value: %.15lf\n", turn + 1, best_value);
		fflush(log_file);
	}
}

void print_best_positions(double* positions, int dimension)
{
	printf("Positions: ");
	for (int i = 0; i < dimension; i++)
	{
		printf("%lf ", positions[i]);
	}
	printf("\n\n");

	fprintf(log_file, "Positions: ");
	for (int i = 0; i < dimension; i++)
	{
		fprintf(log_file, "%lf ", positions[i]);
	}
	fprintf(log_file, "\n\n");
}

void log_positions(double* positions, int dimension, int turn)
{
	if ((turn + 1) % 10000 == 0)
	{
		print_best_positions(positions, dimension);
	}
}

//------------------------------------------------TIME DATA------------------------------------------------------------

struct PerformanceData
{
	double total_time;
	double total_turns_time;
	double set_up_time;
	double clean_up_time;

	double average_turn_time;
	double average_turn_time_without_logs;
	double average_device_run_time;
	double average_find_best_agent_time;

	double time_to_reach_optimum;
	int turns_to_reach_optimum;

	double* turn_times;
	double* device_run_times;
	double* find_best_agent_times;
	double* log_times;
};

struct PerformanceData* create_performance_struct(int turns_number)
{
	struct PerformanceData* data = (struct PerformanceData*) malloc(sizeof(struct PerformanceData));

	if (!data)
	{
		return NULL;
	}

	if (!(data->turn_times = (double*)malloc(turns_number * sizeof(double))))
	{
		return NULL;
	}

	if (!(data->device_run_times = (double*)malloc(turns_number * sizeof(double))))
	{
		return NULL;
	}

	if (!(data->find_best_agent_times = (double*)malloc(turns_number * sizeof(double))))
	{
		return NULL;
	}

	if (!(data->log_times = (double*)malloc(turns_number * sizeof(double))))
	{
		return NULL;
	}

	data->turns_to_reach_optimum = 0;

	return data;
}

void free_performance_struct(struct PerformanceData* data)
{
	free(data->turn_times);
	free(data->device_run_times);
	free(data->find_best_agent_times);
	free(data->log_times);

	free(data);
}

void calc_turn_times(struct PerformanceData* performance_data, int turn, int is_new_best,
	clock_t begin, clock_t sync, clock_t find_best, clock_t end)
{
	performance_data->turn_times[turn] = ((double)end - begin) / CLOCKS_PER_SEC * 1000;
	performance_data->device_run_times[turn] = ((double)sync - begin) / CLOCKS_PER_SEC * 1000;
	performance_data->find_best_agent_times[turn] = ((double)find_best - sync) / CLOCKS_PER_SEC * 1000;
	performance_data->log_times[turn] = ((double)end - find_best) / CLOCKS_PER_SEC * 1000;

	if (is_new_best)
	{
		performance_data->turns_to_reach_optimum = turn + 1;
	}
}

void calc_algorithm_time(struct PerformanceData* performance_data, int turns_num,
	clock_t pso_begin, clock_t turns_begin, clock_t turns_end, clock_t pso_end)
{
	performance_data->total_time = ((double)pso_end - pso_begin) / CLOCKS_PER_SEC * 1000;
	performance_data->set_up_time = ((double)turns_begin - pso_begin) / CLOCKS_PER_SEC * 1000;
	performance_data->total_turns_time = ((double)turns_end - turns_begin) / CLOCKS_PER_SEC * 1000;
	performance_data->clean_up_time = ((double)pso_end - turns_end) / CLOCKS_PER_SEC * 1000;

	performance_data->average_turn_time = performance_data->total_turns_time / turns_num;

	double total_device_run_time = 0;
	double total_find_best_agent_time = 0;
	double time_to_optimum = 0;

	for (int i = 0; i < turns_num; i++)
	{
		total_device_run_time += performance_data->device_run_times[i];
		total_find_best_agent_time += performance_data->find_best_agent_times[i];

		if (i < performance_data->turns_to_reach_optimum)
		{
			time_to_optimum += performance_data->turn_times[i];
		}
	}

	performance_data->average_device_run_time = total_device_run_time / turns_num;
	performance_data->average_find_best_agent_time = total_find_best_agent_time / turns_num;

	performance_data->average_turn_time_without_logs =
		performance_data->average_device_run_time + performance_data->average_find_best_agent_time;

	performance_data->time_to_reach_optimum = time_to_optimum;
}

void report_time(struct PerformanceData* performance_data)
{
	//to stdout
	printf("Time for complete algorithm execution: %.3lf ms\n", performance_data->total_time);
	printf("Time for algorithm set up: %.3lf ms\n", performance_data->set_up_time);
	printf("Time for algorithm clean up: %.3lf ms\n", performance_data->clean_up_time);
	printf("Time for all algorithm turns: %.3lf ms\n\n", performance_data->total_turns_time);

	printf("Average algorithm turn time: %.3lf ms\n", performance_data->average_turn_time);
	printf("Average time for algorithm turn without logs: %.3lf ms\n", performance_data->average_turn_time_without_logs);
	printf("Average time for swarm update (device execution): %.3lf ms\n", performance_data->average_device_run_time);
	printf("Average time for determining the global best position: %.3lf ms\n\n", performance_data->average_find_best_agent_time);

	printf("Time needed to reach reported value: %.3lf ms\n", performance_data->time_to_reach_optimum);
	printf("Algorithm turns needed to reach reported value: %d\n\n", performance_data->turns_to_reach_optimum);

	//to file
	fprintf(log_file, "Time for complete algorithm execution: %.3lf ms\n", performance_data->total_time);
	fprintf(log_file, "Time for algorithm set up: %.3lf ms\n", performance_data->set_up_time);
	fprintf(log_file, "Time for algorithm clean up: %.3lf ms\n", performance_data->clean_up_time);
	fprintf(log_file, "Time for all algorithm turns: %.3lf ms\n\n", performance_data->total_turns_time);

	fprintf(log_file, "Average algorithm turn time: %.3lf ms\n", performance_data->average_turn_time);
	fprintf(log_file, "Average time for algorithm turn without logs: %.3lf ms\n", performance_data->average_turn_time_without_logs);
	fprintf(log_file, "Average time for swarm update (device execution): %.3lf ms\n", performance_data->average_device_run_time);
	fprintf(log_file, "Average time for determining the global best position: %.3lf ms\n\n", performance_data->average_find_best_agent_time);

	fprintf(log_file, "Time needed to reach reported value: %.3lf ms\n", performance_data->time_to_reach_optimum);
	fprintf(log_file, "Algorithm turns needed to reach reported value: %d\n\n", performance_data->turns_to_reach_optimum);
}

void report_algorithm_parameters(int swarm_size, int dimension, int turns_num, int log_density)
{
	printf("Swarm size: %d\n", swarm_size);
	printf("Dimension: %d\n", dimension);
	printf("Number of algorithm turns: %d\n", turns_num);
	printf("Log density: %d\n\n", log_density);

	fprintf(log_file, "Swarm size: %d\n", swarm_size);
	fprintf(log_file, "Dimension: %d\n", dimension);
	fprintf(log_file, "Number of algorithm turns: %d\n", turns_num);
	fprintf(log_file, "Log density: %d\n\n", log_density);
}

void generate_final_report(double* result, int swarm_size, int dimension, int turns_num, int log_density, struct PerformanceData* performance_data)
{
	printf("Final result of optimization: %.15lf\n\n", result[0]);
	fprintf(log_file, "Final result of optimization: %.15lf\n\n", result[0]);

	print_best_positions(&result[1], dimension);

	report_algorithm_parameters(swarm_size, dimension, turns_num, log_density);

	report_time(performance_data);
}


PerformanceData** performance_data_array;

void init_performance_array(int number_of_simulations)
{
	performance_data_array = (PerformanceData**)malloc(number_of_simulations * sizeof(PerformanceData*));
}

void free_performance_structs(int number_of_simulations)
{
	for (int i = 0; i < number_of_simulations; i++)
	{
		free(performance_data_array[i]);
	}
	free(performance_data_array);
}

void add_performance_struct(PerformanceData* performance_data, int simulation_number)
{
	performance_data_array[simulation_number] = performance_data;
}

/*
void generate_performance_simulation_report(char* filename, int M, int swarm_sizes, int number_of_runs)
{
	setup_log_file(filename);

	for (int i = 0; i < swarm_sizes; i++)
	{
		for (int j = 0; j < M; j++)
		{
			double turn_time_without_logs = 0;
			double swarm_update_time = 0;
			double find_best_times = 0;

			int performance_index = i * M * number_of_runs + j * number_of_runs;
			for (int k = 0; k < number_of_runs; k++)
			{
				turn_time_without_logs += performance_data_array[performance_index]->average_turn_time_without_logs;
				swarm_update_time += performance_data_array[performance_index]->average_device_run_time;
				find_best_times += performance_data_array[performance_index]->average_find_best_agent_time;

				performance_index++;
			}
			turn_time_without_logs /= number_of_runs;
			swarm_update_time /= number_of_runs;
			find_best_times /= number_of_runs;

			fprintf(log_file, "%lf,%lf,%lf   ", turn_time_without_logs, swarm_update_time, find_best_times);
		}
		fprintf(log_file, "\n");
	}

	close_log_file();
}
*/