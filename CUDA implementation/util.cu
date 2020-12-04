#include<stdlib.h>
#include<stdio.h>
#include<float.h>
#include<time.h>
#include<random>

#include"util.h"

#define _CRT_SECURE_NO_DEPRECATE 
#define _CRT_SECURE_NO_WARNINGS

std::mt19937 generator(time(NULL));
std::uniform_real_distribution<double> dis(0.0, 1.0);

//generates random [0..1)
double get_random_double()
{
	return dis(generator);
}


static FILE* log_file = NULL;

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

void print_best_positions(double* positions, int dimension)
{
	printf("Positions: ");
	for (int i = 0; i < dimension; i++)
	{
		printf("%.15lf ", positions[i]);
	}
	printf("\n\n");

	fprintf(log_file, "Positions: ");
	for (int i = 0; i < dimension; i++)
	{
		fprintf(log_file, "%.15lf ", positions[i]);
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

void log_results(double best_value, int turn, int log_density)
{
	if ((turn + 1) % log_density == 0)
	{
		printf("Swarm update: # %d, Best value: %.15lf\n", turn + 1, best_value);

		fprintf(log_file, "Swarm update: #%d, Best value: %.15lf\n", turn + 1, best_value);
		fflush(log_file);
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

/*
void generate_simulation_report
(double* avg_values, double* all_values, int number_of_swarm_sizes, int number_of_runs, int Mmax)
{
	fprintf(log_file, "%d %d %d\n\n", number_of_swarm_sizes, number_of_runs, Mmax);

	for (int i = 0; i < number_of_swarm_sizes; i++)
	{
		fprintf(log_file, "%.15lf\n", avg_values[i]);
	}
	fprintf(log_file, "\n\n");

	for (int i = 0; i < number_of_swarm_sizes; i++)
	{
		for (int j = 0; j < number_of_runs; j++)
		{
			fprintf(log_file, "%.15lf\n", all_values[i * number_of_swarm_sizes + j]);
		}
		fprintf(log_file, "\n\n");
	}
}
*/
