#ifndef UTIL_HEADER

#define UTIL_HEADER

#include<time.h>

double get_random_double(int id_thread);


void setup_log_file(char* filename);

void close_log_file();

void log_results(double best_value, int turn, int log_density);

void log_positions(double* positions, int dimension, int turn);


struct PerformanceData;

void calc_turn_times(struct PerformanceData* performance_data, int turn, int is_new_best,
	clock_t begin, clock_t sync, clock_t find_best, clock_t end);

void calc_algorithm_time(struct PerformanceData* performance_data, int turns_num,
	clock_t pso_begin, clock_t turns_begin, clock_t turns_end, clock_t pso_end);

void report_time(struct PerformanceData* performance_data);

void report_algorithm_parameters(int swarm_size, int dimension, int turns_num, int log_density);

void generate_final_report(double* result, int swarm_size, int dimension, int turns_num, int log_density, struct PerformanceData* performance_data);

struct PerformanceData* create_performance_struct(int turns_number);

void free_performance_struct(struct PerformanceData* data);


void init_performance_array(int number_of_simulations);

void free_performance_structs(int number_of_simulations);

void add_performance_struct(PerformanceData* performance_data, int simulation_number);

void generate_performance_simulation_report(char* filename, int M, int swarm_sizes, int number_of_runs);

#endif
