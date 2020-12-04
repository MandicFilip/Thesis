#ifndef PSO_HEADER

#define PSO_HEADER

struct Parameters;

int exec_pso(unsigned swarm_size, unsigned dimension, unsigned turns_num, unsigned log_density,
	struct OptFunctionParameters* opt_fun_params, double* result, struct PerformanceData* performance_data);

#endif // !PSO_HEADER
