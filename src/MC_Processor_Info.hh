#ifndef MC_PROCESSOR_INFO_HH
#define MC_PROCESSOR_INFO_HH

#include "utilsMpi.hh"
#include "macros.hh"

class MC_Processor_Info
{
public:

    int rank;
    int num_processors;
    int shm_rank;
    int shm_num_processors;
    int use_gpu;
    int gpu_id;

    double shm_perf_factor;

    MPI_Comm comm_mc_world;
    MPI_Comm comm_mc_shmcomm;

    MC_Processor_Info() : comm_mc_world(MPI_COMM_WORLD)
    {
      mpiComm_rank(comm_mc_world, &rank);
      mpiComm_size(comm_mc_world, &num_processors);

      MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL, &comm_mc_shmcomm);

      mpiComm_rank(comm_mc_shmcomm, &shm_rank);
      mpiComm_size(comm_mc_shmcomm, &shm_num_processors);

      use_gpu = 0;
      gpu_id = 0;
      shm_perf_factor = 1;
    }

    void SetFactor(double ratio)
    {
        shm_perf_factor = ratio*shm_num_processors;
    }

};


#endif
