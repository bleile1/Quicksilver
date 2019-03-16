#include <iostream>
#include "utils.hh"
#include "Parameters.hh"
#include "utilsMpi.hh"
#include "MonteCarlo.hh"
#include "initMC.hh"
#include "Tallies.hh"
#include "PopulationControl.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Particle_Buffer.hh"
#include "MC_Processor_Info.hh"
#include "MC_Time_Info.hh"
#include "macros.hh"
#include "MC_Fast_Timer.hh"
#include "MC_SourceNow.hh"
#include "SendQueue.hh"
#include "NVTX_Range.hh"
#include "cudaUtils.hh"
#include "cudaFunctions.hh"
#include "qs_assert.hh"
#include "CycleTracking.hh"
#include "CoralBenchmark.hh"
#include "EnergySpectrum.hh"

#include "git_hash.hh"
#include "git_vers.hh"

void gameOver();
void cycleInit( bool loadBalance );
void cycleTracking(MonteCarlo* monteCarlo);
void cycleFinalize();

using namespace std;

MonteCarlo *mcco  = NULL;

int main(int argc, char** argv)
{
   mpiInit(&argc, &argv);
   printBanner(GIT_VERS, GIT_HASH);

   Parameters params = getParameters(argc, argv);
   printParameters(params, cout);

   // mcco stores just about everything. 
   mcco = initMC(params); 

   int loadBalance = params.simulationParams.loadBalance;

   MC_FASTTIMER_START(MC_Fast_Timer::main);     // this can be done once mcco exist.

   const int nSteps = params.simulationParams.nSteps;

   for (int ii=0; ii<nSteps; ++ii)
   {
      cycleInit( bool(loadBalance) );
      cycleTracking(mcco);
      cycleFinalize();

      mcco->fast_timer->Last_Cycle_Report(
            params.simulationParams.cycleTimers,
            mcco->processor_info->rank,
            mcco->processor_info->num_processors,
            mcco->processor_info->comm_mc_world );
   }


   MC_FASTTIMER_STOP(MC_Fast_Timer::main);

   gameOver();

   coralBenchmarkCorrectness(mcco, params);

#ifdef HAVE_UVM
    mcco->~MonteCarlo();
    cudaFree( mcco );
#else
   delete mcco;
#endif

   mpiFinalize();
   
   return 0;
}

void gameOver()
{
    mcco->fast_timer->Cumulative_Report(mcco->processor_info->rank,
                                        mcco->processor_info-> num_processors,
                                        mcco->processor_info->comm_mc_world,
                                        mcco->_tallies->_balanceCumulative._numSegments);
    mcco->_tallies->_spectrum.PrintSpectrum(mcco);
}

void cycleInit( bool loadBalance )
{

    MC_FASTTIMER_START(MC_Fast_Timer::cycleInit);

    mcco->clearCrossSectionCache();

    mcco->_tallies->CycleInitialize(mcco);

    mcco->_particleVaultContainer->swapProcessingProcessedVaults();

    mcco->_particleVaultContainer->collapseProcessed();
    mcco->_particleVaultContainer->collapseProcessing();

    mcco->_tallies->_balanceTask[0]._start = mcco->_particleVaultContainer->sizeProcessing();

    mcco->particle_buffer->Initialize();

    MC_SourceNow(mcco);
   
    PopulationControl(mcco, loadBalance); // controls particle population

    RouletteLowWeightParticles(mcco); // Delete particles with low statistical weight

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleInit);
}


#if defined (HAVE_CUDA)

__global__ void CycleTrackingKernel( MonteCarlo* monteCarlo, int num_particles, ParticleVault* processingVault, ParticleVault* processedVault, const unsigned int stream )
{
   int global_index = getGlobalThreadID(); 

    if( global_index < num_particles )
    {
        CycleTrackingGuts( monteCarlo, global_index, processingVault, processedVault, stream );
    }
}

#endif

inline unsigned int inverseStream( unsigned int stream )
{
    return (stream == 0) 1 : 0;
}

void cycleTracking(MonteCarlo *monteCarlo)
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking);

    bool done = false;

    //Determine whether or not to use GPUs if they are available (set for each MPI rank)
    ExecutionPolicy execPolicy = getExecutionPolicy( monteCarlo->processor_info->use_gpu );

    ParticleVaultContainer &my_particle_vault = *(monteCarlo->_particleVaultContainer);

    //Post Inital Receives for Particle Buffer
    monteCarlo->particle_buffer->Post_Receive_Particle_Buffer( my_particle_vault.getVaultSize() );

    //Get Test For Done Method (Blocking or non-blocking
    MC_New_Test_Done_Method::Enum new_test_done_method = monteCarlo->particle_buffer->new_test_done_method;

    //Setting up Cuda Streams for Asynchronous Kernel Dispatching (2 Steam Mode) CUDA ONLY
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    unsigned int STREAM = 0;

    do
    {
        while ( !done )
        {
            uint64_t fill_vault = 0;

	        uint64_t num_vaults = my_particle_vault.size();
	        if( num_vaults < 2)
	        {
                fprintf(stderr, "This Code is only set to run with 2 or more vaults: num_vaults = %lu\n", num_vaults);    
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            ParticleVault* processingVaults[num_streams];
            ParticleVault* processedVaults[num_streams];
            uint64_t processing_vault = 0;
            uint64_t processing_vaults[2];
            uint64_t processed_vault = my_particle_vault.getFirstEmptyProcessedVault(num_vaults);

            //Initialize First vaults for 2 stream algorithm
            processingVaults[0] = my_particle_vault.getTaskProcessingVault(processing_vault); 
            processing_vaults[0]= processing_vault++;
            processingVaults[1] = my_particle_vault.getTaskProcessingVault(processing_vault++);
            processing_vaults[1]= processing_vault++;
            processedVaults[0]  = my_particle_vault.getTaskProcessingVault(processed_vault+processing_vaults[0]);
            processedVaults[1]  = my_particle_vault.getTaskProcessingVault(processed_vault+processing_vaults[1]);

            //Perform Initial Kernel Launches for 2 stream algorithm
            for( int stream = 0; stream < num_streams; stream++ )
            {
                int numParticles = processingVaults[stream]->size();
                if ( numParticles != 0 )
                {
                    dim3 grid(1,1,1);
                    dim3 block(1,1,1);
                    int runKernel = ThreadBlockLayout( grid, block, numParticles);

                    //Call Cycle Tracking Kernel
                    if( runKernel )
                        CycleTrackingKernel<<<grid, block, streams[stream] >>>( monteCarlo, numParticles, 
                                processingVault[stream], processedVault[stream], stream);
                }
            }

            //Untill Both Streams are at or past num_vaults
            int done_count = 0;
            do
            {
                //Wait for execution on this stream to complete
                cudaStreamSynchronize(STREAM);

                //Do MPI for particles in this stream
                SendQueue &sendQueue = *(my_particle_vault.getSendQueue(STREAM));
                monteCarlo->particle_buffer->Allocate_Send_Buffer( sendQueue );

                //Move particles from send queue to the send buffers
                for ( int index = 0; index < sendQueue.size(); index++ )
                {
                    sendQueueTuple& sendQueueT = sendQueue.getTuple( index );
                    MC_Base_Particle mcb_particle;

                    processingVault[STREAM]->getBaseParticleComm( mcb_particle, sendQueueT._particleIndex );

                    int buffer = monteCarlo->particle_buffer->Choose_Buffer(sendQueueT._neighbor );
                    monteCarlo->particle_buffer->Buffer_Particle(mcb_particle, buffer );
                }

                monteCarlo->particle_buffer->Send_Particle_Buffers(); // post MPI sends

                //reset size to 0 all data can now be safely overwritten
                processingVault[STREAM]->clear(); 
                sendQueue.clear();

                // Move particles in "extra" vaults into the regular vaults. 
                //  Ignoring the other streams active vault
                my_particle_vault.cleanExtraVaults(STREAM, processing_vaults[inverseStream( STREAM)] );

                // receive any particles that have arrived from other ranks
                monteCarlo->particle_buffer->Receive_Particle_Buffers( fill_vault );

                if( processing_vault < num_vaults )
                {
                    processingVaults[STREAM] = my_particle_vault.getTaskProcessingVault(processing_vault); 
                    processing_vaults[STREAM]= processing_vault++;
                    processedVaults[STREAM]  = my_particle_vault.getTaskProcessingVault(processed_vault+processing_vaults[STREAM]);

                    numParticles = processingVaults[STREAM]->size();
                    if ( numParticles != 0 )
	                {
                        dim3 grid(1,1,1);
                        dim3 block(1,1,1);
                        int runKernel = ThreadBlockLayout( grid, block, numParticles);
                    
                        //Call Cycle Tracking Kernel
                        if( runKernel )
                           CycleTrackingKernel<<<grid, block, streams[STREAM] >>>( monteCarlo, numParticles, 
                                   processingVault[STREAM], processedVault[STREAM], STREAM );

                    }
                }
                else
                {
                    done_count++;
                }
                //Swap Streams to work on
                STREAM = inverseStream(STREAM);

            } while( done_count >= num_streams );

            my_particle_vault.collapseProcessing();
            my_particle_vault.collapseProcessed();


            //Test for done - blocking on all MPI ranks
            done = monteCarlo->particle_buffer->Test_Done_New( new_test_done_method );

        } // while not done: Test_Done_New()

        // Everything should be done normally.
        done = monteCarlo->particle_buffer->Test_Done_New( MC_New_Test_Done_Method::Blocking );

    } while ( !done );

    //Make sure to cancel all pending receive requests
    monteCarlo->particle_buffer->Cancel_Receive_Buffer_Requests();
    //Make sure Buffers Memory is Free
    monteCarlo->particle_buffer->Free_Buffers();

   MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking);
}


void cycleFinalize()
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleFinalize);

    mcco->_tallies->_balanceTask[0]._end = mcco->_particleVaultContainer->sizeProcessed();

    // Update the cumulative tally data.
    mcco->_tallies->CycleFinalize(mcco); 

    mcco->time_info->cycle++;

    mcco->particle_buffer->Free_Memory();

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleFinalize);
}

