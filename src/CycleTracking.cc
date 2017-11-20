#include "CycleTracking.hh"
#include "MonteCarlo.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Segment_Outcome.hh"
#include "CollisionEvent.hh"
#include "MC_Facet_Crossing_Event.hh"
#include "MCT.hh"
#include "DeclareMacro.hh"
#include "AtomicMacro.hh"
#include "macros.hh"
#include "qs_assert.hh"
#include "cudaUtils.hh"
#include "cudaFunctions.hh"

#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

HOST_DEVICE
void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault, int num_particles, int* sIndex, MC_Segment_Outcome_type::Enum* sEvent, MC_Particle* sParts );
HOST_DEVICE_END

HOST_DEVICE
void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault, int num_particles, int* sIndex, MC_Segment_Outcome_type::Enum* sEvent, MC_Particle* sParts);
HOST_DEVICE_END

HOST_DEVICE
void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault, int num_particles, int* sIndex, MC_Segment_Outcome_type::Enum* sEvent, MC_Particle* sParts )
{
    MC_Particle mc_particle;

    // Copy a single particle from the particle vault into mc_particle
    MC_Load_Particle(monteCarlo, mc_particle, processingVault, particle_index);

    // loop over this particle until we cannot do anything more with it on this processor
    CycleTrackingFunction( monteCarlo, mc_particle, particle_index, processingVault, processedVault, num_particles, sIndex, sEvent, sParts );
}
HOST_DEVICE_END

HOST_DEVICE
void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault, int num_particles, int* sIndex, MC_Segment_Outcome_type::Enum* sEvent, MC_Particle* sParts)
{
    bool keepTrackingThisParticle = false;
    unsigned int tally_index =      (particle_index) % monteCarlo->_tallies->GetNumBalanceReplications();
    unsigned int flux_tally_index = (particle_index) % monteCarlo->_tallies->GetNumFluxReplications();
    unsigned int cell_tally_index = (particle_index) % monteCarlo->_tallies->GetNumCellTallyReplications();
#ifdef DO_COLLISION_SORT
#ifdef __CUDA_ARCH__
    sIndex[threadIdx.x] = particle_index;
#endif
#endif
    do
    {
        // Determine the outcome of a particle at the end of this segment such as:
        //
        //   (0) Undergo a collision within the current cell,
        //   (1) Cross a facet of the current cell,
        //   (2) Reach the end of the time step and enter census,
        //
#ifdef EXPONENTIAL_TALLY
        monteCarlo->_tallies->TallyCellValue( exp(rngSample(&mc_particle.random_number_seed)) , mc_particle.domain, cell_tally_index, mc_particle.cell);
#endif   
        MC_Segment_Outcome_type::Enum segment_outcome = MC_Segment_Outcome(monteCarlo, mc_particle, flux_tally_index);

        ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._numSegments);

        mc_particle.num_segments += 1.;  /* Track the number of segments this particle has
                                            undergone this cycle on all processes. */
#ifdef DO_COLLISION_SORT
#ifdef __CUDA_ARCH__

        if( sIndex[threadIdx.x] != -1 )
        {
            sEvent[threadIdx.x] = segment_outcome; 
            sParts[threadIdx.x] = mc_particle;
        }

        __syncthreads();

        if(threadIdx.x == 0)
        {
            int front = 0;
            //If the threads in this block would reach past num_particles set the back to be equivalent to num_particles for this block
            int back = ( (blockIdx.x+1)*blockDim.x > num_particles ) ? ( blockDim.x - (((blockIdx.x+1)*blockDim.x) - num_particles) - 1 ) : (blockDim.x-1);
            while(front < back)
            {
                if(sEvent[front] == MC_Segment_Outcome_type::Collision)
                {
                    front++;                
                }
                else if(sEvent[back] == MC_Segment_Outcome_type::Collision)
                {
                    int tIndex    = sIndex[back]; 
                    sIndex[back]  = sIndex[front];
                    sIndex[front] = tIndex;
                    sEvent[back]  = sEvent[front];
                    sEvent[front] = MC_Segment_Outcome_type::Collision; 
                    MC_Particle temp = sParts[back];
                    sParts[back]  = sParts[front];
                    sParts[front] = temp;

                    front++;
                }
                else
                {
                    back--;
                }                   
            }
        }

        __syncthreads();

        particle_index  = sIndex[threadIdx.x];
        segment_outcome = sEvent[threadIdx.x];
        mc_particle     = sParts[threadIdx.x];

        __syncthreads();

#endif
#endif  

        switch (segment_outcome) {
        case MC_Segment_Outcome_type::Collision:
            {
            // The particle undergoes a collision event producing:
            //   (0) Other-than-one same-species secondary particle, or
            //   (1) Exactly one same-species secondary particle.
            if (CollisionEvent(monteCarlo, mc_particle, tally_index ) == MC_Collision_Event_Return::Continue_Tracking)
            {
                keepTrackingThisParticle = true;
            }
            else
            {
                keepTrackingThisParticle = false;
            }
            }
            break;
    
        case MC_Segment_Outcome_type::Facet_Crossing:
            {
                // The particle has reached a cell facet.
                MC_Tally_Event::Enum facet_crossing_type = MC_Facet_Crossing_Event(mc_particle, monteCarlo, particle_index, processingVault);

                if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Transit_Exit)
                {
                    keepTrackingThisParticle = true;  // Transit Event
                }
                else if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Escape)
                {
                    ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._escape);
                    mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Escape;
                    mc_particle.species = -1;
                    keepTrackingThisParticle = false;
                }
                else if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Reflection)
                {
                    MCT_Reflect_Particle(monteCarlo, mc_particle);
                    keepTrackingThisParticle = true;
                }
                else
                {
                    // Enters an adjacent cell in an off-processor domain.
                    //mc_particle.species = -1;
                    keepTrackingThisParticle = false;
                }
            }
            break;
    
        case MC_Segment_Outcome_type::Census:
            {
                // The particle has reached the end of the time step.
                processedVault->pushParticle(mc_particle);
                ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._census);
                keepTrackingThisParticle = false;
                break;
            }
            
        default:
           qs_assert(false);
           break;  // should this be an error
        }

#ifdef DO_COLLISION_SORT
#ifdef __CUDA_ARCH__
        __syncthreads();
        if( sIndex[threadIdx.x] != -1 )
        {
            sEvent[threadIdx.x] = ( keepTrackingThisParticle == true ) ? MC_Segment_Outcome_type::Collision : MC_Segment_Outcome_type::Facet_Crossing; 
            sParts[threadIdx.x] = mc_particle;
        }

        __syncthreads();

        if(threadIdx.x == 0)
        {
            int front = 0;
            int back = ( (blockIdx.x+1)*blockDim.x > num_particles ) ? ( blockDim.x - (((blockIdx.x+1)*blockDim.x) - num_particles) - 1 ) : (blockDim.x-1);
            while(front < back)
            {
                if(sEvent[front] == MC_Segment_Outcome_type::Collision)
                {
                    front++;                
                }
                else if(sEvent[back] == MC_Segment_Outcome_type::Collision)
                {
                    int tIndex    = sIndex[back];               
                    sIndex[back]  = sIndex[front];
                    sIndex[front] = tIndex;
                    sEvent[back]  = sEvent[front];
                    sEvent[front] = MC_Segment_Outcome_type::Collision;              
                    MC_Particle temp = sParts[back];
                    sParts[back]  = sParts[front];
                    sParts[front] = temp;
                    front++;
                }
                else
                {
                    back--;
                }                   
            }
        }

        __syncthreads();

        keepTrackingThisParticle = ( sEvent[threadIdx.x] == MC_Segment_Outcome_type::Collision ) ? true : false;
        sEvent[threadIdx.x] = MC_Segment_Outcome_type::Initialize;

        particle_index      = sIndex[threadIdx.x];
        segment_outcome     = MC_Segment_Outcome_type::Initialize;
        mc_particle         = sParts[threadIdx.x];

        __syncthreads();

        if( !keepTrackingThisParticle )
        {
            sIndex[threadIdx.x] = -1;
            processingVault->putParticle( mc_particle, particle_index );
            processingVault->invalidateParticle( particle_index );
        }
        __syncthreads();
#endif
#endif  
    } while ( keepTrackingThisParticle );

    //Make sure this particle is marked as completed
}
HOST_DEVICE_END

#if defined (HAVE_CUDA)
#define MAX_THREADS_PER_BLOCK NUM_THREADS
__global__ void CycleTrackingKernel( MonteCarlo* monteCarlo, int num_particles, ParticleVault* processingVault, ParticleVault* processedVault )
{
    __shared__ int sIndex[MAX_THREADS_PER_BLOCK];
    __shared__ MC_Segment_Outcome_type::Enum sEvent[MAX_THREADS_PER_BLOCK];
    __shared__ MC_Particle sParts[MAX_THREADS_PER_BLOCK];
    int global_index = getGlobalThreadID(); 

    if( global_index < num_particles )
    {
        CycleTrackingGuts( monteCarlo, global_index, processingVault, processedVault, num_particles, sIndex, sEvent, sParts );
    }
}
#endif

void CycleTrackingKernelLaunch( MonteCarlo* monteCarlo, int numParticles, ParticleVault* processingVault, ParticleVault* processedVault )
{
    #if defined (HAVE_CUDA)
        dim3 grid(1,1,1);
        dim3 block(1,1,1);
        int runKernel = ThreadBlockLayout( grid, block, numParticles);

        //Call Cycle Tracking Kernel
        if( runKernel )
            CycleTrackingKernel<<<grid, block >>>( monteCarlo, numParticles, processingVault, processedVault );
    
        //Synchronize the stream so that memory is copied back before we begin MPI section
        cudaPeekAtLastError();
        cudaDeviceSynchronize();
    #else
        qs_assert(false);
    #endif
}

void CycleTrackingFunctionLaunch( MonteCarlo* monteCarlo, int numParticles, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault )
{
    MC_Segment_Outcome_type::Enum sEvent[1]; int sIndex[1]; MC_Particle sParts[1];
    CycleTrackingGuts( monteCarlo, particle_index, processingVault, processedVault, numParticles, sIndex, sEvent, sParts );
}
