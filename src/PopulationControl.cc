#include "PopulationControl.hh"
#include "MC_Processor_Info.hh"
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Particle.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "utilsMpi.hh"
#include "NVTX_Range.hh"
#include <vector>

namespace
{
   void PopulationControlGuts(const double splitRRFactor, 
                              uint64_t currentNumParticles,
                              ParticleVaultContainer* my_particle_vault,
                              Balance& taskBalance);
}

void PopulationControl(MonteCarlo* monteCarlo, bool loadBalance)
{
    NVTX_Range range("PopulationControl");

    uint64_t targetNumParticles = monteCarlo->_params.simulationParams.nParticles;
    uint64_t globalNumParticles = 0;
    uint64_t numParticles = monteCarlo->_particleVaultContainer->sizeProcessing();
   
    //Count number of particles globally
    mpiAllreduce(&numParticles, &globalNumParticles, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
     
    //Tallies to track population changes
    Balance & taskBalance = monteCarlo->_tallies->_balanceTask[0];

    double splitRRFactor = (double)targetNumParticles / (double)globalNumParticles;

    if (targetNumParticles == globalNumParticles)  // no need to split if population is already correct.
        PopulationControlGuts(splitRRFactor, numParticles, monteCarlo->_particleVaultContainer, taskBalance);


    if( loadBalance )
    {
        uint64_t localNumParticles = 0;
        int localNumRanks = monteCarlo->processor_info->shm_num_processors;
        double factor     = monteCarlo->processor_info->shm_perf_factor; 
        mpiAllreduce(&numParticles, &localNumParticles, 1, MPI_UINT64_T, MPI_SUM, monteCarlo->processor_info->comm_mc_shmcomm);
        targetNumParticles = (uint64_t) (( (double) localNumParticles / (double) localNumRanks ) * factor);
        splitRRFactor = (double) targetNumParticles / (double) localNumParticles;

        printf("LOAD-BALANCE[%d]: localNumParticles: %lu\t localNumRanks: %d\t factor: %g\n targetNumParticles: %lu \t splitRRFactor: %g\n",
               monteCarlo->processor_info->shm_rank, localNumParticles, localNumRanks, factor, targetNumParticles, splitRRFactor );
            

        if (targetNumParticles == localNumParticles) //Do a second splitRR to loadBalance ranks pased on a performance factor  
            PopulationControlGuts(splitRRFactor, numParticles, monteCarlo->_particleVaultContainer, taskBalance);
    }

    monteCarlo->_particleVaultContainer->collapseProcessing();

    return;
}


namespace
{
void PopulationControlGuts(const double splitRRFactor, uint64_t currentNumParticles, ParticleVaultContainer* my_particle_vault, Balance& taskBalance)
{
    uint64_t vault_size = my_particle_vault->getVaultSize();
    uint64_t fill_vault_index = currentNumParticles / vault_size;

    // March backwards through the vault so killed particles doesn't mess up the indexing
    for (int particleIndex = currentNumParticles-1; particleIndex >= 0; particleIndex--)
    {
        uint64_t vault_index = particleIndex / vault_size; 

        ParticleVault& taskProcessingVault = *( my_particle_vault->getTaskProcessingVault(vault_index) );

        uint64_t taskParticleIndex = particleIndex%vault_size;

        MC_Base_Particle &currentParticle = taskProcessingVault[taskParticleIndex];
        double randomNumber = rngSample(&currentParticle.random_number_seed);
        if (splitRRFactor < 1)
        {
            if (randomNumber > splitRRFactor)
            {
                // Kill
	            taskProcessingVault.eraseSwapParticle(taskParticleIndex);
	            taskBalance._rr++; 
	        }
	        else
	        {
	            currentParticle.weight /= splitRRFactor;
	        }
        }
        else if (splitRRFactor > 1)
        {
            // Split
	        int splitFactor = (int)floor(splitRRFactor);
	        if (randomNumber > (splitRRFactor - splitFactor)) { splitFactor--; }
	  
	        currentParticle.weight /= splitRRFactor;
	        MC_Base_Particle splitParticle = currentParticle;
	  
	        for (int splitFactorIndex = 0; splitFactorIndex < splitFactor; splitFactorIndex++)
	        {
	            taskBalance._split++;
	     
	            splitParticle.random_number_seed = rngSpawn_Random_Number_Seed(
			        &currentParticle.random_number_seed);
	            splitParticle.identifier = splitParticle.random_number_seed;

                my_particle_vault->addProcessingParticle( splitParticle, fill_vault_index );

	        }
        }
    }
}
} // anonymous namespace


// Roulette low-weight particles relative to the source particle weight.
void RouletteLowWeightParticles(MonteCarlo* monteCarlo)
{
    NVTX_Range range("RouletteLowWeightParticles");

    const double lowWeightCutoff = monteCarlo->_params.simulationParams.lowWeightCutoff;

    if (lowWeightCutoff > 0.0)
    {

        uint64_t currentNumParticles = monteCarlo->_particleVaultContainer->sizeProcessing();
        uint64_t vault_size          = monteCarlo->_particleVaultContainer->getVaultSize();

        Balance& taskBalance = monteCarlo->_tallies->_balanceTask[0];

	    // March backwards through the vault so killed particles don't mess up the indexing
	    const double source_particle_weight = monteCarlo->source_particle_weight;
	    const double weightCutoff = lowWeightCutoff*source_particle_weight;

	    for ( int64_t particleIndex = currentNumParticles-1; particleIndex >= 0; particleIndex--)
	    {
            uint64_t vault_index = particleIndex / vault_size; 

            ParticleVault& taskProcessingVault = *(monteCarlo->_particleVaultContainer->getTaskProcessingVault(vault_index));
            uint64_t taskParticleIndex = particleIndex%vault_size;
	        MC_Base_Particle &currentParticle = taskProcessingVault[taskParticleIndex];

	        if (currentParticle.weight <= weightCutoff)
	        {
	            double randomNumber = rngSample(&currentParticle.random_number_seed);
	            if (randomNumber <= lowWeightCutoff)
	            {
		            // The particle history continues with an increased weight.
		            currentParticle.weight /= lowWeightCutoff;
	            }
	            else
	            {
		            // Kill
		            taskProcessingVault.eraseSwapParticle(taskParticleIndex);
		            taskBalance._rr++;
	            } 
	        }
	    }
        monteCarlo->_particleVaultContainer->collapseProcessing();
    }
}
