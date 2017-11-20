#include "DeclareMacro.hh"

// Forward Declaration
class ParticleVault;
class MonteCarlo;
class MC_Particle;

void CycleTrackingKernelLaunch( MonteCarlo* monteCarlo, int numParticles, ParticleVault* processingVault, ParticleVault* processedVault );
void CycleTrackingFunctionLaunch( MonteCarlo* monteCarlo, int numParticles, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault );

