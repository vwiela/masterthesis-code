using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
using DataFrames
using CSV
using Random
using Distributions
using SBML
using SymbolicUtils
using StaticArrays
using Catalyst
using AdvancedMH
using MCMCChains
using MCMCChainsStorage
import StatsPlots

using Particles
using ParticlesDE
using StaticDistributions

include("utilities.jl")

# load the model definition
include("sir_model.jl")

# specify the particle number
nparticles = 100

# store the true model parameters
true_par = problem.p

# define the likelihood function
llh_no_priors = LogLikelihood_NoGradient(ssm, real_data; nparticles=nparticles)
no_priors_model = DensityModel(llh_no_priors)

# specify parameter values on log10 scale for the sinlge loops
beta_val = -3:0.01:0; 
gamma_val = -3:0.01:0;

# Slurm Job-array

task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
task_id = parse(Int64, task_id_str)

# explore likelihood landscape of beta
if (task_id==0)
    beta_likelihoods = Vector{Float64}(undef, length(beta_val))
    beta_stde = Vector{Float64}(undef, length(beta_val))
    for i in range(1,length(beta_val))
        par = copy(true_par)
        be = 10^(beta_val[i])
        par[1] = be
        likeli = Vector{Float64}(undef, 500)
        for j in range(1, 500) # 500 iterations
            @inbounds likeli[j] = llh_no_priors(par)
        end    
        @inbounds beta_likelihoods[i] = mean(likeli)
        @inbounds beta_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=beta_val, Likelihood=beta_likelihoods, Std_Error=beta_stde);
    CSV.write("./output/sir_model_beta_likelihoods_$(nparticles)p_no_priors.csv", df)
end

# explore likelihood landscape of gamma
if (task_id==1)
    gamma_likelihoods = Vector{Float64}(undef, length(gamma_val))
    gamma_stde = Vector{Float64}(undef, length(gamma_val))
    for i in range(1,length(gamma_val))
        par = copy(true_par)
        ga = 10^(gamma_val[i])
        par[2] = ga
        likeli = Vector{Float64}(undef, 500)
        for j in range(1, 500) # 500 iterations 
            @inbounds likeli[j] = llh_no_priors(par)
        end    
        @inbounds gamma_likelihoods[i] = mean(likeli)
        @inbounds gamma_stde[i] = std(likeli)
        print(i)
    end

    df = DataFrame(Parameter=gamma_val, Likelihood=gamma_likelihoods, Std_Error=gamma_stde);
    CSV.write("./output/sir_model_gamma_likelihoods_$(nparticles)p_no_priors.csv", df)
end