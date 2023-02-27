using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
using Plots
plotlyjs()
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

include("real_model_with_scaling.jl")


# Slurm Joba-array

task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
task_id = parse(Int64, task_id_str)

# task_id = 5 #not entering the single computations just the all combinations at the end

# calculating corresponding likelihoods

# getting possible parameter vectors
gamma_val = 5:0.1:22;
kappa_val = 1:0.1:15;
beta_val = -5:0.05:-1; # on log10 scale for the single loops
t_val = 100:1:360;
scaling_val = 0:0.1:10


# scaling
if (task_id == 0)

    gamma_likelihoods = Vector{Float64}(undef, length(gamma_val))
    gamma_stde = Vector{Float64}(undef, length(gamma_val))
    for i in range(1,length(gamma_val))
        p = copy(true_par)
        ga = gamma_val[i]
        p[1] = ga
        likeli = Vector{Float64}(undef, 50)
        for j in range(1, 50)
            @inbounds likeli[j] = llh(p)
        end    
        @inbounds gamma_likelihoods[i] = mean(likeli)
        @inbounds gamma_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=gamma_val, Likelihood=gamma_likelihoods, Std_Error=gamma_stde);

    CSV.write("./output/real_model_SparseSynth_scaling_likelihoods_no_priors.csv", df)
end


# gamma
if (task_id == 1)


    gamma_likelihoods = Vector{Float64}(undef, length(gamma_val))
    gamma_stde = Vector{Float64}(undef, length(gamma_val))
    for i in range(1,length(gamma_val))
        p = copy(true_par)
        ga = gamma_val[i]
        p[1] = ga
        likeli = Vector{Float64}(undef, 50)
        for j in range(1, 50)
            @inbounds likeli[j] = llh(p)
        end    
        @inbounds gamma_likelihoods[i] = mean(likeli)
        @inbounds gamma_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=gamma_val, Likelihood=gamma_likelihoods, Std_Error=gamma_stde);

    CSV.write("./output/real_model_SparseSynth_gamma_likelihoods_no_priors.csv", df)
end

# kappa
if (task_id == 2)
    kappa_likelihoods = Vector{Float64}(undef, length(kappa_val))
    kappa_stde = Vector{Float64}(undef, length(kappa_val))
    for i in range(1,length(kappa_val))
        p = copy(true_par)
        ka = kappa_val[i]
        p[2] = ka
        likeli = Vector{Float64}(undef, 50)
        for j in range(1, 50)
            @inbounds likeli[j] = llh(p)
        end    
        @inbounds kappa_likelihoods[i] = mean(likeli)
        @inbounds kappa_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=kappa_val, Likelihood=kappa_likelihoods, Std_Error=kappa_stde);

    CSV.write("./output/real_model_SparseSynth_kappa_likelihoods_no_priors.csv", df)
end
# beta
if (task_id==3)

    beta_likelihoods = Vector{Float64}(undef, length(beta_val))
    beta_stde = Vector{Float64}(undef, length(beta_val))
    for i in range(1,length(beta_val))
        p = copy(true_par)
        be = 10^(beta_val[i])
        p[3] = be
        likeli = Vector{Float64}(undef, 50)
        for j in range(1, 50)
            @inbounds likeli[j] = llh(p)
        end    
        @inbounds beta_likelihoods[i] = mean(likeli)
        @inbounds beta_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=beta_val, Likelihood=beta_likelihoods, Std_Error=beta_stde);

    CSV.write("./output/real_model_SparseSynth_beta_likelihoods_no_priors.csv", df)
end

# tevent
if (task_id==4)
    t_likelihoods = Vector{Float64}(undef, length(t_val))
    t_stde = Vector{Float64}(undef, length(t_val))
    for i in range(1,length(t_val))
        p = copy(true_par)
        time = t_val[i]
        p[4] = time
        likeli = Vector{Float64}(undef, 50)
        for j in range(1, 50)
            @inbounds likeli[j] = llh(p)
        end    
        @inbounds t_likelihoods[i] = mean(likeli)
        @inbounds t_stde[i] = std(likeli)
    end

    df = DataFrame(Parameter=t_val, Likelihood=t_likelihoods, Std_Error=t_stde);

    CSV.write("./output/real_model_SparseSynth_t_likelihoods_no_priors.csv", df)
end

# all joint
if (task_id ==5)
    
    gamma_val = 14:0.5:20;
    kappa_val = 2:0.4:7;
    beta_val = 0.04:0.002:0.1; # on log scale for the sinlge loops
    t_val = 160:1:180;
    all_par_comb = collect(Iterators.product(gamma_val, kappa_val, beta_val, t_val));

    likelihoods = Vector{Float64}(undef, length(all_par_comb))
    parameter = Vector{Tuple{Float64, Float64, Float64, Int64}}()
    for i in range(1,length(all_par_comb))
        p = all_par_comb[i]
        par = copy(true_params)
        par[1:4] .= p
        @inbounds likelihoods[i] = llh(par)
        push!(parameter, p)
    end


    df = DataFrame(Parameter= parameter, Likelihood=likelihoods);

    CSV.write("./output/real_model_SparseSynth_likelihoods_for_test_data.csv", df)
end
    
    
