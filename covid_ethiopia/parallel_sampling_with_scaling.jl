using Distributed


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end


# stuff needed on workers and main
@everywhere begin
    
    using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
    using Plots
    #using PlotlyJS
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
    using StatsPlots
    using ArviZ
    using HDF5

    # Lorenzos packages
    using Particles
    using ParticlesDE
    using StaticDistributions 
    
    
    # Slurm Job-array

    task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    task_id = parse(Int64, task_id_str)

    particles_set = [10,100,200,400,600]



    # set hyperparamters
    niter = 100000
    nparticles = particles_set[task_id+1]
end    

# stuff only needed on workers
@everywhere workers() begin

    using PyCall
    pypesto = pyimport("pypesto")

    function Chains_from_pypesto(result; kwargs...)
        trace_x = result.sample_result["trace_x"] # parameter values
        trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
        samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
        samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
        samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
        param_names = Symbol.(result.problem.x_names)
        return Chains(
            samples,
            vcat(param_names, :lp),
            (parameters = param_names, internals = [:lp]);
            kwargs...
        )
    end

    include("real_model_real_data.jl")
    llh = likelihood(nparticles)

    # for pypesto we need the negative log-likelihood
    neg_llh = let llh = llh
        p -> begin
            return -llh(p)
        end
    end

    # transform to pypesto objective
    objective = pypesto.Objective(fun=neg_llh)


    problem = pypesto.Problem(
        objective,
        x_names=["gamma", "kappa", "beta", "tevent", "scaling"],
        lb=[10,1,0.01,130, 1], # parameter bounds
        ub=[25,9,1,360, 10], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    sampler = pypesto.sample.AdaptiveMetropolisSampler()
    
    # sample start value
    x0 = rand(prior_dist)
    
    # sample
    function chain()
        result = pypesto.sample.sample(
                    problem,
                    n_samples=niter,
                    x0=x0, # starting point
                    sampler=sampler,
                    )
       return  Chains_from_pypesto(result)
    end
end

jobs = [@spawnat(i, @timed(chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end

chs = MCMCChains.Chains(chains, [:gamma, :kappa, :beta, :tevent, :scaling, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:gamma, :kappa, :beta, :tevent, :scaling], :internals => [:lp]))
stop_time = mean([all_chains[i].time for i in 1:nworkers()])
complete_chain = setinfo(complete_chain, (start_time=1.0, stop_time=stop_time))

print("Mean duration per chain: ", stop_time)
# store results
h5open("./output/real_data/real_scaling_no_prior_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open("output/real_data/time_real_scaling_no_prior_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end







