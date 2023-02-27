using Distributed # package for distributed computing in julia


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(dirname(pwd()))
  Pkg.instantiate(); Pkg.precompile()
end

# stuff needed on workers and main
@everywhere begin   
    using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
    using Plots
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
    
    # set hyperparamters; number of iterations and particles
    niter = 100000
    nparticles = 100
end    


# stuff only needed on workers
@everywhere workers() begin

    using PyCall
    pypesto = pyimport("pypesto")

    # convert PyPesto result to MCMCChains.jl chain type
    function Chains_from_pypesto(result; kwargs...)
        trace_x = result.sample_result["trace_x"] # parameter values
        trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
        samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
        samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
        samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
        param_names = Symbol.(result.problem.x_names)
        chain = Chains(
            samples,
            vcat(param_names, :lp),
            (parameters = param_names, internals = [:lp]);
            kwargs...
        )
        return chain
    end
    
    # load model
    include("sir_model.jl")
    llh = likelihood(nparticles)

    # for pypesto we need the negative log-likelihood
    neg_llh = let llh = llh
        p -> begin
            return -llh(p)
        end
    end

    # transform to pypesto objective
    objective = pypesto.Objective(fun=neg_llh)


    # create pypesto problem

    pypesto_problem = pypesto.Problem(
        objective,
        x_names=["beta", "gamma"],
        lb=[0.001, 0.001], # parameter bounds
        ub=[1, 1], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    sampler = pypesto.sample.AdaptiveMetropolisSampler();


    # get initial parameters
    init_par = rand(prior_dist)

    # run it once to get rid of compilation time for next runs
    prep_timing = @timed pypesto.sample.sample(
            pypesto_problem,
            n_samples=10,
            x0=Vector(init_par), # starting point
            sampler=sampler,
        )
        
     # function for sampling and conversion 
    function chain()
        result = pypesto.sample.sample(
                    pypesto_problem,
                    n_samples=niter,
                    x0=Vector(init_par), # starting point
                    sampler=sampler,
                    )
       return  Chains_from_pypesto(result)
    end
end

# initialize and run the jobs for the workers
jobs = [@spawnat(i, @timed(chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

# get the chains
for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end


chs = MCMCChains.Chains(chains, [:beta, :gamma, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:beta, :gamma], :internals => [:lp]))

# get mean computation time per chain
stop_time = mean([all_chains[i].time for i in 1:nworkers()])

# store results
print("Mean runtime for $nparticles particles $niter iterations: ", stop_time)

h5open("./output/paralell_sir_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open("output/time_parallel_sir_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end



