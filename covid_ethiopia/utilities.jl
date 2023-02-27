# Utility functions for the two-variant model experiments

using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
using Plots
using DataFrames
using Random
using Distributions
using Distances

using Particles
using ParticlesDE
using StaticDistributions


# calculate optimal sizes in pixels for latex documents textwidth
dpi = 100 # use the Plots.jl default
width_pts = 455.244
inches_per_points = 1.0/72.27
width_inches = width_pts*inches_per_points
width_px = width_inches*dpi
# note that default in julia is (600,400)


# plot variance of log-likelihood estimator of the Bootstrap filter vs. number of particles
function my_plot_llh_vs_nparticles(ssm::StateSpaceModel, parameters, obs, nparticles::AbstractVector{<:Integer}; nruns::Integer=50)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    variances = Vector{Float64}(undef, length(nparticles))
    k = 1
    i = 1
    for n in nparticles
        llh = LogLikelihood_NoGradient(ssm, obs; nparticles=n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = llh(parameters)
            k += 1
        end
        variances[i] = round(var(y[((i-1)*nruns+1):i*nruns]); digits=3)
        i += 1
    end
    y = reshape(y, (nruns,length(nparticles)))
    x_names = reshape(string.(nparticles), (1,length(nparticles)))
    labels_list = ["$(nparticles[j])p var: $(variances[j])" for j in 1:length(nparticles)] 
    labels = reshape(labels_list, (1,length(nparticles)))
    return Plots.violin(x_names, y, xlabel="particle number",
        trim=false, labels=labels, legend=:outerright,
        size=(width_px, width_px*2/3))
end

    
colors = palette(:Accent_8) #set some color scheme, because I like lighter colors for the following plot

# plot coverage of the fitlering distributions from a Bootstrap Fitler run
function my_plot_filter(ssm::StateSpaceModel, parameters, components::AbstractVector{<:Integer}, 
                        tobs; nparticles::Integer, nsigmas::Real=2)
    
    length(components) > length(colors) && error("too many components, not enough colors")
    ncomp = length(components)
    hidden, obs=rand(ssm, parameters, length(tobs));
    bf = BootstrapFilter(ssm, obs)
    pf = SMC(
        bf, parameters, nparticles,
        (filter=RunningSummary(MeanAndVariance(), FullHistory()), ),
    )
    offlinefilter!(pf);
    hist = pf.history_run.filter
    comp_means = Vector{Vector{Float64}}(undef, ncomp)
    comp_variances = Vector{Vector{Float64}}(undef, ncomp)
    for i in 1:ncomp
        mean = [hist[j].mean[components[i]] for j in 1:length(tobs)]
        variance = [hist[j].var[components[i]] for j in 1:length(tobs)]
        comp_means[i] = mean
        comp_variances[i] = variance
    end
    plt = Plots.plot(size=(width_px/2,width_px/3))
    x = 1:length(tobs)
    for i in 1:ncomp        
        plot!(plt, x, comp_means[i], color=colors[i], label="")
        plot!(plt, x, comp_means[i]+nsigmas*sqrt.(comp_variances[i]),
            fillrange=comp_means[i]-nsigmas*sqrt.(comp_variances[i]),
            alpha=0.35, color=colors[i], label="")
        scatter!(plt, x, [hidden[j][components[i]] for j in 1:length(tobs)], color=colors[i], label="$(components[i]) component")
    end
        return plt
end;




# calculate burn-in using the Geweke diagnostic
function burn_in_from_geweke(chain::Chains, z_threshold::Float64=2.0)
    niter = size(chain)[1]
    npar = size(chain)[2]-1 #-1 because of the internal lp
    nchains = size(chain)[3]
    # number of fragments
    n = 10
    step = Int(floor(niter/n))
    fragments = 0:step:niter-20
    z = zeros(length(fragments), npar, nchains)
    burn_in_list = []
    for j in 1:nchains
        for (i, indices) in enumerate(fragments)
            z[i, :, j] = DataFrame(gewekediag(chain[indices+1:end,:, j]))[!,"zscore"][1]
        end
        max_z = maximum(abs.(z[:,:,j]), dims=2) #note that it returns a matrix with one column
        idxs = sortperm(max_z[:,1], rev=true) #sort descending
        alpha2 = z_threshold * ones(length(idxs))
        max_z = maximum(abs.(z[:,:,j]), dims=2)
        idxs = sortperm(max_z[:,1], rev=true)
        alpha2 = z_threshold * ones(length(idxs))
        for k in 1:length(max_z)
           alpha2[idxs[k]] = alpha2[idxs[k]]/(length(fragments)-findfirst(==(k), idxs) +1) 
        end
        if any(alpha2.>max_z)
            burn_in = findfirst((alpha2 .> max_z)[:,1]) * step
        else
            burn_in = niter
        end
        append!(burn_in_list, burn_in)
    end
    return Int64(maximum(burn_in_list)) #a conservative choice is the maximum of all chains; or median for a less conservative choice
end


# compute the mean-squared jumping distance
function mean_squared_jumping_distance(chain, parameter_names::Vector{String}; burn_in::Integer=0)
    burnin = burn_in+1
    niter = size(chain)[1]
    nchains = size(chain)[3]
    names = cat(parameter_names, "combined", dims=1)
    msjd_df = DataFrame()
    for idx_chain in 1:nchains
        df = DataFrame()
        for p in parameter_names
            m = mean([(chain[i,p,idx_chain] - chain[i-1,p,idx_chain])^2 for i in (burnin+1):niter])
            df[!,p] = [m]
        end
        m_c = mean([euclidean(chain.value[i,:,idx_chain][1:length(parameter_names)],chain.value[i-1,:,idx_chain][1:length(parameter_names)])^2 for i in (burnin+1):niter])
        df[!,"total_msjd"] = [m_c]
        append!(msjd_df, df)
    end
    return msjd_df
end


# compute mean acceptance rate
function acceptance_rate(chain; burn_in::Integer=0)
    burnin = burn_in+1
    nchains=size(chain)[3]
    niter = size(chain)[1]
    acc_rate_df = DataFrame(acceptance_rate=[]) 
    for idx_chain in 1:nchains
        rate = mean([chain.value[i,:,idx_chain] == chain.value[i+1,:,idx_chain] ? 1 : 0 for i in burnin:niter-1])
        push!(acc_rate_df, [rate])
   end
   return acc_rate_df
end


# compute unique ancestor for a particle fitler run
function unique_ancestors_at_previous_times(ancestors::AbstractVector{<:AbstractVector{Int}}; check::Bool=true)
    # NB thanks to the fact that ancestor indices are always ordered,
    #    we only need the history for the first and last particle
    !check || all(issorted, ancestors) || error("ancestors do not satisfy ordering assumptions")
    unique = Vector{Int}(undef, length(ancestors) + 1)
    k = length(ancestors) + 1
    t = lastindex(ancestors)::Int
    B = collect(Int, axes(last(ancestors), 1))
    @inbounds while true
        unique[k] = length(Set(B))
        t ≥ firstindex(ancestors) || break
        B .= getindex.(Ref(ancestors[t]), B)
        t -= 1
        k -= 1
    end
    return unique
end


# plot genealogical tree of the ancestor history for a particle filter run
function plot_ancestor_tree(ancestors)
    nparticles = length(ancestors[1])
    nobs = length(ancestors)
    lines = Array{Float64}(undef, nparticles, nobs+1);
    B = collect(Int, axes(last(ancestors), 1));
    lines[:,1] = B
    for j in 1:nobs
        B.= getindex.(Ref(ancestors[end-(j-1)]), B)
        lines[:,j+1] = B
    end
    p1 = Plots.plot(1:(nobs+1), reverse(lines[1,:]),color=:black, leg=false)
    for i in 2:nparticles
        Plots.plot!(p1, 1:(nobs+1), reverse(lines[i,:]), color=:black, xlabel="", ylabel="", ticks=false)
    end
    return p1
end


# compute resimulation error given a sde-model and a parameter vector
function check_function(SDE_problem, par_new, tobs, y_load) 
    
    prob_func = let SDE_problem = SDE_problem
        (prob, i, repeat) -> begin
            u0_new = copy(SDE_problem.u0)
            parameter = copy(SDE_problem.p)
            parameter[1:length(par_new)] .= par_new
            return remake(SDE_problem; u0=u0_new, p=parameter)
        end
    end
    # callbacks for injection of variant
    if length(par_new)>=4
        tevent = Float64(par_new[4])
    else 
        tevent = 170.0
    end
    jumpsize = Float64(1.0)

    make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
    cb = make_event_cb(tevent, jumpsize)

    # solver settings
    solve_alg = PosEM()
    solve_kwargs = (dt=1e-2, callback=cb, tstops=[tevent])
    nothing
    
    
    output_func = (sol, i) -> ([rand(fobs(sol(t), nothing, t)) for t in tobs], false)
    ensemble_prob = EnsembleProblem(SDE_problem; output_func, prob_func)
    sim = solve(ensemble_prob, solve_alg; trajectories=500, solve_kwargs...);
    
    Prev = Matrix{Float64}(undef, length(sim), length(tobs))
    Infc = Matrix{Float64}(undef, length(sim), length(tobs))
    for i in range(1,length(sim))
        prevalence = []
        infections = []
        for j in range(1, length(tobs))
            push!(prevalence, sim.u[i][j][1])
            push!(infections, sim.u[i][j][2])
        end
        Prev[i,:] = prevalence
        Infc[i,:] = infections
    end
    return mean(Prev; dims=1), mean(Infc; dims=1), std(Prev;dims=1), std(Infc;dims=1)
end


# visualize the resimulation error
function data_check_plot(SDE_problem, par_new, tobs, y_load)
    Prev_mean, Infc_mean, Prev_std, Infc_std = check_function(problem, true_params, tobs, y_load)
    x = range(1,length(tobs))
    Prev_diff = (Prev_mean[1,:] - y_load[!, "y1"]).^2
    Prev_mse = mean(Prev_diff./y_load[!,"y1"])
    Infc_diff = (Infc_mean[1,:]-y_load[!,"y2"]).^2
    Infc_mse = mean(Infc_diff./y_load[!,"y2"])

    plt1 = Plots.plot()
    Plots.plot!(plt1, x, Prev_mean[1,:], color=:black, label="Simulation mean")
    Plots.plot!(plt1, x, Prev_mean[1,:]-1.96*Prev_std[1,:], fillrange=Prev_mean[1,:]+1.96*Prev_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt1, x, y_load[!,"y1"], line=:scatter, marker_color=:black, label="Data")

    plt2 = Plots.plot()
    Plots.plot!(plt2, x, Infc_mean[1,:], color=:black, label="Simulation mean")
    Plots.plot!(plt2, x, Infc_mean[1,:]-1.96*Infc_std[1,:], fillrange=Infc_mean[1,:]+1.96*Infc_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt2, x, y_load[!,"y2"], line=:scatter, marker_color=:black, label="Data")
    print("Relative mean squared error in... Prev: $Prev_mse      Infc: $Infc_mse")
    return plt1, plt2 
end


# resimulation error for the chains obtained with synthetically generated data
function check_synth_chains(SDE_problem, chain, tobs, y_load; burn_in::Integer=0)
    burnin = burn_in+1
    nsim = 500
    
    Prev = Matrix{Float64}(undef, nsim, length(tobs))
    Infc = Matrix{Float64}(undef, nsim, length(tobs))
    
    # Handwritten like Ensemble problem
    for i in 1:1:nsim
        idx_iter = burn_in+rand(axes(chain.value[burnin:end,:,:], 1))
        idx_chain = rand(axes(chain.value, 3))
        p = chain.value[idx_iter, :, idx_chain][2:5] #first is the lp value
        u0_new = copy(SDE_problem.u0)
        parameter = copy(SDE_problem.p)
        parameter[1] = p[2] 
        parameter[2] = p[3]
        parameter[3] = p[1]
        parameter[4] = p[4]
        
        SDE_problem_new = remake(SDE_problem; u0=u0_new, p=parameter)
        # solver settings
        solve_alg = PosEM()
        solve_kwargs = (dt=1e-2,)
        nothing 
    
        sol = solve(SDE_problem_new, solve_alg; solve_kwargs...)
        sim_data = [rand(fobs(sol(t), nothing, t)) for t in tobs]

        prevalence = []
        infections = []
        for j in range(1, length(tobs))
            push!(prevalence, sim_data[j][1])
            push!(infections, sim_data[j][2])
        end
        Prev[i,:] = prevalence
        Infc[i,:] = infections
    end
    return mean(Prev; dims=1), mean(Infc; dims=1), std(Prev;dims=1), std(Infc;dims=1)    
end


# visualization of resimulation error for synth. data chains
function data_synth_check(SDE_problem, chain, tobs, y_load; burn_in::Integer=0)
    Prev_mean, Infc_mean, Prev_std, Infc_std = check_synth_chains(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    x = range(1,length(tobs))
    
    plt1 = Plots.plot(title = "Prevalence")
    Plots.plot!(plt1, x, Prev_mean[1,:], label="simulation mean")
    Plots.plot!(plt1, x, Prev_mean[1,:]-1.96*Prev_std[1,:], fillrange=Prev_mean[1,:]+1.96*Prev_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt1, x, y_load[!,"y1"], line=:scatter, label="data")
    
    plt2 = Plots.plot(title="Antibody measurements")
    Plots.plot!(plt2, x, Infc_mean[1,:], label="simulation mean")
    Plots.plot!(plt2, x, Infc_mean[1,:]-1.96*Infc_std[1,:], fillrange=Infc_mean[1,:]+1.96*Infc_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt2, x, y_load[!,"y2"], line=:scatter, label="data")

    return plt1, plt2
end


# resimulation error for chains obtained with the real data
function check_real_chains(SDE_problem, chain, tobs, real_data; burn_in::Integer=0)
    burnin = burn_in+1
    nsim = 500
    
    Prev = Matrix{Float64}(undef, nsim, length(tobs))
    Infc = Matrix{Float64}(undef, nsim, length(tobs))
    
    # Handwritten like Ensemble problem
    for i in 1:1:nsim
        idx_iter = burn_in+rand(axes(chain.value[burnin:end,:,:], 1))
        idx_chain = rand(axes(chain.value, 3))
        p = chain.value[idx_iter, :, idx_chain][2:6] #first is the lp value
        u0_new = copy(SDE_problem.u0)
        parameter = copy(SDE_problem.p)
        parameter[1] = p[2] 
        parameter[2] = p[3]
        parameter[3] = p[1]
        parameter[4] = p[5]
        parameter[5] = p[4]
        
        SDE_problem_new = remake(SDE_problem; u0=u0_new, p=parameter)
        # solver settings
        solve_alg = PosEM()
        solve_kwargs = (dt=1e-2,)
        nothing 
    
        sol = solve(SDE_problem_new, solve_alg; solve_kwargs...)
        sim_data = [rand(fobs(sol(t), nothing, t, scaling=parameter[5])) for t in tobs]

        prevalence = []
        infections = []
        for j in range(1, length(tobs))
            push!(prevalence, sim_data[j][1])
            push!(infections, sim_data[j][2])
        end
        Prev[i,:] = prevalence
        Infc[i,:] = infections
    end
    return mean(Prev; dims=1), mean(Infc; dims=1), std(Prev;dims=1), std(Infc;dims=1)    
end


# visualization of the resimulation error for real data chains
function data_real_check(SDE_problem, chain, tobs, real_data; burn_in::Integer=0)
    Prev_mean, Infc_mean, Prev_std, Infc_std = check_real_chains(SDE_problem, chain, tobs, real_data; burn_in=burn_in)
    x = range(1,length(tobs))
    
    plt1 = Plots.plot(title = "Prevalence")
    Plots.plot!(plt1, x, Prev_mean[1,:], label="simulation mean")
    Plots.plot!(plt1, x, Prev_mean[1,:]-1.96*Prev_std[1,:], fillrange=Prev_mean[1,:]+1.96*Prev_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt1, x, [real_data[j][1] for j in 2:length(tobs)], line=:scatter, label="data") #real_data[1]=missing
    
    plt2 = Plots.plot(title="Antibody measurements")
    Plots.plot!(plt2, x, Infc_mean[1,:], label="simulation mean")
    Plots.plot!(plt2, x, Infc_mean[1,:]-1.96*Infc_std[1,:], fillrange=Infc_mean[1,:]+1.96*Infc_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt2, x, [real_data[j][1] for j in 2:length(tobs)], line=:scatter, label="data")

    return plt1, plt2
end


# evaluate all summaries about the chain with user specified burn-in
function simple_evaluate_synth_chain(SDE_problem, chain, tobs, y_load, parameter_names; burn_in::Integer=10000)
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(summarize(chain[burn_in:end])))
    display(StatsPlots.plot(chain[burn_in:end]))
    plt1, plt2 = data_synth_check(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    display(Plots.plot(plt1, plt2, layout=(1,2), size=(800,350), legend=false))
    ArviZ.plot_autocorr(chain[burn_in:end], max_lag=100)
    return nothing
end


# evaluate all summaries of the chain including burn-in calculation first
function evaluate_synth_chain(SDE_problem, chain, tobs, y_load, parameter_names)
    burn_in = burn_in_from_geweke(chain)
    if burn_in==size(chain)[1]
        print("Geweke test indicates no convergence of the chain")
        return nothing
    end
    print("Burn In: ", burn_in, "\n")
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(summarize(chain[burn_in:end])))
    display(StatsPlots.plot(chain[burn_in:end]))
    plt1, plt2 = data_synth_check(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    display(Plots.plot(plt1, plt2, layout=(1,2), size=(800,350), legend=false))
    ArviZ.plot_autocorr(chain[burn_in:end], max_lag=100)
    return nothing
end


# evaluate all summaries about the chain with user specified burn-in
function simple_evaluate_real_chain(SDE_problem, chain, tobs, y_load, parameter_names; burn_in::Integer=10000)
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(summarize(chain[burn_in:end])))
    display(StatsPlots.plot(chain[burn_in:end]))
    plt1, plt2 = data_real_check(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    display(Plots.plot(plt1, plt2, layout=(1,2), size=(800,350), legend=false))
    ArviZ.plot_autocorr(chain[burn_in:end], max_lag=100)
    return nothing
end


# evaluate all summaries of the chain including burn-in calculation first
function evaluate_real_chain(SDE_problem, chain, tobs, real_data, parameter_names)
    burn_in = burn_in_from_geweke(chain)
    if burn_in==size(chain)[1]
        print("Geweke test indicates no convergence of the chain")
        return nothing
    end
    print("Burn In: ", burn_in, "\n")
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(summarize(chain[burn_in:end])))
    display(StatsPlots.plot(chain[burn_in:end]))
    plt1, plt2 = data_real_check(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    display(Plots.plot(plt1, plt2, layout=(1,2), size=(800,350), legend=false))
    ArviZ.plot_autocorr(chain[burn_in:end], max_lag=100)
    return nothing
end


# some violin plot functions

function my_plot_llh_vs_nparticles(ssm::StateSpaceModel, parameters, obs, nparticles::AbstractVector{<:Integer}; nruns::Integer=50)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    variances = Vector{Float64}(undef, length(nparticles))
    k = 1
    i = 1
    for n in nparticles
        llh = LogLikelihood_NoGradient(ssm, obs; nparticles=n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = llh(parameters)
            k += 1
        end
        variances[i] = round(var(y[((i-1)*nruns+1):i*nruns]); digits=3)
        i += 1
    end
    y = reshape(y, (nruns,length(nparticles)))
    x_names = reshape(string.(nparticles), (1,length(nparticles)))
    labels_list = ["$(nparticles[j])p var: $(variances[j])" for j in 1:length(nparticles)] 
    labels = reshape(labels_list, (1,length(nparticles)))
    return Plots.violin(x_names, y, xlabel="particle number",
        trim=false, labels=labels, legend=:outerright,
        size=(width_px, width_px*2/3))
end

    
colors = palette(:Accent_8) #set some color scheme, because I like lighter colors for the following plot

function my_plot_filter(ssm::StateSpaceModel, parameters, components::AbstractVector{<:Integer}, 
                        tobs; nparticles::Integer, nsigmas::Real=2)
    
    length(components) > length(colors) && error("too many components, not enough colors")
    ncomp = length(components)
    hidden, obs=rand(ssm, parameters, length(tobs));
    bf = BootstrapFilter(ssm, obs)
    pf = SMC(
        bf, parameters, nparticles,
        (filter=RunningSummary(MeanAndVariance(), FullHistory()), ),
    )
    offlinefilter!(pf);
    hist = pf.history_run.filter
    comp_means = Vector{Vector{Float64}}(undef, ncomp)
    comp_variances = Vector{Vector{Float64}}(undef, ncomp)
    for i in 1:ncomp
        mean = [hist[j].mean[components[i]] for j in 1:length(tobs)]
        variance = [hist[j].var[components[i]] for j in 1:length(tobs)]
        comp_means[i] = mean
        comp_variances[i] = variance
    end
    plt = Plots.plot(size=(width_px/2,width_px/3))
    x = 1:length(tobs)
    for i in 1:ncomp        
        plot!(plt, x, comp_means[i], color=colors[i], label="")
        plot!(plt, x, comp_means[i]+nsigmas*sqrt.(comp_variances[i]),
            fillrange=comp_means[i]-nsigmas*sqrt.(comp_variances[i]),
            alpha=0.35, color=colors[i], label="")
        scatter!(plt, x, [hidden[j][components[i]] for j in 1:length(tobs)], color=colors[i], label="$(components[i]) component")
    end
        return plt
end;