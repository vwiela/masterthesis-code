#using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
#using DataFrames
#using CSV
#using Random
#using Distributions
#using SBML
#using SymbolicUtils
#using StaticArrays
#using Catalyst
#using AdvancedMH
#using MCMCChains
#using MCMCChainsStorage
#using HDF5
#
#using Particles
#using ParticlesDE
#using StaticDistributions

# loading the model from sbml
model_name = "covid_ethiopia_seir_variant_model_transformed_real_updated2"
petab_folder = "./petab_models/petab_virus_variant_model"
sbml_file = string(petab_folder, "/", model_name, ".sbml")
SBMLToolkit.checksupport_file(sbml_file)

model = readSBML(sbml_file, doc -> begin
    set_level_and_version(3, 2)(doc)
    convert_simplify_math(doc)
end)

rs_sde = ReactionSystem(model, constraints=nothing)

SDEsys = convert(SDESystem, rs_sde)
# update to real population size of 3.77 million

tspan = (0., 1000. )

SDE_problem = SDEProblem(rs_sde, [], tspan, []);


# real observation function from petab-files
nobs = 2

function prev_comb(x, p, t)
    nom = (x[3]+x[4]+x[6]+x[8]+x[9])
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return nom/nsum
end

function infc_rel(x, p, t)
    nom = x[3]+x[8]+x[9]
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return nom/nsum
end

function fobs(x, p, t)

        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.027361836386980545; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel(x, p, t), 0.048944430985639616; check_args=false), 0.0, nothing),
            )
end


# sparse measurements
tobs = [262, 308, 304, 328, 343, 353, 242, 273, 304, 333, 363, 393]
tobs = sort(tobs)
tobs = unique(tobs);

y_load = CSV.read("./data/real_model_SparseSynth.csv", DataFrame)

real_data = Vector{Vector{Union{Missing, Float64}}}()

for i in range(1,length(tobs))
    prev_meas = y_load[!, "y1"][Int64(i)]
    infc_meas = y_load[!, "y2"][Int64(i)]
    append!(real_data, [Vector{Union{Missing, Float64}}([prev_meas, infc_meas])])
end

real_data = collect(SVector{2, Union{Missing, Float64}}, real_data)

# augment data with initial observation
if tobs[1] != SDE_problem.tspan[1]
   real_data = vcat(missing, real_data)
end;

# priors for initializing the chains; for gamma and kappa priors known from PEtab problem
# for others used uniform priors between bounds
prior_dist = SIndependent(
        truncated(Normal(15.7, 6.7; check_args=false), 0.0, nothing),
        truncated(Normal(5.1, 0.5; check_args=false), 0.0, nothing),
        Uniform(1e-3, 1.0), 
        Uniform(120.0, 360),
    )   

# initial values
initial_state = SDE_problem.u0;

true_params = SDE_problem.p[1:4]

# callbacks for injection of variant
tevent = Float64(170.0)
jumpsize = Float64(5000.0)

make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
cb = make_event_cb(tevent, jumpsize)

# solver settings
solve_alg = PosEM()
solve_kwargs = (dt=1e-2, callback=cb, tstops=[tevent])
nothing

# creating the SDEStateSpaceModel
ssm = SDEStateSpaceModel(SDE_problem, initial_state, fobs, nobs, tobs, solve_alg; solve_kwargs...);

# creating the likelihood function

function likelihood(nparticles)
    llh_ssm = LogLikelihood_NoGradient(ssm, real_data; nparticles=nparticles)
    llh = let llh_ssm = llh_ssm, ssm = ssm
        p -> begin
            tevent = Float64(p[4])
            jumpsize = Float64(1.0)
            ssm.solve_kwargs = (dt = 0.01, callback = make_event_cb(tevent, jumpsize), tstops = [tevent]) # NB must be of the same type (order included) as the template passed to SDEStateSpaceModel
            all(≥(0) ,p[begin:end-3]) || return -Inf64
            p_full = copy(ssm.sprob.p)
            p_full[1:4] = p[1:4]
            return llh_ssm(p_full)
        end
    end
    return llh
end