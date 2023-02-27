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

tspan = (0., 1000. )

SDE_problem = SDEProblem(rs_sde, [], tspan, []);


# real observation function from petab-files
nobs = 2

function prev_comb(x, p, t)
    nom = (x[3]+x[4]+x[6]+x[8]+x[9])
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return nom/nsum
end

function infc_rel_nat(x, p, t; scaling::Real=2.33)
    nom = x[5]+x[6]+x[10]
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return scaling*nom/nsum
end

function fobs(x, p, t; scaling::Real=2.33)
    
    if t == 242
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05086166751331457; check_args=false), 0.0, nothing),
        )
    end
    if t == 262
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.019903244690974285; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
        )
    end
    if t == 273
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.049501502894785186; check_args=false), 0.0, nothing),
        )
    end
    if t == 304
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.027361836386980545; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.048944430985639616; check_args=false), 0.0, nothing),
        )
    end
    if t == 308
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05399166006542568; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
        )
    end
    if t == 328
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.09578163507252513; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
        )
    end
    if t == 333
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.06107833946305408; check_args=false), 0.0, nothing),
        )
    end
    if t == 343
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.03940999395589195; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
        )
    end
    if t == 353
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.0452031209168885; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
        )
    end
    if t == 363
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.0695389424489959; check_args=false), 0.0, nothing),
        )
    end
    if t == 393
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.07212313609837218; check_args=false), 0.0, nothing),
        )
    end
    return SIndependent(
        truncated(Normal(prev_comb(x, p, t), 0.1; check_args=false), 0.0, nothing),
        truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.1; check_args=false), 0.0, nothing),
    )
end

# measurement times from the petab files
tobs = [262, 308, 304, 328, 343, 353, 242, 273, 304, 333, 363, 393]
tobs = sort(tobs)
tobs = unique(tobs);

 # data extracted from petab file
data = CSV.read("./petab_models/petab_virus_variant_model/covid_ethiopia_seir_variant_measurements.tsv", DataFrame)

# filter for data from jimma
jimma(condition::String3) = condition == "c_J"
data = filter(:simulationConditionId => jimma, data)

real_data = Vector{Vector{Union{Missing, Float64}}}()

for ref_t in tobs
    obs_time(time::Int64) = time == ref_t
    data_point = filter(:time => obs_time, data)
    
    if any(occursin.("prev_comb", data_point.observableId))
        prev_meas = filter(x -> any(occursin.(["prev_comb"], x.observableId)), data_point).measurement[1]
    else
        prev_meas = missing
    end
        
    if any(occursin.("infc_rel_nat", data_point.observableId))
        infc_meas = filter(x -> any(occursin.(["infc_rel_nat"], x.observableId)), data_point).measurement[1]
    else
        infc_meas = missing
    end

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
        Uniform(1.0, 10.0),
    )   


# initial values
initial_state = SDE_problem.u0;

true_params = SDE_problem.p[1:4]
true_par = 1


# callbacks for injection of variant
tevent = Float64(170.0)
jumpsize = Float64(1.0)

make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
cb = make_event_cb(tevent, jumpsize)

# solver settings
solve_alg = PosEM()
solve_kwargs = (dt=1e-2, callback=cb, tstops=[tevent])
nothing

# creating the SDEStateSpaceModel
ssm = SDEStateSpaceModel(SDE_problem, initial_state, (fobs, (scaling=2.33,)), nobs, tobs, solve_alg; solve_kwargs...);

# creating the likelihood function

function likelihood(nparticles)
    llh_ssm = LogLikelihood_NoGradient(ssm, real_data; nparticles=nparticles)
    llh = let llh_ssm = llh_ssm, ssm = ssm
        p -> begin
            tevent = Float64(p[4])
            jumpsize = Float64(1.0)
            ssm.solve_kwargs = (dt = 0.01, callback = make_event_cb(tevent, jumpsize), tstops = [tevent]) # NB must be of the same type (order included) as the template passed to SDEStateSpaceModel
            ssm.fobs_kwargs = (scaling=Float64(p[end]),)
            
            all(≥(0) ,p[begin:end-2]) || return -Inf64 # negative rates are not possible
            p_full = copy(ssm.sprob.p)
            p_full[1:4] = p[1:4]
            return llh_ssm(p_full)
        end
    end
    return llh
end