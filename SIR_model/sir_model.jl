
# SIR-model settings and parameters
N = 100
u0 = [0.96; 0.04]
tspan = (0.0, 100.0)
p = [0.2; 0.05];

@parameters β γ
@variables t s(t) i(t)
D = Differential(t)

drift = [D(s) ~ -β*s*i,
      D(i) ~ β*s*i-γ*i]

diffusion = [sqrt(β*s*i/N) 0 ;-sqrt(β*s*i/N) sqrt(γ*i/N)]

@named sde = SDESystem(drift, diffusion, t, [s, i], [β, γ])

u0map = [s => u0[1], i => u0[2]] # initial condition
parammap = [β=>p[1], γ=>p[2]]

# define SDe problem
problem = SDEProblem(sde, u0map, tspan, parammap);

# observation model
nobs = 2

function prev(x, p, t) #prevalence
    return x[2]
end

function infc_rel(x, p, t) # antibody measurements
    return 1-x[1]
end

function fobs(x, p, t)
        return SIndependent(
            truncated(Normal(prev(x, p, t), 0.02; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel(x, p, t), 0.02; check_args=false), 0.0, nothing),
            )
end;

tobs = [i*10 for i in range(start=1, step=1, stop=10)]; # measurement every 10th-day

# get data
y_load = CSV.read("./data/sir_model_SparseSynth.csv", DataFrame)

real_data = Vector{Vector{Union{Missing, Float64}}}()

for i in range(1,length(tobs))
    prev_meas = y_load[!, "y1"][Int64(i)]
    infc_meas = y_load[!, "y2"][Int64(i)]
    append!(real_data, [Vector{Union{Missing, Float64}}([prev_meas, infc_meas])])
end

real_data = collect(SVector{2, Union{Missing, Float64}}, real_data)

# augment data with initial observation
if tobs[1] != problem.tspan[1]
   real_data = vcat(missing, real_data)
end;

# initial values
initial_state = problem.u0;

true_params = problem.p


# solver settings
solve_alg = PosEM()
solve_kwargs = (dt=1e-2,)
nothing

# creating the SDEStateSpaceModel
ssm = SDEStateSpaceModel(problem, initial_state, fobs, nobs, tobs, solve_alg; solve_kwargs...);

# define likelihood function
function likelihood(nparticles)
    llh_ssm = LogLikelihood_NoGradient(ssm, real_data;nparticles=nparticles)
    llh = let llh_ssm =llh_ssm, ssm=ssm
        p -> llh_ssm(p)
    end
    return llh
end
