export SSD_LDSImplem, pykalman_LDSImplem, Dynamax_LDSImplem, SSA_LDSImplem, build_model, run_benchmark, recover_states
using Accessors




global recover_iters = 10
global bench_repeats = 5




struct SSD_LDSImplem <: Implementation end
Base.string(::SSD_LDSImplem) = "StateSpaceDynamics.jl"

function build_model(::SSD_LDSImplem, instance::LDSInstance, params::LDSParams)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance
    (; A, Q, x0, P0, C, R)  = params

    # Create the model
    state_model = GaussianStateModel(
        A = A,
        Q = Q,
        x0 = x0,
        P0 = P0,
        )

    obs_model = GaussianObservationModel(
        C = C,
        R = R,
        )

    glds = LinearDynamicalSystem(;
            state_model=state_model,
            obs_model=obs_model,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            fit_bool=fill(true, 6))

    return glds
end

"""
Build pykalman model.
"""

struct pykalman_LDSImplem <: Implementation end
Base.string(::pykalman_LDSImplem) = "pykalman"

function build_model(::pykalman_LDSImplem, instance::LDSInstance, params::LDSParams)
    pykalman = pyimport("pykalman")
    numpy = pyimport("numpy")

    (; latent_dim, obs_dim) = instance
    (; A, Q, x0, P0, C, R) = params

    kf = pykalman.KalmanFilter(
        n_dim_state=latent_dim,
        n_dim_obs=obs_dim,
        transition_matrices=numpy.array(A),
        transition_covariance=numpy.array(Q),
        initial_state_mean=numpy.array(x0),
        initial_state_covariance=numpy.array(P0),
        observation_matrices=numpy.array(C),
        observation_covariance=numpy.array(R),
        em_vars=["transition_matrices", "transition_covariance", "initial_state_mean", "initial_state_covariance", "observation_matrices", "observation_covariance"],
    )

    return kf
end

"""
Build Dynamax model.
"""

struct Dynamax_LDSImplem <: Implementation end
Base.string(::Dynamax_LDSImplem) = "Dynamax"

function build_model(::Dynamax_LDSImplem, instance::LDSInstance, params::LDSParams)
    dynamax = pyimport("dynamax")
    jr = pyimport("jax.random")
    dlds = pyimport("dynamax.linear_gaussian_ssm")
    np = pyimport("numpy")

    (; latent_dim, obs_dim) = instance
    (; A, Q, x0, P0, C, R) = params

    # Convert everything to NumPy arrays
    A_np = np.array(A)
    Q_np = np.array(Q)
    x0_np = np.array(x0)
    P0_np = np.array(P0)
    C_np = np.array(C)
    R_np = np.array(R)

    # Create the Dynamax model
    lds = dlds.LinearGaussianSSM(latent_dim, obs_dim)
    key = jr.PRNGKey(0)
    dyn_params, props = lds.initialize(
        key=key,
        dynamics_weights=A_np,
        dynamics_covariance=Q_np,
        initial_mean=x0_np,
        initial_covariance=P0_np,
        emission_weights=C_np,
        emission_covariance=R_np,
    )

    return (dyn_params, props, lds)
end



"""
Build SSA model.
"""

struct SSA_LDSImplem <: Implementation end
Base.string(::SSA_LDSImplem) = "SSA"
function build_model(::SSA_LDSImplem, instance::LDSInstance, params::LDSParams)
   
    (; latent_dim, obs_dim, num_trials, seq_length) = instance
    (; A, Q, x0, P0, C, R) = params

    print("num trials: $num_trials, seq length: $seq_length \n")

    S = core_struct(
        prm=param_struct(

            save_path = ".",
            load_path = ".",

            seed = 99,
            model_name = "test",
            changelog = "run test",
            load_name = "example",
            pt_list = 1:1, # always has to be range

            max_iter_em = 100,
            test_iter = 1,
            early_stop = false,

            x_dim_fast = latent_dim:latent_dim,
            ), 
        dat=data_struct(
            pt = 1, # pt default
            x_dim = latent_dim , # x_dim default
            y_dim = obs_dim,
            n_steps = seq_length,
            n_train = num_trials,
            n_test = num_trials,
            u_dim = 1,
            u0_dim = 1,
            ),

        res=results_struct(),

        est=estimates_struct(
    
        ),

        mdl=set_model(;A=A, B=zeros(latent_dim, 1), Q=tol_PD(Q), C=C, R=tol_PD(R), B0=zeros(latent_dim, 1), P0=tol_PD(P0)),

        fcn=function_struct{core_struct}(),

    );
    
    return S
end




function run_benchmark(::SSD_LDSImplem, model::LinearDynamicalSystem, Y::AbstractArray)
    # Run 1 EM iteration to compile
    StateSpaceDynamics.fit!(deepcopy(model), Y, max_iter=1, tol=1e-50)

    # run Benchmark
    bench = @benchmark begin
        StateSpaceDynamics.fit!($model, $Y; max_iter=100, tol=1e-50)
    end samples=bench_repeats

    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)

end

function run_benchmark(::pykalman_LDSImplem, model::Any, Y::AbstractArray)
    # No need to run an initial iteration, no JIT
    Y = Y[:, :, 1]

    # run benchmark
    np = pyimport("numpy")
    Y_np = np.array(Y).transpose()
    bench = @benchmark begin
        $model.em($Y_np, n_iter=100)
    end samples=bench_repeats
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

function run_benchmark(::Dynamax_LDSImplem, model::Tuple, Y::AbstractArray)
    # Run an initial iteration to compile
    dynamax = pyimport("dynamax")
    np = pyimport("numpy")
    jax = pyimport("jax")

    Y = Y[:, :, 1]
    Y_np = np.array(Y).transpose()

    (params, props, lds) = model

    fit_em_ = jax.jit(lds.fit_em, static_argnames=("num_iters",))

    bench = @benchmark begin
        $fit_em_($params,
            $props,
            $Y_np,
            num_iters=100)
    end samples=bench_repeats
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end



function run_benchmark(::SSA_LDSImplem, S::Any, Y::AbstractArray)
    
    @reset S.dat.y_train = deepcopy(Y)
    @reset S.dat.u_train = ones(1, S.dat.n_steps, S.dat.n_train)
    @reset S.dat.u0_train = ones(1, S.dat.n_train)

    @reset S.dat.y_test = deepcopy(Y)
    @reset S.dat.u_test = ones(1, S.dat.n_steps, S.dat.n_train)
    @reset S.dat.u0_test = ones(1, S.dat.n_train)


    @reset S.est = deepcopy(set_estimates(S));

    @reset S.prm.max_iter_em = 1
    SSA_EM!(S)
    @reset S.prm.max_iter_em = 100


    bench = @benchmark begin
        SSA_EM!($S)
    end samples=bench_repeats
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end



function recover_states(::SSD_LDSImplem, model::LinearDynamicalSystem, Y::AbstractArray, x::AbstractArray)
   
        # fit
        StateSpaceDynamics.fit!(model, Y, max_iter=recover_iters, tol=1e-50)
        # smooth
        xhat, _ = smooth(model, Y)

        x_long = reshape(x, size(x, 1), size(x, 2) * size(x, 3))'
        xhat_long = reshape(xhat, size(xhat, 1), size(xhat, 2) * size(xhat, 3))'
        b_x = xhat_long \ x_long

        # compute R2 
        r2          = 1 .- sum((x_long .- xhat_long).^2) / sum((x_long .- mean(x_long, dims=1)).^2)
        r2_aligned  = 1 .- sum((x_long .- xhat_long*b_x).^2) / sum((x_long .- mean(x_long, dims=1)).^2)
        println("R2: ", r2)
        println("R2 (aligned): ", r2_aligned)

        return (R2=r2, R2aligned=r2_aligned)

end


function recover_states(::SSA_LDSImplem, S::Any, Y::AbstractArray, x::AbstractArray)
   
    @reset S.dat.y_train = deepcopy(Y)
    @reset S.dat.u_train = ones(1, S.dat.n_steps, S.dat.n_train)
    @reset S.dat.u0_train = ones(1, S.dat.n_train)

    @reset S.dat.y_test = deepcopy(Y)
    @reset S.dat.u_test = ones(1, S.dat.n_steps, S.dat.n_train)
    @reset S.dat.u0_test = ones(1, S.dat.n_train)

    @reset S.est = deepcopy(set_estimates(S));

    @reset S.prm.max_iter_em = recover_iters
    
    # fit
    S = SSA_EM!(S)
    # smooth
    P = posterior_mean(S, S.dat.y_train, S.dat.y_train, S.dat.u_train, S.dat.u0_train)
    xhat = P.smooth_mean


    x_long = reshape(x, size(x, 1), size(x, 2) * size(x, 3))'
    xhat_long = reshape(xhat, size(xhat, 1), size(xhat, 2) * size(xhat, 3))'
    b_x = xhat_long \ x_long

    # compute R2 
    r2          = 1 .- sum((x_long .- xhat_long).^2) / sum((x_long .- mean(x_long, dims=1)).^2)
    r2_aligned  = 1 .- sum((x_long .- xhat_long*b_x).^2) / sum((x_long .- mean(x_long, dims=1)).^2)

    println("R2: ", r2)
    println("R2 (aligned): ", r2_aligned)

    return (R2=r2, R2aligned=r2_aligned)

end 





function SSA_EM!(S)

    # main EM loop ===================================================================
    for em_iter = 1:S.prm.max_iter_em

        # ==== E-STEP ================================================================
        @inline StateSpaceAnalysis.ESTEP!(S);

        # ==== M-STEP ================================================================
        @reset S.mdl = deepcopy(StateSpaceAnalysis.MSTEP(S));

        # ==== TOTAL LOGLIK ==========================================================
        StateSpaceAnalysis.total_loglik!(S);
        
        # ==== TEST LOGLIK ==========================================================
        # @reset S.est = deepcopy(set_estimates(S));
        # StateSpaceAnalysis.test_loglik!(S);
        # push!(S.res.test_R2_proj, ll_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    

    end

    return S

end

