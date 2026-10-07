#=============================================================================
causal_benchmark.jl — what the `:causal` mode costs, against `:lqr`.

Times one EM iteration of each mode on the same data (drawn from a causal
controller), split into its E-step (smoother + statistics), ELBO and M-step,
with the memory each allocates, after a warm-up that excludes compilation. The
two modes fit the same plant and cost; `:lqr` carries its terminal factor with
the conditional score (the package default), `:causal` the same terminal cost
in its sweep.

  julia --project=<env with StateSpaceDynamics> -t 1 docs/dev/lqr/causal_benchmark.jl [--quick]

Single-threaded on purpose: the per-iteration cost is what scales with the
data, and threading changes the E-step only.
=============================================================================#

using StateSpaceDynamics, LinearAlgebra, Random, Printf, Statistics
const SSD = StateSpaceDynamics
BLAS.set_num_threads(1)
const QUICK = "--quick" in ARGS

"A stable, invertible plant with a well-conditioned controller, at plant dimension `n`."
function problem(n; seed=1)
    rng = MersenneTwister(seed)
    A = Matrix(0.97I, n, n) .+ 0.04 .* randn(rng, n, n) ./ sqrt(n)
    B = randn(rng, n, n) ./ sqrt(n)
    S = 0.05 .* (B * B') .+ 0.01I
    Q = Matrix(1.0I, n, n)
    C = randn(rng, 3n, n) ./ sqrt(n)
    return A, Matrix(S), Q, C
end

function causal_lds(n, emission)
    A, S, Q, C = problem(n)
    d = 2n
    sm = causal_state_model(
        A, S, Q, Matrix(0.01I, n, n), Matrix(0.5I, n, n);
        terminal_cost=true, x0=vcat(ones(n), zeros(n)), P0=Matrix(0.1I, d, d),
    )
    Cz = hcat(C, zeros(3n, n))
    om = emission === :gaussian ?
         GaussianObservationModel(Cz, Matrix(0.05I, 3n, 3n), zeros(3n)) :
         PoissonObservationModel(0.3 .* Cz, fill(0.5, 3n))
    return LinearDynamicalSystem(sm, om)
end

function lqr_lds(n, emission)
    A, S, Q, C = problem(n)
    d = 2n
    sm = LQRStateModel(
        A, S, [Q, copy(Q)], Matrix(0.05I, d, d);
        schedule=cost_schedule(400; terminal=true), terminal=true, terminal_regime=2,
        Σf=Matrix(0.01I, n, n), x0=vcat(ones(n), zeros(n)), P0=Matrix(0.1I, d, d),
    )
    Cz = hcat(C, zeros(3n, n))
    om = emission === :gaussian ?
         GaussianObservationModel(Cz, Matrix(0.05I, 3n, 3n), zeros(3n)) :
         PoissonObservationModel(0.3 .* Cz, fill(0.5, 3n))
    return LinearDynamicalSystem(sm, om)
end

"Median seconds and MB of one EM iteration's pieces, after `warm` iterations."
function iteration_cost(lds, y; reps=QUICK ? 2 : 4)
    T = Float64
    quad = lds.obs_model isa GaussianObservationModel
    fit!(lds, y; max_iter=2, progress=false)            # compile everything
    data = SSD.Data(lds, y)
    SSD._prepare_lqr!(lds, data)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    hs = SSD._initialize_td_sufficient_statistics(T, lds, data.tsteps)
    SSD._td_init_const_blocks!(pool[1], lds, data)
    estep() = quad ? SSD.estep!(lds, hs, tfs, data, pool) : SSD.estep!(lds, hs, tfs, data, pool)
    function elbo_()
        if quad
            ent = sum(fs.entropy for fs in tfs.FilterSmooths)
            return SSD.elbo!(lds, hs, pool[1], ent)
        end
        return SSD.elbo!(lds, hs, tfs, data, pool)
    end
    mstep() = quad ? SSD.mstep!(lds, hs, pool[1]) : SSD.mstep!(lds, hs, tfs, data, pool)
    te, tl, tm, ae, am = Float64[], Float64[], Float64[], Float64[], Float64[]
    for _ in 1:reps
        push!(ae, @allocated(estep()) / 2^20)
        push!(te, @elapsed estep())
        push!(tl, @elapsed elbo_())
        push!(am, @allocated(mstep()) / 2^20)
        estep()
        push!(tm, @elapsed mstep())
    end
    return (e=median(te), l=median(tl), m=median(tm), ae=median(ae), am=median(am))
end

function row(label, n, lengths; emission=:gaussian)
    truth = causal_lds(n, emission)
    _, y = rand(MersenneTwister(2), truth, lengths)
    out = Dict{Symbol,Any}()
    for (mode, mk) in ((:lqr, lqr_lds), (:causal, causal_lds))
        lds = mk(n, emission)
        out[mode] = iteration_cost(lds, y)
    end
    for mode in (:lqr, :causal)
        c = out[mode]
        @printf("%-26s %-7s %8.1f %8.2f %8.1f %9.1f %8.1f %8.1f\n", label, mode,
                1e3 * c.e, 1e3 * c.l, 1e3 * c.m, 1e3 * (c.e + c.l + c.m), c.ae, c.am)
    end
    flush(stdout)
end

println("one EM iteration, median over repetitions, single thread")
@printf("%-26s %-7s %8s %8s %8s %9s %8s %8s\n", "case", "mode", "E (ms)", "ELBO", "M (ms)",
        "total", "E (MB)", "M (MB)")
row("n=2  N=100  T=30", 2, fill(30, 100))
row("n=2  N=1000 T=30", 2, fill(30, QUICK ? 300 : 1000))
row("n=2  N=100  T=100", 2, fill(100, 100))
row("n=4  N=100  T=30", 4, fill(30, 100))
QUICK || row("n=8  N=100  T=50", 8, fill(50, 100))
row("n=2  N=200  ragged 20:40", 2, rand(MersenneTwister(3), 20:40, 200))
row("n=2  N=100  T=30 Poisson", 2, fill(30, 100); emission=:poisson)
