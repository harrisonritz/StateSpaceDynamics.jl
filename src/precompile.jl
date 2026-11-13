#=============================================================================
A precompile workload for the kernel-prior emissions.

Their fits compile a long chain of methods (the kernel prior and its trees, the
scale-correlation structure, the weights' L-BFGS, the structured E- and
M-steps), and a caller that starts a fresh Julia process per fit pays that
compilation every time. A tiny fit of each kind here caches those methods in
the package image. Dense and sparse correlation factors, a joint tree, both
gauges, inputs and a held-out score are covered.
=============================================================================#

@setup_workload begin
    rng = Random.Xoshiro(1)
    p, r, k, n, q = 6, 4, 2, 12, 1
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    G = randn(rng, r, n)
    v = 0.5 .+ rand(rng, n)
    block = repeat(1:3; inner=4)
    names = ["a", "b", "c"]
    indicator(cols) = sparse(reshape(Float64.(in(cols).(1:n)), :, 1))
    function local_factor(cols)
        F = zeros(n, 2)
        F[cols, :] .= randn(rng, length(cols), 2)
        return sparse(F)
    end
    nuisance = Matrix(1.0I, r, r)
    state() = GaussianStateModel(;
        A=Matrix(0.8I, k, k),
        Q=Matrix(0.1I, k, k),
        b=zeros(k),
        B=0.1 .* randn(rng, k, q),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        AB_prior=MNPrior(; M₀=zeros(k, k + 1 + q), Λ=Matrix(1e-4I, k + 1 + q, k + 1 + q)),
    )
    y = [randn(rng, p, 8) for _ in 1:3]
    u = [randn(rng, q, 8) for _ in 1:3]
    H = randn(rng, r, k)

    @compile_workload begin
        families = [
            CorrelationFamily(
                [indicator(1:8), indicator(9:12)]; type=:network, names=["n1", "n2"]
            ),
            CorrelationFamily(
                [indicator(1:4), indicator(5:8), indicator(9:12)];
                type=:parcel,
                names,
                groups=["n1", "n1", "n2"],
            ),
            CorrelationFamily(
                [local_factor(1:4), local_factor(5:8), local_factor(9:12)];
                type=:geo,
                names,
                groups=["n1", "n1", "n2"],
            ),
        ]
        structure = ScaleCorrelation(
            G, v, block; names, type=:parcel_scale, groups=["n1", "n1", "n2"], families
        )
        pooled = Dict(
            :parcel_scale => :hierarchical,
            :network => :hierarchical,
            :parcel => :hierarchical,
            :geo => :hierarchical,
        )
        nested = Dict(
            :parcel_scale => :nested,
            :network => :hierarchical,
            :parcel => :nested,
            :geo => :nested,
        )
        joint = [(:parcel_scale, :parcel, :geo)]
        for (levels, correlated) in ((pooled, joint), (nested, ()), (nested, joint))
            prior = KernelPrior(
                [Kernel(nuisance; name="I", type=:nuisance)]; structure, levels, correlated
            )
            S = H * H' ./ k
            fit_kernel_weights!(prior, S, k; max_iter=20, rounds=2)
            kernel_weights(prior)
            kernel_evidence(prior, S, k)
        end
        sums = KernelPrior([
            Kernel(G .* sqrt.(v)'; name="depth", type=:cortex, factor=true),
            Kernel(nuisance; name="I", type=:nuisance),
        ])
        for (prior, gauge) in (
            (
                KernelPrior(
                    [Kernel(nuisance; name="I", type=:nuisance)];
                    structure,
                    levels=pooled,
                    correlated=joint,
                ),
                :moment,
            ),
            (sums, :bound),
        )
            om = ConstrainedKernelGaussianObservationModel(;
                basis=Φ,
                prior,
                C=Φ * H,
                R=Matrix(1.0I, p, p),
                gauge,
                weight_iters=5,
                tree_every=2,
            )
            lds = LinearDynamicalSystem(state(), om)
            fit!(lds, y; ux=u, max_iter=3, progress=false, y_test=y, ux_test=u)
            loglikelihood(lds, y; ux=u)
            emission_posterior(om)
            kernel_weights(om)
        end
        plain = ConstrainedGaussianObservationModel(; basis=Φ, H, R=Matrix(1.0I, p, p))
        fit!(LinearDynamicalSystem(state(), plain), y; ux=u, max_iter=2, progress=false)
    end
end
