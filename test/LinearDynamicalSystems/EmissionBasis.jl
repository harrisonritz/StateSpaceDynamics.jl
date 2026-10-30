using Test, StateSpaceDynamics, LinearAlgebra, Random
const EBSSD = StateSpaceDynamics

function basis_fixture(rng; p=6, k=2, q=1, basis=nothing)
    sm = GaussianStateModel(;
        A=[0.9 -0.2; 0.2 0.9],
        Q=Matrix(0.1I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
    )
    root = randn(rng, p, p)
    C = randn(rng, p, k)
    basis === nothing || (C = basis * (basis \ C))
    om = GaussianObservationModel(;
        C, R=root * root' / p + I, d=zeros(p), D=zeros(p, q), C_basis=basis
    )
    return LinearDynamicalSystem(sm, om)
end

@testset "emission basis C = C_basis * H" begin
    rng = MersenneTwister(211)
    p, k, q, r = 6, 2, 1, 3
    A = randn(rng, p, r)

    # The M-step against the constrained least squares solved directly:
    # vec([C d D]) = J θ with θ = [vec(H); vec([d D])].
    lds = basis_fixture(rng; basis=A)
    sw = EBSSD.SmoothWorkspace(Float64, k, p, 8; uy_dim=q)
    suf = EBSSD._initialize_td_sufficient_statistics(Float64, lds, [8])
    width = k + 1 + q
    Z = randn(rng, width, 3width)
    G, XY = Z * Z', randn(rng, width, p)
    suf.obs_xx[] = EBSSD.pd_gram(G; name="basis fixture")
    suf.obs_xy .= XY
    EBSSD.update_C_d!(lds, suf, sw)
    Ri = inv(lds.obs_model.R)
    J = cat(
        kron(Matrix(1.0I, k, k), A), Matrix(1.0I, p * (1 + q), p * (1 + q)); dims=(1, 2)
    )
    θ = (J' * kron(G, Ri) * J) \ (J' * vec(Ri * XY'))
    @test hcat(lds.obs_model.C, lds.obs_model.d, lds.obs_model.D) ≈ reshape(J * θ, p, width)

    y = [randn(rng, p, 30) for _ in 1:5]
    uy = [randn(rng, q, 30) for _ in 1:5]

    # A square, invertible basis constrains nothing: the fit is the unconstrained one.
    free = basis_fixture(MersenneTwister(5))
    square = deepcopy(free)
    square.obs_model.C_basis = randn(rng, p, p)
    elbo_free = fit!(free, y; uy, max_iter=6, progress=false)
    elbo_square = fit!(square, y; uy, max_iter=6, progress=false)
    @test collect(elbo_square) ≈ collect(elbo_free)
    @test square.obs_model.C ≈ free.obs_model.C && square.obs_model.R ≈ free.obs_model.R

    # A narrower basis: the ELBO still never decreases and C stays in the span.
    lds = basis_fixture(rng; basis=A)
    elbos = collect(fit!(lds, y; uy, max_iter=25, progress=false))
    @test all(diff(elbos) .>= -1e-8 * abs(last(elbos)))
    C = lds.obs_model.C
    @test norm(C - A * (A \ C)) <= 1e-10 * norm(C)
    @test last(elbos) <
        last(collect(fit!(basis_fixture(rng), y; uy, max_iter=25, progress=false)))

    sm = lds.state_model
    R = Matrix(1.0I, p, p)
    @test_throws ArgumentError LinearDynamicalSystem(
        sm, GaussianObservationModel(; C=randn(rng, p, k), R, d=zeros(p), C_basis=A)
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        sm, GaussianObservationModel(; C=A[:, 1:k], R, d=zeros(p), C_basis=[A A[:, 1]])
    )
    @test_throws EBSSD.DimensionMismatchError LinearDynamicalSystem(
        sm, GaussianObservationModel(; C=A[:, 1:k], R, d=zeros(p), C_basis=A[1:4, :])
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        sm,
        GaussianObservationModel(;
            C=A[:, 1:k],
            R,
            d=zeros(p),
            C_basis=A,
            CD_prior=MNPrior(; M₀=zeros(p, k + 1), Λ=Matrix(0.1I, k + 1, k + 1)),
        ),
    )
end
