using Test, StateSpaceDynamics, LinearAlgebra, Random
const FDSSD = StateSpaceDynamics

function fixed_d_fixture(;poisson=false,grouped=false,inputs=1)
    labels = ["a","a","b","b"]
    sm = GaussianStateModel(A=Matrix(.8I,2,2),Q=Matrix(.2I,2,2),b=zeros(2),
        x0=zeros(2),P0=Matrix(1.0I,2,2))
    kw = (;C=[.3 .1;-.2 .4;.1 -.2], d=zeros(3),D=fill(.1,3,inputs),
        d_fixed_zero=true,depends_on=grouped ? (C=labels,d=labels,D=labels) : nothing)
    om = poisson ? PoissonObservationModel(;kw...) :
        GaussianObservationModel(;kw...,R=Matrix(.3I,3,3))
    om.CD_prior = MNPrior(M₀=zeros(3,3+inputs),Λ=Matrix(.05I,3+inputs,3+inputs))
    return LinearDynamicalSystem(sm,om)
end

@testset "fixed-zero observation intercept" begin
    rng = MersenneTwister(132)
    for poisson in (false,true),grouped in (false,true)
        lds = fixed_d_fixture(;poisson,grouped)
        C0=copy(lds.obs_model.C)
        y = poisson ? [Float64.(rand(rng,0:2,3,20)) for _ in 1:4] :
            [randn(rng,3,20).+.4 for _ in 1:4]
        uy = [randn(rng,1,20) for _ in 1:4]
        tr = fit!(lds,y;uy,max_iter=4,progress=false)
        @test all(isfinite,tr)
        @test all(iszero,lds.obs_model.d)
        @test lds.obs_model.C != C0
        if grouped
            @test all(v.d_fixed_zero && all(iszero,v.d) for v in lds.obs_model.variants)
        end
        if !poisson
            @test FDSSD._cd_prior_width(lds)==3 # two C columns + one D, excluding d
        end
    end
    # Check the constrained Gaussian solution directly. An unconstrained solve
    # followed by zeroing d gives different C/D for this correlated Gram.
    lds=fixed_d_fixture()
    sw=FDSSD.SmoothWorkspace(Float64,2,3,8;uy_dim=1)
    suf=FDSSD._initialize_td_sufficient_statistics(Float64,lds,[8])
    G=[4. 1 .8 .3;1 3 .4 .2;.8 .4 2 .7;.3 .2 .7 3]
    XY=randn(rng,4,3)
    suf.obs_xx[]=FDSSD.pd_gram(G;name="fixed-d fixture")
    suf.obs_xy.=XY
    FDSSD.update_C_d!(lds,suf,sw)
    free=[1,2,4]
    expected=XY[free,:]'/(G[free,free]+lds.obs_model.CD_prior.Λ[free,free])
    @test hcat(lds.obs_model.C,lds.obs_model.D) ≈ expected
    @test all(iszero,lds.obs_model.d)
    @test FDSSD._obs_prior_logdensity(lds,nothing) ≈
        FDSSD.mn_logprior_term(expected,lds.obs_model.R,
            FDSSD._restrict_mn_prior(lds.obs_model.CD_prior,free))
    @test_throws ArgumentError LinearDynamicalSystem(lds.state_model,
        PoissonObservationModel(C=lds.obs_model.C,d=ones(3),d_fixed_zero=true))
    # Composite members retain separate policies.
    neural=fixed_d_fixture(poisson=true)
    behavior=GaussianObservationModel(C=[.2 .3],R=reshape([.2],1,1),d=[.7])
    composite=LinearDynamicalSystem(neural.state_model,
        CompositeObservationModel((spikes=neural.obs_model,kin=behavior)))
    y=[(spikes=Float64.(rand(rng,0:2,3,15)),kin=randn(rng,1,15).+2) for _ in 1:3]
    # Composite data use one vector per member rather than per-trial tuples.
    obs=(spikes=[v.spikes for v in y],kin=[v.kin for v in y])
    tr=fit!(composite,obs;uy=(spikes=[zeros(1,15) for _ in y],kin=[zeros(0,15) for _ in y]),max_iter=3,progress=false)
    @test all(isfinite,tr)
    @test all(iszero,composite.obs_model.models.spikes.d)
    @test any(!iszero,composite.obs_model.models.kin.d)
end

@testset "fixed intercept in shared and switching emissions" begin
    rng=MersenneTwister(141)
    for poisson in (false,true)
        first=fixed_d_fixture(;poisson)
        second=deepcopy(first)
        second.state_model.A .*= .9
        model=SLDS(A=[.95 .05;.05 .95],πₖ=[.5,.5],LDSs=[first,second])
        y=poisson ? [Float64.(rand(rng,0:2,3,12)) for _ in 1:3] : [randn(rng,3,12) for _ in 1:3]
        uy=[randn(rng,1,12) for _ in 1:3]
        tied=poisson ? [:C,:d,:D] : [:d]
        tr=fit!(model,y;uy,tied_params=tied,max_iter=3,progress=false,smoothing_iters=2,num_samples=1,rng)
        @test all(isfinite,tr)
        @test all(all(iszero,lds.obs_model.d) for lds in model.LDSs)
    end
    # One CD shared across two different R groups exercises the GLS route.
    m=fixed_d_fixture()
    StateSpaceDynamics.set_depends_on!(m.obs_model,(R=["a","a","b","b"],))
    tr=fit!(m,[randn(rng,3,12) for _ in 1:4];uy=[randn(rng,1,12) for _ in 1:4],max_iter=3,progress=false)
    @test all(isfinite,tr)
    @test all(all(iszero,v.d) for v in m.obs_model.variants)
end

@testset "fixed intercept with hidden LQR costate" begin
    rng=MersenneTwister(142)
    for poisson in (false,true)
        state=LQRStateModel([.96 .07;-.05 .93],[.06 .01;.01 .05],
            [Matrix(.2I,2,2)],Matrix(.03I,4,4);terminal=false,
            x0=zeros(4),P0=Matrix(.25I,4,4),observe_costate=false)
        C=hcat(.2randn(rng,3,2),zeros(3,2))
        om=poisson ? PoissonObservationModel(C=C,d=zeros(3),d_fixed_zero=true) :
            GaussianObservationModel(C=C,R=Matrix(.2I,3,3),d=zeros(3),d_fixed_zero=true)
        om.CD_prior=MNPrior(M₀=zeros(3,5),Λ=Matrix(.01I,5,5))
        model=LinearDynamicalSystem(state,om;fit_bool=poisson ?
            [false,false,false,false,true] : [false,false,false,false,true,false])
        y=poisson ? [Float64.(rand(rng,0:2,3,8)) for _ in 1:2] :
            [randn(rng,3,8) for _ in 1:2]
        trace=fit!(model,y;max_iter=2,progress=false)
        @test all(isfinite,trace)
        @test all(iszero,om.d)
        @test all(iszero,om.C[:,3:4])
        if !poisson
            @test FDSSD._cd_prior_width(model)==2
        end
    end
end
