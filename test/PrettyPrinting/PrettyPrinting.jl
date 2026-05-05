function test_pretty_printing()
    # Set up IO buffer for printing

    io1 = IOBuffer()
    objs = []

    # Filter Smooth object
    fs = StateSpaceDynamics.FilterSmooth(
        rand(2, 2),      # x_smooth (2D)
        rand(2, 2, 2),   # p_smooth (3D)
        rand(2, 2, 2),   # p_smooth_tt1 (3D)
        rand(2, 2),      # E_z (2D)
        rand(2, 2, 2),   # E_zz (3D)
        rand(2, 2, 2),   # E_zz_prev (3D)
        0.0,              # entropy (scalar)
    )
    push!(objs, fs)

    @test println(io1, fs) === nothing

    # Gaussian State Model

    gsm1 = GaussianStateModel(rand(5, 5), rand(5, 5), rand(5), rand(5), rand(5, 5))
    gsm2 = GaussianStateModel(rand(2, 2), rand(2, 2), rand(2), rand(2), rand(2, 2))
    push!(objs, gsm1, gsm2)

    @test println(io1, gsm1) === nothing
    @test println(io1, gsm2) === nothing

    # Gaussian Observation Model

    gom1 = GaussianObservationModel(rand(5, 5), rand(5, 5), rand(5))
    gom2 = GaussianObservationModel(rand(3, 3), rand(3, 3), rand(3))
    push!(objs, gom1, gom2)

    @test println(io1, gom1) === nothing
    @test println(io1, gom2) === nothing

    # Poisson Observation Model

    pom1 = PoissonObservationModel(rand(5, 5), rand(5))
    pom2 = PoissonObservationModel(rand(2, 2), rand(2))
    push!(objs, pom1, pom2)

    @test println(io1, pom1) === nothing
    @test println(io1, pom2) === nothing

    # Linear Dynamical System

    lds1 = LinearDynamicalSystem(
        gsm1, gom1; fit_bool=[true, true, true, true, true, true], kalman_filter=false
    )
    lds2 = LinearDynamicalSystem(
        gsm2, gom2; fit_bool=[true, true, true, true, true, true], kalman_filter=false
    )

    push!(objs, lds1, lds2)

    @test println(io1, lds1) === nothing
    @test println(io1, lds2) === nothing

    # Probabilistic PCA

    ppca = ProbabilisticPCA(rand(5, 5), 0.5, rand(5))
    push!(objs, ppca)

    @test println(io1, ppca) === nothing

    # Switching Linear Dynamical System (SLDS)
    slds1 = SLDS(rand(5, 5), rand(5), [lds1, lds1, lds1, lds1, lds1])
    slds2 = SLDS(rand(2, 2), rand(2), [lds2, lds2])
    push!(objs, slds1, slds2)

    @test println(io1, slds1) === nothing
    @test println(io1, slds2) === nothing

    # testing `print_full`
    io2 = IOBuffer()

    for obj in objs
        @test print_full(io2, obj) === nothing
    end

    # last tests
    seekstart(io1)
    seekstart(io2)

    str1 = read(io1, String)
    str2 = read(io2, String)

    @test str1 isa String
    @test length(str1) <= length(str2)

    return nothing
end
