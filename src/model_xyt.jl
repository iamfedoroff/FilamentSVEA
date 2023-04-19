struct ModelXYT{T, TG, TF, TM, G, TFFT, TK, TC} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_x :: G
    guard_y :: G
    guard_t :: G
    # spectral parameters:
    FFTxy :: TFFT
    FFTt :: TFFT
    kx :: TK
    ky :: TK
    w :: TK
    Rdiffx :: TC
    Rdiffy :: TC
    Rdisp :: TC
end

@adapt_structure ModelXYT


function Model(grid::GridXYT, field, medium; zu=1, xguard=0, yguard=0, tguard=0)
    (; Nx, Ny, Nt, xu, yu, tu, dx, dy, dt, x, y, t) = grid
    (; w0, E) = field

    FFTxy = plan_fft!(E, [1,2])
    FFTt = plan_fft!(E, [3])

    kxu = 1 / xu
    kyu = 1 / yu
    wu = 1 / tu
    kx = 2*pi * fftfreq(Nx, 1/dx)
    ky = 2*pi * fftfreq(Ny, 1/dy)
    w = 2*pi * fftfreq(Nt, 1/dt)

    k0 = k_func(medium, w0)
    k2 = k2_func(medium, w0)
    Rdiffx = kxu^2 / (2 * k0) * zu
    Rdiffy = kyu^2 / (2 * k0) * zu
    Rdisp = k2 * zu * wu^2 / 2

    guard_x = guard(x, xguard; shape=:both)
    guard_y = guard(y, yguard; shape=:both)
    guard_t = guard(t, tguard; shape=:both)

    return ModelXYT(
        zu, grid, field, medium, guard_x, guard_y, guard_t,
        FFTxy, FFTt, kx, ky, w, Rdiffx, Rdiffy, Rdisp,
    )
end


function model_step!(model::ModelXYT, z, dz)
    (; field, kx, ky, w, Rdiffx, Rdiffy, Rdisp, FFTxy, FFTt,
       guard_x, guard_y, guard_t) = model
    (; E) = field

    FFTxy * E
    diffraction!(E, kx, ky, Rdiffx, Rdiffy, dz)
    FFTxy \ E

    FFTt * E
    dispersion!(E, w, Rdisp, dz)
    FFTt \ E

    mulvec!(E, guard_x; dim=1)
    mulvec!(E, guard_y; dim=2)
    mulvec!(E, guard_t; dim=3)
    return nothing
end


function model_run!(
    model::ModelXYT; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
)
    model = adapt(arch, model)

    (; zu, grid, field) = model
    (; dx, dy, dt) = grid
    (; E) = field

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf)

    while z <= zmax + dz0
        Imax = maximum(abs2, E)
        Fmax = 0
        nemax = 0
        radx = 0
        rady = 0
        tau = 0
        W = sum(abs2, E) * dx * dy * dt
        @printf("%18.12e %18.12e %18.12e\n", z, Imax, nemax)
        writetxt(outtxt, (z, Imax, Fmax, nemax, radx, rady, tau, W))
        writehdf(outhdf, z)

        z += dz0

        model_step!(model, z, dz0)
    end
    return nothing
end
