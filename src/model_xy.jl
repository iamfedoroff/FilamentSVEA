struct ModelXY{T, TG, TF, TM, G, TFFT, TK, TC} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_x :: G
    guard_y :: G
    # spectral parameters:
    FFT :: TFFT
    kx :: TK
    ky :: TK
    Rdiffx :: TC
    Rdiffy :: TC
end

@adapt_structure ModelXY


function Model(grid::GridXY, field, medium; zu=1, xguard=0, yguard=0)
    (; Nx, Ny, xu, yu, dx, dy, x, y) = grid
    (; w0, E) = field

    FFT = plan_fft!(E)

    kxu = 1 / xu
    kyu = 1 / yu
    kx = 2*pi * fftfreq(Nx, 1/dx)
    ky = 2*pi * fftfreq(Ny, 1/dy)

    k0 = k_func(medium, w0)
    Rdiffx = kxu^2 / (2 * k0) * zu
    Rdiffy = kyu^2 / (2 * k0) * zu

    guard_x = guard(x, xguard; shape=:both)
    guard_y = guard(y, yguard; shape=:both)

    return ModelXY(
        zu, grid, field, medium, guard_x, guard_y, FFT, kx, ky, Rdiffx, Rdiffy,
    )
end


function model_step!(model::ModelXY, z, dz)
    (; field, kx, ky, Rdiffx, Rdiffy, FFT, guard_x, guard_y) = model
    (; E) = field
    FFT * E
    diffraction!(E, kx, ky, Rdiffx, Rdiffy, dz)
    FFT \ E
    mulvec!(E, guard_x; dim=1)
    mulvec!(E, guard_y; dim=2)
    return nothing
end


function model_run!(
    model::ModelXY; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
)
    model = adapt(arch, model)

    (; zu, grid, field) = model
    (; dx, dy) = grid
    (; E) = field

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf)

    while z <= zmax + dz0
        Imax = maximum(abs2, E)
        radx = 0
        rady = 0
        P = sum(abs2, E) * dx * dy
        @printf("%18.12e %18.12e\n", z, Imax)
        writetxt(outtxt, (z, Imax, radx, rady, P))
        writehdf(outhdf, z)

        z += dz0

        model_step!(model, z, dz0)
    end
    return nothing
end
