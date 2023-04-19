struct ModelT{T, TG, TF, TM, G, TFFT, TK, TC} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_t :: G
    # spectral parameters:
    FFT :: TFFT
    w :: TK
    Rdisp :: TC
end

@adapt_structure ModelT


function Model(grid::GridT, field, medium; zu=1, tguard=0)
    (; Nt, tu, dt, t) = grid
    (; w0, E) = field

    FFT = plan_fft!(E)

    wu = 1 / tu
    w = 2*pi * fftfreq(Nt, 1/dt)

    k2 = k2_func(medium, w0)
    Rdisp = k2 * zu * wu^2 / 2

    guard_t = guard(t, tguard; shape=:both)

    return ModelT(zu, grid, field, medium, guard_t, FFT, w, Rdisp)
end


function model_step!(model::ModelT, z, dz)
    (; field, w, Rdisp, FFT, guard_t) = model
    (; E) = field
    FFT * E
    dispersion!(E, w, Rdisp, dz; tdim=1)
    FFT \ E
    mulvec!(E, guard_t; dim=1)
    return nothing
end


function model_run!(
    model::ModelT; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
)
    model = adapt(arch, model)

    (; zu, grid, field) = model
    (; dt) = grid
    (; E) = field

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf)

    while z <= zmax + dz0
        Imax = maximum(abs2, E)
        nemax = 0
        tau = 0
        F = sum(abs2, E) * dt
        @printf("%18.12e %18.12e %18.12e\n", z, Imax, nemax)
        writetxt(outtxt, (z, Imax, nemax, tau, F))
        writehdf(outhdf, z)

        z += dz0

        model_step!(model, z, dz0)
    end
    return nothing
end
