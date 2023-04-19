module FilamentSVEA


import Adapt: @adapt_structure, adapt
import CUDA: @cuda, launch_configuration, CuArray, threadIdx, blockIdx,
             blockDim, gridDim
import FFTW: fftfreq, plan_fft!
import Printf: @printf

using FilamentBase
export CPU, GPU, GridT, GridRT, GridXY, GridXYT, Field, Medium,
       refractive_index, k_func, k1_func, k2_func, phase_velocity,
       group_velocity, diffraction_length, dispersion_length, absorption_length,
       chi1_func, chi3_func, critical_power, nonlinearity_length,
       selffocusing_length


export Model, model_run!


abstract type Model end

include("model_t.jl")
include("model_xy.jl")
include("model_xyt.jl")


# ******************************************************************************
function diffraction!(E, kx, ky, Rdiffx, Rdiffy, dz; xdim=1, ydim=2)
    ci = CartesianIndices(E)
    for ici in eachindex(ci)
        ix = ci[ici][xdim]
        iy = ci[ici][ydim]
        E[ici] *= exp(-1im * (Rdiffx * kx[ix]^2 + Rdiffy * ky[iy]^2) * dz)
    end
    return nothing
end


function diffraction!(E::CuArray, kx, ky, Rdiffx, Rdiffy, dz; xdim=1, ydim=2)
    N = length(E)
    @krun N diffraction_kernel(E, kx, ky, Rdiffx, Rdiffy, dz, xdim, ydim)
    return nothing
end
function diffraction_kernel(E, kx, ky, Rdiffx, Rdiffy, dz, xdim, ydim)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    ci = CartesianIndices(E)
    for ici=id:stride:length(ci)
        ix = ci[ici][xdim]
        iy = ci[ici][ydim]
        E[ici] *= exp(-1im * (Rdiffx * kx[ix]^2 + Rdiffy * ky[iy]^2) * dz)
    end
    return nothing
end


# ******************************************************************************
function dispersion!(E, w, Rdisp, dz; tdim=3)
    ci = CartesianIndices(E)
    for ici in eachindex(ci)
        it = ci[ici][tdim]
        E[ici] *= exp(1im * Rdisp * w[it]^2 * dz)
    end
    return nothing
end


function dispersion!(E::CuArray, w, Rdisp, dz; tdim=3)
    N = length(E)
    @krun N dispersion_kernel(E, w, Rdisp, dz, tdim)
    return nothing
end
function dispersion_kernel(E, w, Rdisp, dz, tdim)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    ci = CartesianIndices(E)
    for ici=id:stride:length(ci)
        it = ci[ici][tdim]
        E[ici] *= exp(1im * Rdisp * w[it]^2 * dz)
    end
    return nothing
end


end
