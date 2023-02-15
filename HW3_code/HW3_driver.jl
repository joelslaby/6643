import Pkg.instantiate
instantiate()
using BenchmarkTools: @ballocated
using Random: randperm
using LinearAlgebra: I, norm, UpperTriangular
using CairoMakie
include("HW3_your_code.jl")


#----------------------------------------
# Problem a + b
#----------------------------------------
########################################
# A = randn(20, 20) + 100 * I
# b = randn(20)
# reference_x = A \ b
# unpivoted_LU!(A) 
# substitution!(b, A)
# @assert reference_x ≈ b

# allocated_memory = @ballocated  unpivoted_LU!(A)
# allocated_memory += @ballocated  substitution!(b, A)
# @assert allocated_memory < 450

# #----------------------------------------
# # Problem c
# #----------------------------------------
# size_list = 2 .^ (2 : 10) 
# error = Float64[]
# growth_factor = Float64[]

# function inf_norm(X)
#     return maximum(sum(X, dims=2))
# end

# for m = size_list
#     A = randn(m, m)  + 100 * I
#     b = randn(m)
#     ref_norm = inf_norm(A)
#     reference_x = A \ b
#     unpivoted_LU!(A) 
#     substitution!(b, A)

#     push!(error, norm(reference_x - b)/norm(b))
#     push!(growth_factor, inf_norm(UpperTriangular(A))/ref_norm)
# end

# error_plot = scatter(size_list, error, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Relative Error"))
# save("relative_error.pdf", error_plot)

# growth_factor_plot = scatter(size_list, growth_factor, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Growth Factor"))
# save("growth_factor.pdf", growth_factor_plot)


#----------------------------------------
# Problem d
#----------------------------------------
########################################
# A = (randn(20, 20) + 100 * I)[randperm(20), :]
# b = randn(20)
# reference_x = A \ b
# P = pivoted_LU!(A) 
# substitution!(b, A, P)
# @assert reference_x ≈ b

size_list = 2 .^ (2 : 10) 
growth_factor_pivot = Float64[]
error_pivot = Float64[]

function inf_norm(X)
    return maximum(sum(X, dims=2))
end

for m = size_list
    A = (randn(m, m) + 100 * I)[randperm(m), :]
    b = randn(m)
    ref_norm = inf_norm(A)
    reference_x = A \ b
    P = pivoted_LU!(A) 
    substitution!(b, A, P)

    push!(error_pivot, norm(reference_x - b)/norm(b))
    push!(growth_factor_pivot, inf_norm(UpperTriangular(A))/ref_norm)
end

error_plot = scatter(size_list, error_pivot, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Relative Error"))
save("relative_error_pivot.pdf", error_plot)

growth_factor_plot = scatter(size_list, growth_factor_pivot, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Growth Factor"))
save("growth_factor_pivot.pdf", growth_factor_plot)

#----------------------------------------
# Problem e
#----------------------------------------
# YOUR CODE GOES HERE
