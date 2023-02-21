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
A = randn(20, 20) + 100 * I
b = randn(20)
reference_x = A \ b
unpivoted_LU!(A) 
substitution!(b, A)
@assert reference_x ≈ b

allocated_memory = @ballocated  unpivoted_LU!(A)
allocated_memory += @ballocated  substitution!(b, A)
@assert allocated_memory < 450

# #----------------------------------------
# # Problem c
# #----------------------------------------
size_list = 2 .^ (2 : 10) 
error = Float64[]
growth_factor = Float64[]

function inf_norm(X)
    return maximum(sum(X, dims=2))
end

function grab_U(U)
    up = copy(U)
    for i=1:size(up, 1)
        for j = 1:i-1
            up[i,j] = 0
        end
    end
    return up
end

for m = size_list
    A = randn(m, m)  + 100 * I
    b = randn(m)
    ref_norm = inf_norm(A)
    reference_x = A \ b
    unpivoted_LU!(A) 
    substitution!(b, A)

    push!(error, norm(reference_x - b)/norm(b))
    push!(growth_factor, inf_norm(grab_U(A))/ref_norm)
end

error_plot = scatter(size_list, error, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Relative Error"))
save("relative_error.pdf", error_plot)

growth_factor_plot = scatter(size_list, growth_factor, axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Growth Factor"))
save("growth_factor.pdf", growth_factor_plot)


#----------------------------------------
# Problem d
#----------------------------------------
########################################
A = (randn(20, 20) + 100 * I)[randperm(20), :]
b = randn(20)
reference_x = A \ b
P = pivoted_LU!(A) 
substitution!(b, A, P)
@assert reference_x ≈ b

size_list = 2 .^ (2 : 10) 
growth_factor_pivot = Float64[]
error_pivot = Float64[]
error_ref = Float64[]

for m = size_list
    A = (randn(m, m))
    b = randn(m)
    save_A = copy(A)
    save_b = copy(b)
    ref_norm = inf_norm(A)
    reference_x = A \ b
    P = pivoted_LU!(A) 
    substitution!(b, A, P)

    push!(error_pivot, norm(save_A*b - save_b)/norm(save_b))
    push!(error_ref, norm(save_A*reference_x - save_b)/norm(save_b))
    push!(growth_factor_pivot, inf_norm(grab_U(A))/ref_norm)
end

error_plot_pivot = scatter(size_list, error_ref, label="Reference", axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Relative Error"))
scatter!(size_list, error_pivot, label="Pivoted")
axislegend(position=:lt)
save("relative_error_pivot.pdf", error_plot_pivot)

growth_factor_plot_pivot = scatter(size_list, growth_factor_pivot, label="Pivoted", axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Growth Factor"))
scatter!(size_list, growth_factor, label="Unpivoted")
axislegend(position=:lt)
save("growth_factor_compare.pdf", growth_factor_plot_pivot)

#----------------------------------------
# Problem e
#----------------------------------------
size_list = 2 .^ (2 : 10) 
growth_factor_pivot = Float64[]
error_pivot = Float64[]
error_ref = Float64[]

for m = size_list
    A = growth_matrix(m)
    display(A)
    b = randn(m)
    save_A = copy(A)
    save_b = copy(b)
    ref_norm = inf_norm(A)
    reference_x = A \ b
    P = pivoted_LU!(A) 
    substitution!(b, A, P)

    push!(error_pivot, norm(save_A*b - save_b)/norm(save_b))
    push!(error_ref, norm(save_A*reference_x - save_b)/norm(save_b))
end

error_plot_growth = scatter(size_list, error_ref, label="Reference", axis=(yscale=log10, xscale=log2, xlabel="Matrix Size, m", ylabel="Relative Error"))
scatter!(size_list, error_pivot, label="Pivoted")
axislegend(position=:lt)
save("growth_matrix_rel_error.pdf", error_plot_growth)