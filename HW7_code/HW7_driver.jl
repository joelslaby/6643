using BenchmarkTools: @ballocated
using LinearAlgebra: I, norm, triu, tril, tr, diagm, diag, qr, eigvals, istriu
using CairoMakie
include("HW7_your_code.jl")


#----------------------------------------
# Problem a 
#----------------------------------------
########################################
m = 5
A = rand(m,m)
A = A * A' 
H = arnoldi(A, randn(m), m)
@assert istriu(H[2 : end, 1 : (end - 1)])
@assert eigvals(H) ≈ eigvals(A) 

println("Passed part (a) test")


#----------------------------------------
# Problem b 
#----------------------------------------
########################################
m = 5
A = rand(m,m)
A = A * A' 
α, β = lanczos(A, randn(m), m)
H = diagm(-1 => β, 0 => α, 1 => β)
@assert eigvals(H) ≈ eigvals(A) 
println("Passed part (b) test")

#----------------------------------------
# Problem c
#----------------------------------------
########################################

# Compare Arnoldi to Lanczos
m = 8
A = rand(m,m)
Q,_ = qr(A)
λ = 3 .^ range(0,m-1)
Σ = diagm(λ)
T = Q'Σ*Q

max_eigs_A = zeros(m)
max_eigs_L = zeros(m)

# display(T)
for kmax = 1:m
    H_A = arnoldi(T, randn(m), kmax)
    max_eigs_A[kmax] = sort(eigvals(H_A), rev=true)[1]

    a, b = lanczos(T, randn(m), kmax)
    H_L = diagm(-1 => b, 0 => a, 1 => b)
    max_eigs_L[kmax] = sort(eigvals(H_L), rev=true)[1]
end

max_eig_A_error = abs.(max_eigs_A .- 3^(m-1))/3^(m-1)
max_eig_L_error = abs.(max_eigs_L .- 3^(m-1))/3^(m-1)

error_plot = scatter(1:m, max_eig_A_error, label="Arnoldi", axis=(yscale=log10, xlabel="iteration, k", ylabel="Normalized Eigenvalue Error"))
scatter!(1:m, max_eig_L_error, label="Lanczos")
axislegend(position=:rt)
save("compare.pdf", error_plot)


# Create ghost eigenvalues
m = 8
A = rand(m,m)
Q,_ = qr(A)
λ = [1, 2, 3, 4, 5, 6, 7, 1e5]
Σ = diagm(λ)
T = Q'Σ*Q

a, b = lanczos(T, randn(m), m)
H_L = diagm(-1 => b, 0 => a, 1 => b)
display(eigvals(H_L))
display(eigvals(T))
