using BenchmarkTools: @ballocated
using LinearAlgebra: I, norm, triu, tril, tr, diagm, diag, qr
using CairoMakie
include("HW6_your_code.jl")


#----------------------------------------
# Problem a 
#----------------------------------------
########################################
m = 20
T = rand(m,m)
T = T'T
traceA = tr(T) 
hessenberg_form!(T)
@assert sum(tril(T,-2)) ≈ 0
@assert tr(T) ≈ traceA
T = randn(5, 5)
#allocated_memory = @ballocated  hessenberg_form!(T)
#@assert allocated_memory == 0
println("Passed part (a) test")


#----------------------------------------
# Problem b 
#----------------------------------------
########################################
m = 20
T = rand(m,m)
T = T'T
hessenberg_form!(T)
Q,R =qr(T)
RQ = R*Q
givens_qr!(T)
@assert abs.(T) ≈ abs.(RQ)
println("Passed part (b) test")

#----------------------------------------
# Problem c
#----------------------------------------
########################################
m = 10
A = rand(m,m)
Q,_ = qr(A)
λ = 3 .^ range(0,m-1)
Σ = diagm(λ)
T = Q'Σ*Q
hessenberg_form!(T)
shift = "wilkinson"
practical_QR_with_shifts!(T,shift)
@assert λ ≈ sort(diag(T))


#----------------------------------------
# Problem d
#----------------------------------------
# YOUR CODE GOES HERE
m = 32
A = rand(m,m)
Q,_ = qr(A)
λ = 3 .^ range(0,m-1)
Σ = diagm(λ)
T = Q'Σ*Q
hessenberg_form!(T)
T_single = copy(T)
T_wilk = copy(T)

T_hist_single = []
T_conv_single = []
T_hist_wilk = []
T_conv_wilk = []

practical_QR_with_shifts_hist!(T_single, T_hist_single, T_conv_single, "single")
practical_QR_with_shifts_hist!(T_wilk, T_hist_wilk, T_conv_wilk, "wilkinson")

max_eigs_single = Float64[]
max_eigs_wilk = Float64[]

for t in T_hist_single
    sort!(t)
    push!(max_eigs_single, t[end])
end
for t in T_hist_wilk
    sort!(t)
    push!(max_eigs_wilk, t[end])
end

max_eig_error_single = abs.(max_eigs_single .- λ[end])./λ[end]
max_eig_error_wilk = abs.(max_eigs_wilk .- λ[end])./λ[end]


error_plot = scatter(1:length(max_eigs_single), max_eig_error_single, label="Single Shift", axis=(yscale=log10, xlabel="iteration, k", ylabel="Normalized Eigenvalue Error"))
scatter!(1:length(max_eig_error_wilk), max_eig_error_wilk, label="Wilkinson Shift")
xlims!(0, 30)
axislegend(position=:rt)
save("compare.pdf", error_plot)