import Pkg.instantiate
instantiate()
using BenchmarkTools: @ballocated
using LinearAlgebra: I, norm, istriu, triu, qr
using CairoMakie
include("HW4_your_code.jl")


#----------------------------------------
# Problem a 
#----------------------------------------
########################################
A = randn(30, 20) 
b = randn(30)
Q, R = classical_gram_schmidt(A) 
@assert Q' * Q ≈ I
@assert Q * R ≈ A

#----------------------------------------
# Problem b 
#----------------------------------------
########################################
A = randn(30, 20) 
b = randn(30)
Q, R = modified_gram_schmidt(A) 
@assert Q' * Q ≈ I
@assert Q * R ≈ A

#----------------------------------------
# Problem c
#----------------------------------------
########################################
A = randn(25, 20) 
allocated_memory = @ballocated  householder_QR!(A)
@assert allocated_memory == 0
A = randn(25, 20)
true_R = Matrix(qr(A).R)
householder_QR!(A)
# Checks if the R part of the factorization is correct
@assert vcat(true_R, zeros(5,20)) ≈ triu(A)

#----------------------------------------
# Problem d
#----------------------------------------
########################################
# Testing for memory allocation:
A = randn(25, 20) 
householder_QR!(A)
QR = A
x = randn(20)
b = randn(25)
out_mul = randn(25)
out_div = randn(20)

allocated_memory_mul = @ballocated  householder_QR_mul!(out_mul, x, QR)
allocated_memory_div = @ballocated  householder_QR_div!(out_div, b, QR)
@assert allocated_memory_mul == 0
@assert allocated_memory_div == 0

# Testing for correctness:
A = randn(25, 20) 
x = randn(20)
b = randn(25)
out_mul = randn(25)
out_div = randn(20)
true_mul = A * x 
true_div = A \ b 

householder_QR!(A)
QR = A
householder_QR_mul!(out_mul, x, QR)
householder_QR_div!(out_div, b, QR)

# checks whether the results are approximately correct
@assert true_mul ≈ out_mul
@assert true_div ≈ out_div


#----------------------------------------
# Problem e
#----------------------------------------
# YOUR CODE GOES HERE
size_list = 2 .^ (2 : 12)

# instantiate error arrays
error_true = Float64[]
error_cGS = Float64[]
error_mGS = Float64[]
error_house = Float64[]

function make_high_cond(cond_num, mat_size)
    D = 1.0 * I(mat_size)
    D[1,1] = cond_num
    for i=2:size(D, 1)-1
        D[i, i] = rand()*(cond_num-1) + 1
    end

    L = randn(mat_size, mat_size)
    return L * D * L'
end

for m = size_list
    mat_size = 2^8
    A = make_high_cond(m, mat_size)
    b = randn(mat_size)
    A_original = deepcopy(A)
    b_original = deepcopy(b)
    x_true = A\b

    Q_cGS, R_cGS = classical_gram_schmidt(A)
    Q_mGS, R_mGS = modified_gram_schmidt(A)
    householder_QR!(A)
    QR_house = A

    x_house = zeros(mat_size)

    x_cGS = inv(R_cGS) * Q_cGS' * b
    x_mGS = inv(R_mGS) * Q_mGS' * b
    householder_QR_div!(x_house, b, QR_house)

    push!(error_true, norm(A_original * x_true - b_original)/norm(b_original))
    push!(error_cGS, norm(A_original * x_cGS - b_original)/norm(b_original))
    push!(error_mGS, norm(A_original * x_mGS - b_original)/norm(b_original))
    push!(error_house, norm(A_original * x_house - b_original)/norm(b_original))
end

error_plot = scatter(size_list, error_true, label="Reference Error", axis=(yscale=log10, xscale=log2, xlabel="Condition Number, K", ylabel="Relative Error"))
scatter!(size_list, error_cGS, label="Classic Gram-Schmidt")
scatter!(size_list, error_mGS, label="Modified Gram-Schmidt")
scatter!(size_list, error_house, label="Householder")
axislegend(position=:lt)
save("cond_num_compare.pdf", error_plot)