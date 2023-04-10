#Part A
# creates Identity matrix of correct size
m = 8
T = 1.0 * I(m)

# populates diagonal with correct eigenvalues
for i=1:size(T, 1)
    T[m-i+1,m-i+1] = 3^(i-1)
end

# generates random eigenvector matrix
L = randn(m, m)

# creates matrix A from eigendecomposition
A = L * T * inv(L)


#Part B
# Initializes Qk and Rk in QR factorization. Rk has n more dimensions to save the old iterations of Rk matrices
n = 5
Qk = randn(m, m)
Rk = zeros(n, m, m)

# run for n iterations (5)
for k=1:n
    # Update Qk
    Qk = A * Qk

    # Calculate QR factorization 
    Qk, Rk[k,:,:] = qr(Qk)

    # Grab eigenvalues after each iteration
    d = round.(diag(Rk[k,:,:]), sigdigits=4)
    println("itr $k: diag $d")
end


#Part C
# Calculate eigenvalue estimate error
p_max = 3
eig_error = zeros(p_max, n)
for p=1:p_max
    for k=1:n
        eig_error[p, k] = abs((diag(Rk[k,:,:])[p] - T[p,p]) / T[p, p])
    end
end

# Calculate theoretical convergence
conv_theory = zeros(m, n)
for j = 1:m
    for i = 1:n
        if j==m
            conv_theory[j, i] = (T[j, j] / T[j-1, j-1]) ^ i
        else
            conv_theory[j, i] = (T[j+1, j+1] / T[j, j]) ^ i
        end
    end
end

# Plot Error
k_list = 1:n
error_plot = scatter(k_list, eig_error[1,:], label="λ_1 = $(T[1,1])", axis=(yscale=log10, xlabel="iteration, k", ylabel="Normalized Eigenvalue Error"))
for p=2:p_max
    scatter!(k_list, eig_error[p,:], label="λ_$p = $(T[p,p])")
end
scatter!(k_list, conv_theory[1,:], label="Theory")
axislegend(position=:rt)
save("eig_error.pdf", error_plot)

#Part D
# Calculate the norm of the submatrix
p = 4
mat_error = zeros(n,1)
for k = 1:n
    mat_error[k] = norm(Ak[k, (p+1):m, 1:p])
end

# Plot the error and convergence
mat_error_plot = scatter(1:n, mat_error[:], label="Experimental", axis=(yscale=log10, xlabel="iteration, k", ylabel="Norm"))
scatter!(k_list, conv_theory[1,:], label="Theory")
axislegend(position=:rt)
save("mat_error.pdf", mat_error_plot)
