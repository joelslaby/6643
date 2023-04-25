
#----------------------------------------
# Problem a
#----------------------------------------
# This function takes in a matrix A and returns the 
# kmax × kmax supmatrix of itsupper Hessenberg form 
# The matrix A should be only accessed through 
# (kmax - 1) matrix-vector products
function arnoldi(A, q1, kmax)
    H = zeros(kmax, kmax)
    Q = zeros(size(q1, 1), kmax)

    Q[:, 1] = q1 / norm(q1)
    for k = 1:kmax
        if k > 1
            Q[:, k] = q1/ H[k, k-1]
        end

        q1 = A * Q[:, k]
        for i = 1:k
            H[i, k] = 0
            for j = 1:size(q1, 1)
                H[i, k] += Q[j, i] * q1[j]
            end
            q1 -= H[i, k] * Q[:, i]
        end

        if k < kmax
            H[k+1, k] = norm(q1)
        end
    end
    return H
end

#----------------------------------------
# Problem a
#----------------------------------------
# This function takes in a matrix A and returns the 
# kmax × kmax supmatrix of its tridiagonal form 
# computed by the Lanczos iteration.
# The matrix A should be only accessed through 
# (kmax - 1) matrix-vector products
# The output vectors should be the diagonal (α)
# and the offdiagonal (β) of the tridiagonal matrix
function lanczos(A, q1, kmax)
    T = zeros(kmax, kmax)
    r = copy(q1[:, 1])
    beta = norm(r)
    q1 = zeros(size(A, 1))

    for k = 1:kmax
        q0 = copy(q1)
        q1 = r/beta
        r = A * q1
        alpha = 0
        for j = 1:size(r, 1)
            alpha += q1[j] * r[j]
        end
        T[k, k] = alpha
        if k > 1
            T[k-1, k] = beta
            T[k, k-1] = beta
        end
        r = r - alpha*q1 - beta * q0
        beta = norm(r)
    end
    return diag(T), diag(T,1)
end