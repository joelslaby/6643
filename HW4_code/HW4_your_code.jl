#----------------------------------------
# Problem a
#----------------------------------------
# This function takes in a matrix A and returns 
# a reduced QR factorization with factors Q and R.
# It should not modify A
function classical_gram_schmidt(A)
    # YOUR CODE HERE
    m = size(A, 1)
    n = size(A, 2)
    R = zeros(Float64, n, n)
    Q = copy(A)
    for j=1:n
        for i=1:j-1, k=1:m
            R[i, j] += Q[k, i] * Q[k, j]
        end
        for i=1:j-1, k=1:m
            Q[k,j] -= Q[k,i] * R[i,j]
        end
        R[j,j] = norm( Q[:, j] )
        Q[:, j] /= R[j,j]
    end
    return Q, R
end

#----------------------------------------
# Problem b
#----------------------------------------
# This function takes in a matrix A and returns 
# a reduced QR factorization with factors Q and R.
# It should not modify A
function modified_gram_schmidt(A)
    m = size(A, 1)
    n = size(A, 2)
    R = zeros(Float64, n, n)
    Q = copy(A)
    for j=1:n
        for i=1:j-1
            for k=1:m
                R[i,j] += Q[k,i] * Q[k,j]
            end
            for k=1:m
                Q[k,j] -= Q[k,i] * R[i,j]
            end
        end
        R[j,j] = norm(Q[:,j])
        Q[:,j] /= R[j,j]
    end
    return Q, R
end

#----------------------------------------
# Problem c
#----------------------------------------
# This function takes in a matrix A 
# and computes its QR factorization in place,
# using householder reflections.
# It should not allocate any memory.  
function householder_QR!(A)
    # YOUR CODE HERE
    m = size(A, 1)
    n = size(A, 2)
    vA = zeros(n)
    kend = (m > n ? n : m-1)
    for k=1:kend
        beta, v = house(A[k:end, k])
        for j=k:n
            vA[j] = 0
            for i=k:m
                vA[j] += v[i-k+1] * A[i,j]
            end
            vA[j] *= beta
        end
        for j=k:n, i=k:m
            A[i,j] -= v[i-k+1] * vA[j]
        end
        A[k+1:end,k] = v[2:end]
    end
end

#----------------------------------------
# Problem d
#----------------------------------------
# These two functions take in the housholder
# QR factorization from part c and multiply them
# to a vector (mul) or solve the least squares 
# problem in A (div), in place.
# They should not allocate any memory and instead
# use the preallocated output vector to record the result. 
function householder_QR_mul!(out, x, QR)
    # YOUR CODE HERE
end

function householder_QR_div!(out, b, QR)
    # YOUR CODE HERE
end