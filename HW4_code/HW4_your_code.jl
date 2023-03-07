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
function my_dot!(k, m, A)
    temp = 0
    for i=k:m
        temp += A[i,k]^2
    end
    return temp
end

# This function takes in a matrix A 
# and computes its QR factorization in place,
# using householder reflections.
# It should not allocate any memory.  
function householder_QR!(A)
    m = size(A, 1)
    n = size(A, 2)
    
    for k=1:n
        norm_of_x = my_dot!(k, m, A)^0.5
        A_sign = sign(A[k,k])

        A[k, k] += A_sign * norm_of_x
        for i=m:-1:k
            A[i,k] /= A[k,k]
        end

        v_inner = 2 / my_dot!(k, m, A)

        for j=n:-1:k+1
            col_temp = 0
            for l=k:m
                col_temp += A[l, k] * A[l, j]
            end
            col_temp *= v_inner

            for i=k:m
                A[i,j] -= A[i,k]*col_temp
            end
        end
        A[k,k] = -A_sign * norm_of_x
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
    m = size(QR,1)
    n = length(x)
    for i=1:length(out)
        out[i] = 0
    end
    # First we do Rx=out
    for i=1:n
        for j = i:n
            out[i] += QR[i,j] * x[j]
        end
    end

    # Then we do Qout = out
    for k=n:-1:1
        v_inner = 1
        for i=k+1:m
            v_inner += QR[i,k]^2
        end
        v_inner = 2 / v_inner

        col_temp = out[k]

        for l=k+1:m
            col_temp += out[l] * QR[l, k]
        end
        col_temp *= v_inner

        out[k] -= col_temp
        for i=k+1:m
            out[i] -= QR[i, k]*col_temp
        end
    end
end

function householder_QR_div!(out, b, QR)
    # First we calculate y=Q*b
    m = size(QR,1)
    n = size(QR,2)

    for k=1:n
        v_inner = 1
        for i=k+1:m
            v_inner += QR[i,k]^2
        end
        v_inner = 2 / v_inner

        col_temp = b[k]

        for l=k+1:m
            col_temp += b[l] * QR[l, k]
        end
        col_temp *= v_inner

        b[k] -= col_temp
        for i=k+1:m
           b[i] -= QR[i, k]*col_temp
        end
    end

    # Assign b to out
    for i=1:n
        out[i] = b[i]
    end

    # Solving Rx=out using back substitution
    for i=n:-1:1
        for j = i+1:n
            out[i] -= QR[i,j] * out[j]
        end
        out[i] /= QR[i, i]
    end
end