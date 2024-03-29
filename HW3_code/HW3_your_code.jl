#----------------------------------------
# Problem a
#----------------------------------------
# This function takes in the LU factorization
# together with the permutation P (in the form of
# an array of integers such that P[i] = j means that multiplication with P moves the i=th row to the j-th position)
# It should modify the input variable x in place
function substitution!(x, LU, P = 1 : size(x, 1)) 
    # YOUR CODE HERE
    n = length(x)

    # permute x
    permute!(x, P)

    # Solve Ly=b
    for i=1:n
        for j = i-1:-1:1
            x[i] -= LU[i,j] * x[j]
        end
    end

    #Solve Ux=y
    for i=n:-1:1
        for j = i+1:n
            x[i] -= LU[i,j] * x[j]
        end
        x[i] /= LU[i, i]
    end
end

#----------------------------------------
# Problem b
#----------------------------------------
# This function takes in a matrix A and modifies
# it in place, such that is contains the stricly
# lower triangular part of L together with the
# upper triangular part of U.
function unpivoted_LU!(A)
    # YOUR CODE HERE
    n = size(A, 1)
    for j=1:n
        for k=1:j-1
            for i=k+1:n
                A[i, j] -= A[i, k] * A[k, j]
            end
        end

        for i = j+1:n
            A[i, j] /= A[j, j]
        end
    end
end

#----------------------------------------
# Problem d
#----------------------------------------
# This function takes in a matrix A and modifies
# it in place, such that is contains the stricly
# lower triangular part of L together with the
# upper triangular part of U.
# It uses row-pivoting and stores the resulting 
# row permutation in the array P
function pivoted_LU!(A)
    # The array that will be used to keep track of the permutation
    P = collect(1 : size(A, 1))
    # YOUR CODE HERE
    # returns the array representing the permutation
    n = size(A, 1)
    for k=1:n
        imx = k-1+argmax(abs.(A[k:end,k]))
        for j=1:n
            A[k,j],A[imx,j] = A[imx, j],A[k,j]
        end
        P[[k,imx]] = P[[imx,k]]

        for i=k+1:n
            A[i,k] /= A[k,k]
        end

        for j=k+1:n,i=k+1:n
            A[i,j] -= A[i,k]*A[k,j]
        end
    end
    return P
end

#----------------------------------------
# Problem e
#----------------------------------------
# Creates an m × m matrix with a particularly 
# large growth factor
function growth_matrix(m)
    # YOUR CODE GOES HERE
    A = zeros(m, m)
    for i=1:m
        A[i, i] = 1
        A[i, m] = 1
        for j=1:i-1
            A[i, j] = -1
        end
    end
    return(A)
end
