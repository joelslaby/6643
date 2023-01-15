# This function takes in a matrix A and a vector v and writes their product into the vector v
function u_is_A_times_v!(u, A, v)
    for row in 1:size(A)[1]
        u[row] = 0
        for col in 1:size(A)[2]
            u[row] += A[row, col] * v[col]
        end
    end
end

# This function takes in matrices ABC and writes B times C into the matrix A
function A_is_B_times_C!(A, B, C)
    # Write your function here! 
    for row in 1:size(B)[1]
        for col in 1:size(C)[2]
            A[row, col] = 0
            for idx in 1:size(B)[2]
                A[row, col] += B[row, idx] * C[idx, col]
            end
        end
    end
end

