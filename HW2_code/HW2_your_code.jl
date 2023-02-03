# This macro helps optimize the innermost loop in the kernel on your machine. You should not need to use it anywhere else
using LoopVectorization: @turbo
# This function takes in matrices ABC and adds B times C to the matrix A
function add_to_A_B_times_C!(A, B, C)
    @turbo for j in axes(C, 2)
        for k in axes(B, 2)
            for i in axes(A, 1)
                A[i, j] += B[i, k] * C[k, j]
            end
        end
    end
end

# This function takes in matrices ABC and adds B times C to the matrix A
# It uses blocking into blocks of size bks
# Make sure that your function does not allocate memory
function add_to_A_B_times_C!(A, B, C, bks)
    for j in 1:ceil(Int, size(C, 2)/bks)
        for k in 1:ceil(Int, size(B, 2)/bks)
            for i in 1:ceil(Int, size(A, 1)/bks)
                for block_j in Int((j-1)*bks):min(Int((j)*bks-1), size(C, 2)-1)
                    for block_k in Int((k-1)*bks):min(Int((k)*bks-1), size(B,2)-1)
                        for block_i in Int((i-1)*bks):min(Int((i)*bks-1), size(A,1)-1)
                            A[block_i+1, block_j+1] += B[block_i+1, block_k+1] * C[block_k+1, block_j+1]
                        end
                    end
                end
            end
        end
    end
end

# Implements a recursive, cache oblivious algorithm
# complete this skeleton
function oblivious_add_to_A_B_times_C!(A, B, C, bks)
    i_size = size(A, 1)
    j_size = size(C, 2)
    k_size = size(B, 2)

    # If we want to further subdivide
    if min(i_size, j_size, k_size) > bks
        # a11 = b11*c11 + b12 * c21
        oblivious_add_to_A_B_times_C!(
            @view(A[1:div(i_size, 2), 1:div(j_size, 2)]),
            @view(B[1:div(i_size, 2), 1:div(k_size, 2)]),
            @view(C[1:div(k_size, 2), 1:div(j_size, 2)]),
            bks
        )
        oblivious_add_to_A_B_times_C!(
            @view(A[1:div(i_size, 2), 1:div(j_size, 2)]),
            @view(B[1:div(i_size, 2), div(k_size, 2)+1:end]),
            @view(C[div(k_size, 2)+1:end, 1:div(j_size, 2)]),
            bks
        )
        # a12 = b11*c12 + b12 * c22
        oblivious_add_to_A_B_times_C!(
            @view(A[1:div(i_size, 2), div(j_size, 2)+1:end]),
            @view(B[1:div(i_size, 2), 1:div(k_size, 2)]),
            @view(C[1:div(k_size, 2), div(j_size, 2)+1:end]),
            bks
        )
        oblivious_add_to_A_B_times_C!(
            @view(A[1:div(i_size, 2), div(j_size, 2)+1:end]),
            @view(B[1:div(i_size, 2), div(k_size, 2)+1:end]),
            @view(C[div(k_size, 2)+1:end, div(j_size, 2)+1:end]),
            bks
        )
        # a21 = b21*c11 + b22 * c21
        oblivious_add_to_A_B_times_C!(
            @view(A[div(i_size, 2)+1:end, 1:div(j_size, 2)]),
            @view(B[div(i_size, 2)+1:end, 1:div(k_size, 2)]),
            @view(C[1:div(k_size, 2), 1:div(j_size, 2)]),
            bks
        )
        oblivious_add_to_A_B_times_C!(
            @view(A[div(i_size, 2)+1:end, 1:div(j_size, 2)]),
            @view(B[div(i_size, 2)+1:end, div(k_size, 2)+1:end]),
            @view(C[div(k_size, 2)+1:end, 1:div(j_size, 2)]),
            bks
        )
        # a22 = b21*c12 + b22 * c22
        oblivious_add_to_A_B_times_C!(
            @view(A[div(i_size, 2)+1:end, div(j_size, 2)+1:end]),
            @view(B[div(i_size, 2)+1:end, 1:div(k_size, 2)]),
            @view(C[1:div(k_size, 2), div(j_size, 2)+1:end]),
            bks
        )
        oblivious_add_to_A_B_times_C!(
            @view(A[div(i_size, 2)+1:end, div(j_size, 2)+1:end]),
            @view(B[div(i_size, 2)+1:end, div(k_size, 2)+1:end]),
            @view(C[div(k_size, 2)+1:end, div(j_size, 2)+1:end]),
            bks
        )
    #If we are ready to break the recursion
    else
        add_to_A_B_times_C!(A, B, C)
    end
end

