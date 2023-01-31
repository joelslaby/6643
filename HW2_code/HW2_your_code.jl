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
    @turbo for j in 1:ceil(Int, size(C, 2)/bks)
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

    i_size /= 2
    j_size /= 2
    k_size /= 2

    # If we want to further subdivide
    if min(i_size, j_size, k_size) > bks
        oblivious_add_to_A_B_times_C!(A, B, C, bks)
    #If we are ready to break the recursion
    else
        add_to_A_B_times_C!(A[], B[], C[])
    end
end

