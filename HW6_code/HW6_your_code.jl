#----------------------------------------
# Problem a
#----------------------------------------
# This function takes in a matrix T and modifies it 
# in place to Hessenberg form using Householder reduction.
function hessenberg_form!(T)
    # based on Trefethen pg 198
    m = size(T, 1)
    for k = 1:m-2
        vk = T[k+1:m, k]

        save_norm = sign(vk[1])*norm(vk)
        vk[1] += sign(vk[1])*norm(vk)

        vk /= norm(vk)

        T[k+1:m, k:m] -= 2 * vk * vk' * T[k+1:m, k:m]
        T[1:m, k+1:m] -= 2 * T[1:m,k+1:m] * vk * vk'

        T[k+1, k] = -save_norm

        for i = k+2:m
        T[i,k] = 0.0
        end
    end
end

#----------------------------------------
# Problem b
#----------------------------------------
# This funciton takes in a matrix T in Hessenberg form
# and runs a single iteration of the unshifted QR Algorithm 
# using Givens rotations
function givens_qr!(T)
    m = size(T, 1)
    for k = 1:m-1
        # make givens matrix
        G = zeros(m, m)
        u = [T[k,k]; T[k+1, k]]

        if u[2] == 0
            c = 1
            s = 0
        else
            if abs(u[2]) > abs(u[1])
                tau = -u[1]/u[2]
                s = 1/sqrt(1+tau*tau)
                c = s*tau
            else
                tau = -u[2]/u[1]
                c = 1/sqrt(1+tau*tau)
                s = c*tau
            end
        end
    
        G = [c s; -s c]

        T[:, k:k+1] = T[:, k:k+1] * G
        T[k:k+1, :] = G' * T[k:k+1, :]
    end
end

#----------------------------------------
# Problem c
#----------------------------------------
# This function takes in a matrix T in Hessenberg form and 
# implements the practical QR algorithm with shifts. 
# The input shift dictates which shift type your 
# algorithm should use. For shift = "single" implement the single shift 
# and for shift = "wilkinson" implement the Wilkinson shift
function practical_QR_with_shifts!(T, shift)
    tol = 1e-15
    m = size(T, 1)

    while true
        min_val = Inf
        min_val_idx = -1
        for i=1:m-1
            if (abs(T[i+1, i]) < min_val)
                min_val = abs(T[i+1, i])
                min_val_idx = i
            end
        end

        if min_val < tol 
            if min_val_idx != 1
                # println("bounds: 1, ", min_val_idx)
                practical_QR_with_shifts!(@view(T[1:min_val_idx, 1:min_val_idx]), shift)
            end
            if min_val_idx != m-1
                # println("bounds: ", min_val_idx+1, " , ", m)
                practical_QR_with_shifts!(@view(T[min_val_idx+1:m, min_val_idx+1:m]), shift)
            end
            break
        else
            if shift == "single"
                mu = T[m, m]
            elseif shift == "wilkinson"
                # Implemented from Trefethen 222
                B = T[m-1:m,m-1:m]
                
                delta = (B[1,1] - B[2,2] )/2
                del_sgn = sign(delta)
                if delta == 0
                    del_sgn = 1
                end
                mu = B[2, 2] - del_sgn * B[1, 1]^2 / (abs(delta) + sqrt(delta^2 + B[1, 2]^2))
            end

            for i=1:m
                T[i,i] -= mu 
            end
            givens_qr!(T)
            for i=1:m
                T[i,i] += mu
            end
        end
    end
end

#----------------------------------------
# Problem d
#----------------------------------------
function practical_QR_with_shifts_hist!(T, T_hist, T_conv, shift)
    tol = 1e-15
    m = size(T, 1)

    while true
        min_val = Inf
        min_val_idx = -1
        for i=1:m-1
            if (abs(T[i+1, i]) < min_val)
                min_val = abs(T[i+1, i])
                min_val_idx = i
            end
        end

        if min_val < tol 
            # store converged eigenvalues
            if min_val_idx == 1
                push!(T_conv, T[min_val_idx, min_val_idx])
            elseif min_val_idx == m-1
                push!(T_conv, T[min_val_idx+1, min_val_idx+1])
            end

            # deflation
            if min_val_idx != 1
                practical_QR_with_shifts_hist!(@view(T[1:min_val_idx, 1:min_val_idx]), T_hist, T_conv,  shift)
            end
            if min_val_idx != m-1
                practical_QR_with_shifts_hist!(@view(T[min_val_idx+1:m, min_val_idx+1:m]), T_hist, T_conv, shift)
            end
            break
        else
            if shift == "single"
                mu = T[m, m]
            elseif shift == "wilkinson"
                # Implemented from Trefethen 222
                B = T[m-1:m,m-1:m]
                
                delta = (B[1,1] - B[2,2] )/2
                del_sgn = sign(delta)
                if delta == 0
                    del_sgn = 1
                end
                mu = B[2, 2] - del_sgn * B[1, 1]^2 / (abs(delta) + sqrt(delta^2 + B[1, 2]^2))
            end

            # apply shift
            for i=1:m
                T[i,i] -= mu 
            end
            # apply givens rotations
            givens_qr!(T)
            # undo shift
            for i=1:m
                T[i,i] += mu
            end

            # save diagonal history
            push!(T_hist, diag(T))
            T_hist[end] = append!(T_hist[end], T_conv)
        end
    end
end
