function u = cvx_solve_u(A, B, x_cell)
    T = length(x_cell);
    n = numel(x_cell{1});
    Xvec = [];
    for id = 1 : T
        xmat = x_cell{id};
        xmat_t = xmat';
        Xvec = [Xvec; xmat_t(:)];
    end
    A = sparse(A);
    B = sparse(B);
    BigA = A;
    BigB = B;
    for id = 1 : T - 2
        BigA = blkdiag(BigA, A);
        BigB = blkdiag(BigB, B);
    end
    addpath(genpath('..\..\..\cvx')); % CVX path
    cvx_begin quiet
        cvx_precision low
        variables u(n);
        minimize(norm(Xvec(n+1 : n*T)- BigA * Xvec(1 : n*(T-1)) - BigB * repmat(u, T-1, 1))/(T-1))
     %   subject to
     %       u >= 0;
    cvx_end  
end