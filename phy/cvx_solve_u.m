function u = cvx_solve_u(A_in, B_in, x_cell_in)
    a = [A_in{:}]; 
    xa = cell2mat(a); 
    A = double(reshape(xa,1024,1024));
    b = [B_in{:}]; 
    xb = cell2mat(b); 
    B = double(reshape(xb,1024,1024));
    [x,y] = size(x_cell_in);
    x_cell = cell([y 1]);
    for i=1:y
        cell_ele = x_cell_in{1,i};
        cell_ele_li = [cell_ele{:}]; 
        cell_ele_mat = cell2mat(cell_ele_li); 
        x_cell{i,1} = double(reshape(cell_ele_mat,16,64));
    end

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