function [x, y, z] = getEllipsoidSurf(A, c, scale)
    

    [eigenvectors, eigenvalues] = eig(A);


    [x, y, z] = sphere();

    [N, M] = size(x);
   
    for i=1:N
        for j=1:M

            p(1) = x(i,j);
            p(2) = y(i,j);
            p(3) = z(i,j);

            
            p_new = c + scale * eigenvectors * eigenvalues * p';

    
            x(i,j) = p_new(1);
            y(i,j) = p_new(2);
            z(i,j) = p_new(3);

        end
    end

end

