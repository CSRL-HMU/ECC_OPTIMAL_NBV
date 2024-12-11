function S = sigmaPointsUKF(n, x, P, alpha_)
    

    % Lambda and kappa values
    kappa = 0;
    Lambda = alpha_^2 * (n + kappa) - n;  

    % Compute the scaling factor
    scaling_factor = Lambda + n + kappa;

    % Compute the square root matrix
    sqrMtrx = scaling_factor * P;
    
    % Cholesky decomposition to obtain the lower triangular matrix
    L = chol(sqrMtrx, 'lower');  % Lower triangular Cholesky decomposition

    % Initialize sigma point matrix S (n x 2n + 1)
    S = zeros(n, 2 * n + 1);

    % Ensure x is a column vector
    x = x(:);

    % Set the first column of S to x
    S(:, 1) = x;

    % Loop through to set the remaining sigma points
    for i = 1:n
        S(:, i + 1)       = x + L(:, i);
        S(:, i + n + 1)   = x - L(:, i);
    end
end


