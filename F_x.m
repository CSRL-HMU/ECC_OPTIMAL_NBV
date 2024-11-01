function state = F_x(X, QdiagIn)
    % F_x - State transition function for the pendulum system.
    %
    % Inputs:
    %   X        - Current state vector [x_pos; y_pos; z_pos; vx; vy; vz] (6x1)
    %   QdiagIn  - Process noise covariance matrix or standard deviation vector
    %
    % Outputs:
    %   state    - Updated state vector with process noise (6x1)

    % Global variables
    global L g b m dt
    g = 9.81;   % gravitational acceleration (m/s^2)
    L = 1;      % length of the pendulum (m)
    b = 0.04;   % damping coefficient
    m = 1;      % mass of the pendulum bob (kg)
    dt = 0.033; % time step (s)

    % Extract positions and velocities from X
    p = X(1:3);      % Position vector (3x1)
    p_dot = X(4:6);  % Velocity vector (3x1)

    % Compute Np matrix
    Np = eye(3) - (p * p') / (norm(p)^2);

    % Compute accelerations from state-space equations
    v_dot = (Np * [0; 0; -g]) - (b / m) * p_dot;

    % Update velocities
    p_dot = p_dot + v_dot * dt;

    % Update positions
    p = p + (Np * p_dot) * dt;

    % Generate process noise
    process_noise = QdiagIn .* randn(6,1); % Adjust this line based on QdiagIn

    % Return the updated state with noise
    state = [p; p_dot] + process_noise;
end

