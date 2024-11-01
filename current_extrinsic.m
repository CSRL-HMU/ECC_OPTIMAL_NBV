function T = current_extrinsic(X_pos_of_len, Y_pos_of_len, Z_pos_of_len)
    % current_extrinsic computes the extrinsic transformation matrix for the camera.
    %
    % Inputs:
    %   X_pos_of_len - X-coordinate of the camera position
    %   Y_pos_of_len - Y-coordinate of the camera position
    %   Z_pos_of_len - Z-coordinate of the camera position
    %
    % Output:
    %   T - Homogeneous transformation matrix (4x4)

    % Target point at the origin
    target_point = [0; 0; 0];
    
    % Translation vector (camera position)
    translation_vector = [X_pos_of_len; Y_pos_of_len; Z_pos_of_len];
    
    % Forward vector (pointing from camera to target point)
    forward = target_point - translation_vector;
    forward = forward / norm(forward);  % Normalize forward vector
    
    % Up vector (towards z-axis)
    up = [0; 0; 1];
    
    % Check for the degenerate case at the "north pole" (Phi = 0, forward = up)
    if norm(forward - up) < 1e-6
        % When forward and up are the same, we set an arbitrary right vector
        right = [1; 0; 0];  % You can choose any vector perpendicular to up
        up = cross(right, forward);  % Recalculate up as orthogonal to forward and right
    else
        % Right vector (cross product of up and forward)
        right = cross(up, forward);
        right = right / norm(right);  % Normalize right vector
        
        % Recompute up vector as orthogonal to forward and right
        up = cross(forward, right);
    end
    
    % Rotation matrix R
    R = [right, up, forward];
    
    % Homogeneous transformation matrix T
    T = [R, translation_vector; 0, 0, 0, 1];
end

