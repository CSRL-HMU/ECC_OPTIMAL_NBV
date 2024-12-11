function [X, Y, Z] = get_camera_position(r, phi, theta, offset)

    % Set default offset if not provided
    if nargin < 4
        offset = -1;
    end

    % Compute the Cartesian coordinates
    X = r * cos(phi) + offset;
    Y = r * sin(phi) * cos(theta);
    Z = r * sin(phi) * sin(theta);
end


