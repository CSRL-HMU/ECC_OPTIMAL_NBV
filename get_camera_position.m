function [X, Y, Z] = get_camera_position(r, phi, theta, offset)
    % get_camera_position computes the camera's position in 3D space.
    %
    % Syntax:
    %   [X, Y, Z] = get_camera_position(r, phi, theta)
    %   [X, Y, Z] = get_camera_position(r, phi, theta, offset)
    %
    % Inputs:
    %   r      - Radius or distance from the origin
    %   phi    - Azimuth angle in radians (angle from the x-axis in the xy-plane)
    %   theta  - Elevation angle in radians (angle from the z-axis)
    %   offset - (Optional) Offset along the x-axis (default value: -1)
    %
    % Outputs:
    %   X, Y, Z - Cartesian coordinates of the camera position

    % Set default offset if not provided
    if nargin < 4
        offset = -1;
    end

    % Compute the Cartesian coordinates
    X = r * cos(phi) + offset;
    Y = r * sin(phi) * cos(theta);
    Z = r * sin(phi) * sin(theta);
end


