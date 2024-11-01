
close all;
% Parameters
r = 5;             % Radius of the hemisphere
offset = -1;       % Offset for the dome center along the x-axis
n_points = 100;    % Number of points for the trajectory
z_level = r / 2;   % Set a constant z-level for the horizontal trajectory within the dome

% Create mesh grid for the hemispherical dome (theta, phi)
theta = linspace(0, 2*pi, 50);    % Azimuth angle
phi = linspace(0, pi/2, 25);      % Elevation angle (upper half only)
[Theta, Phi] = meshgrid(theta, phi);

% Parametric equations for the hemisphere in Cartesian coordinates
X_dome = r * cos(Phi) + offset;
Y_dome = r * sin(Phi) .* cos(Theta);
Z_dome = r * sin(Phi) .* sin(Theta);

% Create a horizontal cosine trajectory along the y-axis at z = z_level
x_traj =  ones(1, n_points);                 % Keep x constant
y_traj = (linspace(-r/2, r/8, n_points));                  % Vary y across the dome
z_traj = z_level+sin( 2*pi * y_traj / r);   % Sine wave along y-axis

% Plot the hemispherical dome
figure;
hold on;
surf(X_dome, Y_dome, Z_dome, 'FaceColor', [0.8, 0.5, 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Plot the horizontal cosine trajectory inside the dome along y-axis
plot3(x_traj, y_traj, z_traj, 'b', 'LineWidth', 2, 'DisplayName', 'Horizontal Sine Trajectory');

% Additional plot settings
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Horizontal Cosine Trajectory Inside Hemispherical Dome');
axis equal;
xlim([-r+offset, r+offset]);
ylim([-r, r]);
zlim([0, r]);

% Show legend and grid
legend('show');
grid on;
hold off;
