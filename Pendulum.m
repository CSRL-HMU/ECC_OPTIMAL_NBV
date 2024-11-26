function state = Pendulum(x_pos0,y_pos0,z_pos0,vx0,vy0,vz0)
global L  g  b  m dt
% Initial conditions [x, x_dot, y, y_dot, z, z_dot]
% x_pos0 = L/sqrt(2);  % Initial x-position
% vx0 = 0;  % Initial x-velocity
% y_pos0 = L/sqrt(2);  % Initial y-position
% vy0 = 0;  % Initial y-velocity
% z_pos0 = 0; % Initial z-position (length of the pendulum)
% vz0 = 0;  % Initial z-velocity

% Time parameters
     
t_end = 3;    % End time for simulation
t = 0:dt:t_end; % Time vector

% Preallocate arrays for storing results

v_dot = zeros(length(t),3);
p_dot = zeros(length(t),3);
p  = zeros(length(t),3);

% Set initial conditions
p(1,1) = x_pos0;
p(1,2) = y_pos0;
p(1,3) = z_pos0;
p_dot(1,1) = vx0;
p_dot(1,2) = vy0;
p_dot(1,3) = vz0;
    
    
%     % Time-stepping loop
%     for i = 1:length(t)-1
%         % Compute accelerations from state-space equations
%         
%         v_dot(i,1) = (g * p(i,3) / L^2) * p(i,1) - b*p_dot(i,1);
%         v_dot(i,2) = (g * p(i,3) / L^2) * p(i,2) - b*p_dot(i,2);
%         v_dot(i,3) = -g +(g * (p(i,3)*p(i,3)) / L^2) - b*p_dot(i,3);
%         
%         % Update velocities 
%         p_dot(i+1,:) = p_dot(i,:) + v_dot(i,:)*dt;
%         
%         % Update positions 
%         p(i+1,:) = p(i,:) +  p_dot(i+1,:)*dt; 
%         
%     end
 

    % Time-stepping loop
    for i = 1:length(t)-1
      
        Np = eye(3) - p(i,:)'*p(i,:)/(norm(p(i,:))^2);
        % Compute accelerations from state-space equations
        
        v_dot(i,:) = (Np*[0; 0; -g])' - (b/m)*p_dot(i,:);
        
        % Update velocities 
        
        p_dot(i+1,:) = p_dot(i,:) + v_dot(i,:)*dt;
        
        % Update positions 
        p(i+1,:) = p(i,:) + (Np*p_dot(i+1,:)')'*dt; 
        
    end

state = [p(:,1),p(:,2),p(:,3),p_dot(:,1),p_dot(:,2),p_dot(:,3)];
end
