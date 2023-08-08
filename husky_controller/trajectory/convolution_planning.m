clear all;clc;close all
%% Path Setting

% Circle (mobile, r = 2, hz = 50)
% is_mobile = true;
% hz = 50;
% title = "circle";
% tt = 0:0.05:2*pi;
% tt = [tt, 2*pi];
% Desired_position = [2*sin(tt)', 2*(1-cos(tt))'];

% Square(mobile, l = 4, hz = 50) 
% is_mobile = true;
% hz = 50;
% title = "square";
% Desired_position = [0, 0;
%                     4, 0;
%                     4, 4;
%                     0, 4;
%                     0, 0];

% Eight (mobile, r = 2, hz = 50)
is_mobile = true;
hz = 50;
title = "eight";
tt = 0:0.05:2*pi;
tt = [tt, 2*pi];
Desired_position = [2*sin(2*tt)', 2*(1-cos(tt))'];




% Circle (manipulator, r = 0.1, hz = 100)
% is_mobile = false;
% hz = 2000;
% title = "circle";
% tt = 0:0.05:2*pi;
% tt = [tt, 2*pi];
% Desired_position = [0.1*(1-cos(tt))', 0.1*sin(tt)'];

% Square(manipulator, l = 0.2, hz = 100) 
% is_mobile = false;
% hz = 2000;
% title = "square";
% Desired_position = [0, 0;
%                     0, -0.1;
%                     0.2, -0.1;
%                     0.2, 0.1;
%                     0, 0.1;
%                     0, 0];

% Eight (manipulator, r = 0.1, hz = 100)
% is_mobile = false;
% hz = 2000;
% title = "eight";
% tt = -pi/2:0.05:3/2*pi;
% tt = [tt, 3/2*pi];
% Desired_position = [0.1*sin(2*tt)', 0.1*(1-cos(tt))'-0.1];



Desired_velocity = 0.25; % unit : m/s

Sampling_time = 1/hz; % unit : s



%% Trajactory planning
if is_mobile
    Acceleration_time = 1; % unit : s
else
    Acceleration_time = 0.2; % unit : s
end
Starting_time = 2; % unit : s
Last_resting_time = 2; % unit : s


for i = 2 : size(Desired_position,1)
    % 1) tangential dist
    Tangential_dist = sqrt((Desired_position(i,1)-Desired_position(i-1,1))^2+(Desired_position(i,2)-Desired_position(i-1,2))^2);
    % 2) travel time
    Travel_time(i-1,1) =  Tangential_dist/Desired_velocity;
    
    time_step = 0:Sampling_time:Travel_time(i-1,1);
    Interpolated_posi_X = Desired_position(i-1,1)+((Desired_position(i,1)-Desired_position(i-1,1))/Travel_time(i-1,1))*time_step;
    Interpolated_posi_Y = Desired_position(i-1,2)+((Desired_position(i,2)-Desired_position(i-1,2))/Travel_time(i-1,1))*time_step;
    
    if i == 2
       Total_interpolated_posi_X =  Interpolated_posi_X';
       Total_interpolated_posi_Y =  Interpolated_posi_Y';
    else    
       Total_interpolated_posi_X = [Total_interpolated_posi_X;Interpolated_posi_X']; %Position_X Profile
       Total_interpolated_posi_Y = [Total_interpolated_posi_Y;Interpolated_posi_Y']; %Position_Y Profile
    end

end

Total_time = [0:Sampling_time:(size(Total_interpolated_posi_X,1)-1)*Sampling_time]'; 



Ti=Starting_time; % Starting time (s)
Ta=Acceleration_time; % Acceleration time (s)
Ts=Sampling_time; % Sampling time
Tl=Last_resting_time; % Last resting time(s)

Tf= Ti+Ts*length(Total_interpolated_posi_X)+Tl ; % Simulation time(Action Time + Last resting time) (s) %why Last resting time
Target_velocity_X(1,1) = 0;
Target_velocity_Y(1,1) = 0;

for j = 2 : length(Total_interpolated_posi_X)
  
    Target_velocity_X(j,1) = (Total_interpolated_posi_X(j)-Total_interpolated_posi_X(j-1))/Ts; % Target velocity X (mm/s), %Velocity_X Profile
    Target_velocity_Y(j,1) = (Total_interpolated_posi_Y(j)-Total_interpolated_posi_Y(j-1))/Ts; % Target velocity Y (mm/s), %Velocity_Y Profile
    
end

t=0:Ts:Tf; % Time (s)
Interpolated_X=[0,0]; % Interpoletedprofile X (s, m/s)
Interpolated_Y=[0,0]; % Interpoletedprofile Y (s, m/s)
Last_flag = 0;

for i=1:length(t)
    if t(i)<=Ti
        Interpolated_X(i,:)=[t(i),0];
        Interpolated_Y(i,:)=[t(i),0];
        Last_flag = Last_flag+1;
    elseif t(i) <= Ti+Ts*max(size(Total_interpolated_posi_X))
        Interpolated_X(i,:)=[t(i),Target_velocity_X(i-Last_flag,1)]; %Velocity_X Profile
        Interpolated_Y(i,:)=[t(i),Target_velocity_Y(i-Last_flag,1)]; %Velocity_Y Profile
    else
        Interpolated_X(i,:)=[t(i),0];
        Interpolated_Y(i,:)=[t(i),0];
    end
end

% Velocity profiler(2)

m=Ta/Ts; % Points for acceleration
Final_X=[0,0]; % Final profile (s, m/s)
Final_Y=[0,0]; % Final profile (s, m/s)
acc_X = [0,0];
acc_Y = [0,0];
Step1_output = [Total_time(1),Total_interpolated_posi_X(1),Total_interpolated_posi_Y(1)];

for i=2:length(t)
    if i<=m
        Final_X(i,:)=[t(i), Final_X(i-1,2)+(Interpolated_X(i,2)-Interpolated_X(1,2))/(i-1)];
        Final_Y(i,:)=[t(i), Final_Y(i-1,2)+(Interpolated_Y(i,2)-Interpolated_Y(1,2))/(i-1)];
    else
        Final_X(i,:)=[t(i), Final_X(i-1,2)+(Interpolated_X(i,2)-Interpolated_X(i-m,2))/m];
        Final_Y(i,:)=[t(i), Final_Y(i-1,2)+(Interpolated_Y(i,2)-Interpolated_Y(i-m,2))/m];
    end
    Step1_output(i,:) = [t(i), Step1_output(i-1,2) + Final_X(i,2)*Ts, Step1_output(i-1,3) + Final_Y(i,2)*Ts];
end
for i=2:length(t)
    acc_X(i, :) =[t(i), (Final_X(i,2)-Final_X(i-1,2))*hz];
    acc_Y(i, :) =[t(i), (Final_Y(i,2)-Final_Y(i-1,2))*hz];
end

figure("Name", "Position")
plot(Step1_output(:, 2), Step1_output(:, 3)); hold on
plot(Desired_position(:,1), Desired_position(:,2))
grid on

Final(:, 1) = Final_X(:, 1);
Final(:, 2) = sqrt(Final_X(:, 2).^2 + Final_Y(:, 2).^2);

figure("Name", "Velocity")
subplot(3, 1, 1)
plot(Final_X(:, 1), Final_X(:, 2))
grid on
subplot(3, 1, 2)
plot(Final_Y(:, 1), Final_Y(:, 2))
grid on
subplot(3, 1, 3)
plot(Final(:, 1), Final(:, 2))
grid on

acc(:, 1) = acc_X(:, 1);
acc(:,2) = sqrt(acc_X(:,2).^2 + acc_Y(:,2).^2);

figure("Name", "Accelaration")
subplot(3, 1, 1)
plot(acc_X(:, 1), acc_X(:, 2))
grid on
subplot(3, 1, 2)
plot(acc_Y(:, 1), acc_Y(:, 2))
grid on
subplot(3, 1, 3)
plot(acc(:, 1), acc(:, 2))
grid on

fileID = fopen(title+".txt",'w');

if is_mobile  
%     fprintf(fileID,'%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[Step1_output'; Final_X(:, 2)'; Final_Y(:, 2)'; acc_X(:, 2)'; acc_Y(:,2)']);
%     fprintf(fileID,'%.4f %.4f %.4f %.4f %.4f\n',[Step1_output'; Final_X(:, 2)'; Final_Y(:, 2)']);
else
    fprintf(fileID,'%.4f %.4f %.4f %.4f %.4f\n',[Step1_output'; Final_X(:, 2)'; Final_Y(:, 2)']);
end
fclose(fileID);


if is_mobile
    len = length(Final);
    x = Step1_output(:, 2);
    y = Step1_output(:, 3);
    x_dot = Final_X(:, 2);
    y_dot = Final_Y(:, 2);
    x_dotdot = acc_X(:,2);
    y_dotdot = acc_Y(:,2);
    
    th = zeros(len,1);
    for i=2:len
        if abs(atan2(y_dot(i), x_dot(i))) < 0.0001
            th(i) =  th(i-1);
        else
            th(i) = atan2(y_dot(i), x_dot(i));
        end
    end
    w = zeros(len,1);
    for i=1:len
        if((x_dot(i)^2 + y_dot(i)^2) > 0.001)
            w(i) = (x_dot(i)*y_dotdot(i)-y_dot(i)*x_dotdot(i))/(x_dot(i)^2 + y_dot(i)^2);
        end
    end
    vx = x_dot.*cos(th) + y_dot.*sin(th);
    vy = -x_dot.*sin(th) + y_dot.*cos(th);
    fileID = fopen(title+".txt",'w');
    fprintf(fileID,'%.4f %.4f %.4f %.4f %.4f %.4f\n',[x'; y'; th'; vx'; vy'; w']);
    fclose(fileID)
end
