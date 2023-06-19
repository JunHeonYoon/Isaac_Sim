clear all; close all; clc

load("mesh_light.mat")

is_static = false;
is_gain_1 = true;

if is_static
    data_dir = "trajectory/static";
else
    data_dir = "trajectory/dynamic";
end
if is_gain_1
    data_dir = strcat(data_dir, "/gain/");
else
    data_dir = strcat(data_dir, "/gain_2/");
end
data_file_name = ["log", "linear", "RBF"];

data = cell(length(data_file_name),1);
time = cell(length(data_file_name),1);
obs_posi = cell(length(data_file_name),1);
tar_posi = cell(length(data_file_name),1);
ee_posi = cell(length(data_file_name),1);
ee_pose = cell(length(data_file_name),1);
ee_vel = cell(length(data_file_name),1);
for i=1:length(data_file_name)
    data{i} = importdata(strcat(data_dir, data_file_name(i),".txt"), ' ', 0);
    time{i}     = data{1}(:,1);
    obs_posi{i} = data{i}(:,2:4);
    tar_posi{i} = data{i}(:,5:7);
    ee_posi{i}  = data{i}(:,8:10);
    ee_pose{i}  = data{i}(:,8:19); 
    ee_vel{i} = data{i}(:,20:22);
end


min_dist = cell(length(data_file_name),1);
for i=1:length(min_dist)
    min_dist{i} = zeros(length(ee_posi{i}),1);
    for j=1:length(min_dist{i})
        min_dist{i}(j,1) = getSD(ee_pose{i}(j,:), obs_posi{i}(j,:),mesh{9});
        min_dist{i}(j,1) = min_dist{i}(j,1) - 0.05; % radius of sphere
    end
end


mesh{9}.v = mesh{9}.v *[1 0 0;0 -1 0;0 0 -1];

for i=1:length(data_file_name)
    f = figure("Name",data_file_name(i));
    f.WindowState = 'maximized';
    tab1 = uitab("Title","Trajectory");
    ax1 = axes(tab1);
    plot3(ax1,tar_posi{i}(:,1), tar_posi{i}(:,2), tar_posi{i}(:,3), "k--", "LineWidth",3); hold on
    plot3(ax1,ee_posi{i}(:,1), ee_posi{i}(:,2), ee_posi{i}(:,3), "r", "LineWidth",2)
    xlabel("X","FontWeight","bold"); ylabel("Y","FontWeight","bold"); zlabel("Z","FontWeight","bold")
    axis equal
    xlim([min([ee_posi{i}(:,1); obs_posi{i}(:,1)])-0.05, max([ee_posi{i}(:,1); obs_posi{i}(:,1)])+0.05]);
    ylim([min([ee_posi{i}(:,2); obs_posi{i}(:,2)])-0.05, max([ee_posi{i}(:,2); obs_posi{i}(:,2)])+0.05]);
    zlim([min([ee_posi{i}(:,3); obs_posi{i}(:,3)])-0.05, max([ee_posi{i}(:,3); obs_posi{i}(:,3)])+0.05])
    [tmp_X,tmp_Y,tmp_Z] = sphere;
    for j=1:length(obs_posi{i})
        X=tmp_X*0.05+obs_posi{i}(j,1);Y=tmp_Y*0.05+obs_posi{i}(j,2);Z=tmp_Z*0.05+obs_posi{i}(j,3);
        p1 = surf(ax1,X,Y,Z);  
        p2 = patch(ax1,'Faces',mesh{9}.f,'Vertices', mesh{9}.v+repmat(ee_posi{i}(j,:),[length(mesh{9}.v),1]) ...
            ,'facecolor',[1 0 0],'Facealpha',0.05,'edgealpha',0);
        pause(0.01)
        delete(p1)
        delete(p2)
    end
    [tmp_X,tmp_Y,tmp_Z] = sphere;
    X=tmp_X*0.05+obs_posi{i}(1,1);Y=tmp_Y*0.05+obs_posi{i}(1,2);Z=tmp_Z*0.05+obs_posi{i}(1,3);
    surf(ax1,X,Y,Z)
    for j=0:7    
        patch(ax1,'Faces',mesh{9}.f,'Vertices', mesh{9}.v+repmat(ee_posi{i}(j*100+1,:),[length(mesh{9}.v),1]) ...
            ,'facecolor',[1 0 0],'Facealpha',0.05,'edgealpha',0)
    end
    hold off
    % saveas(gca, strcat("figure/Traj(",data_file_name(i),").svg" ))


    tab2 = uitab("Title","Error");
    ax2 = axes(tab2);
    subplot(4,1,1)
    plot(time{i}, tar_posi{i}(:,1)-ee_posi{i}(:,1), "LineWidth",2)
    title("X Error", "FontWeight","bold"); xlabel("Time [sec]", "FontWeight","bold"); ylabel("Error [m]", "FontWeight","bold")
    grid on
    subplot(4,1,2)
    plot(time{i}, tar_posi{i}(:,2)-ee_posi{i}(:,2), "LineWidth",2)
    title("Y Error", "FontWeight","bold"); xlabel("Time [sec]", "FontWeight","bold"); ylabel("Error [m]", "FontWeight","bold")
    grid on
    subplot(4,1,3)
    plot(time{i}, tar_posi{i}(:,3)-ee_posi{i}(:,3), "LineWidth",2)
    title("Z Error", "FontWeight","bold"); xlabel("Time [sec]", "FontWeight","bold"); ylabel("Error [m]", "FontWeight","bold")
    grid on
    subplot(4,1,4)
    plot(time{i}, row_norm(tar_posi{i}-ee_posi{i}), "LineWidth",2)
    title("Error", "FontWeight","bold"); xlabel("Time [sec]", "FontWeight","bold"); ylabel("Error [m]", "FontWeight","bold")
    grid on


    disp(strcat("Mininum distance (", data_file_name(i), ") [cm]:"))
    disp(min(min_dist{i})*100)
    disp(strcat("Error Average(",data_file_name(i), ") [cm] : "))
    disp(mean(row_norm(tar_posi{i}-ee_posi{i}))*100)
    disp(strcat("Max Speed(",data_file_name(i), ") [m/s] : "))
    disp(max(row_norm(ee_vel{i})))
    % saveas(gca, strcat("figure/Error(",data_file_name(i),").svg" ))
end


function result = row_norm(x)
    n = size(x);
    result = zeros(n(1),1);
    for i=1:n(1)
        result(i,1) = norm(x(i,:));
    end
end

function distance = getSD(pose, obs_posi, mesh)
    % t and r transposed
    t = pose(1:3);
    r = [pose(4:6) ; pose(7:9); pose(10:12)];

    f = mesh.f;
    v = mesh.v*r + t;

    [dst, pt_closest] = point2trimesh('Faces',f,...
                                      'Vertices',v,...
                                      'QueryPoints',obs_posi,...
                                      'Algorithm', 'vectorized');
    IN = inpolyhedron(f, v, obs_posi);
    distance = abs(dst);
    if IN
        distance = distance * -1;
    end
end

