# Isaac_Sim

## Model Predictive control with QP
paper : 
Lee, J., Seo, M., Bylard, A., Sun, R., & Sentis, L. (2023, May). Real-Time Model Predictive Control for Industrial Manipulators with Singularity-Tolerant Hierarchical Task Control. In 2023 IEEE International Conference on Robotics and Automation (ICRA) (pp. 12282-12288). IEEE.

## VScode
If you code this pkg by vscode, follow the support :
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_standalone_python.html#isaac-sim-python-vscode

## Command
  ```
  python /path_of_this_pkg/panda_controller/main.py
  ```
  
  ```
  python /path_of_this_pkg/husky_controller/main.py
  ```
  
## Requirement
  - Conda environment
    ```
    conda env create -f isaac_sim.yaml
    conda activate isaac-sim
    ```

    ```
    mkdir -p /usr/anaconda3/envs/isaac-sim/etc/conda/activate.d
    cd /usr/anaconda3/envs/isaac-sim/etc/conda/activate.d
    gedit activate.d
    ```
    
    Write Script like this
    
    ```
    #!/bin/bash
    source ~/.local/share/ov/pkg/isaac_sim-2022.2.1/setup_conda_env.sh
    ```
    
  - Edit code in Isaac Sim
    ```
    gedit ~/.local/share/ov/pkg/isaac_sim-2022.2.1/exts/omni.isaac.manipulators/omni/isaac/manipulators/single_manipulator.py
    ```
    
    In 117 line, add like below code
    
    ```
    if self._gripper is not None:
            self._gripper.post_reset()
    ```
  - Install franka-ros pkg: 
    https://frankaemika.github.io/docs/installation_linux.html
    
  - Change Rviz pkg directory to your ros work-space
    ```
      cd Issac_Sim/panda_controller
      mv panda_visualize /path_to_your_ros_ws/src
      cd /path_to_your_ros_ws && catkin build
      roslaunch panda_visualize display.launch
     ```
