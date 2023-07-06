# Isaac_Sim

## Obstacle avoidance based on JSDF
paper : 
Koptev, Mikhail, Nadia Figueroa, and Aude Billard. "Neural Joint Space Implicit Signed Distance Functions for Reactive Robot Manipulator Control." IEEE Robotics and Automation Letters 8.2 (2022): 480-487.

## VScode
If you code this pkg by vscode, follow the support :
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_standalone_python.html#isaac-sim-python-vscode

## Command
1. Go to Isaac Sim root folder (default :)
  ```
  ~/.local/share/ov/pkg/isaac_sim-2022.2.0
  ```
2. command
  ```
  sudo ./python.sh /path_of_this_pkg/panda_controller/main.py
  ```
  
  ```
  sudo ./python.sh /path_of_this_pkg/husky_controller/main.py
  ```
  
  3. Requirement
  - Conda environment
    ```
    conda env create -f isaac_sim.yaml
    conda activate isaac-sim
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
  - Make IK shared file(.so) coded by C++
    ```
      cd Issac_Sim/panda_controller/ik
      mkdir build && cd build
      cmake ..
      make
     ```
