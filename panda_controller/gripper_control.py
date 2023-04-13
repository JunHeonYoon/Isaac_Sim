from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka.franka import Franka
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import math

my_world = World(stage_units_in_meters=1.0)
#TODO: change this to your own path
asset_path = "/home/yoonjunheon/Isaac_Sim/panda_controller/model/panda_arm_hand.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/panda")

#define the manipulator
my_denso = my_world.scene.add(Franka(prim_path="/World/panda", name="franka_robot",
                                                end_effector_prim_name="panda_hand",
                                                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
                                                gripper_open_position = np.array([0, 0]),
                                                gripper_closed_position = np.array([0.04, 0.04]),))
#set the default positions of the other gripper joints to be opened so
#that its out of the way of the joints we want to control when gripping an object for instance.
joints_default_positions = np.zeros(9)
joints_default_positions[3] = -math.pi/2
joints_default_positions[5] = math.pi/2
joints_default_positions[6] = math.pi/4
joints_default_positions[7] = 0
joints_default_positions[8] = 0
my_denso.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()
my_world.reset()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_denso.gripper.get_joint_positions()
        if i > 500:
            #close the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.04/500, gripper_positions[1] - 0.04/500]))
        if i < 500:
            #open the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.04/500, gripper_positions[1] + 0.04/500]))
        if i == 1000:
            i = 0

simulation_app.close()