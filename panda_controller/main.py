from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import keyboard
import time
import numpy as np
from armcontroller.controller import ArmController
from armcontroller.sensor import ContactSensors


hz = 100
pkg_path = "/home/dyros/Isaac_Sim/panda_controller"
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=pkg_path+"/model/panda_arm_hand.usd", prim_path="/World/panda")
world.set_simulation_dt(1/hz, 1/hz) 
ac = ArmController(hz, world, pkg_path)
contact_sensor_prim_path =  ["/World/panda/panda_leftfinger/contact_sensor", 
                             "/World/panda/panda_rightfinger/contact_sensor"]
contact_sensor_translations = np.array([[0., 0., 0.045],
                                        [0., 0., 0.045]])
contact = ContactSensors(world=world, 
                         prim_path=contact_sensor_prim_path, 
                         translations=contact_sensor_translations,
                         sensor_radius=0.01)
world.reset()


is_simulation_run = True
exit_flag = False
is_first = True

while (simulation_app.is_running() and (not exit_flag)):
    world.step(render=True)
    ac.UpdateData()
    if(is_first):
        world.reset()
        ac.UpdateData()
        print("Initial q : " )
        print(ac.getPosition()) 
        is_first = False
        ac.initPosition()
    
    if keyboard.is_pressed("i"):
        time.sleep(0.1)
        ac.setMode("joint_ctrl_init")
    elif keyboard.is_pressed("1"):
        time.sleep(0.1)
        ac.setMode("CLIK")
    elif keyboard.is_pressed("o"):
        time.sleep(0.1)
        ac.setMode("gripper_open")
    elif keyboard.is_pressed("c"):
        time.sleep(0.1)
        ac.setMode("gripper_close")
    elif keyboard.is_pressed("p"):
        if is_simulation_run:
            time.sleep(0.1)
            world.pause()  
            print("Simulation Paused")
            is_simulation_run = False
        else:
            time.sleep(0.1)
            world.play()
            print("Simulation Play")
            is_simulation_run = True
    elif keyboard.is_pressed("q"):
        time.sleep(0.1)
        is_simulation_run = False
        exit_flag = True
        
    if is_simulation_run:
        ac.compute()
        print(contact.getForce())
        ac.write()
simulation_app.close()