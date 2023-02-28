from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
from wheelcontroller.controller import WheelController

np.set_printoptions(precision=3, suppress=True)
import time
import keyboard

hz = 50
pkg_path = "/home/dyros/Isaac_Sim/husky_controller"
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=pkg_path+"/model/husky(two_wheel_sphere).usd", prim_path="/World/husky")
world.set_simulation_dt(1/hz, 1/hz)
wc = WheelController(hz=hz, world=world, pkg_path=pkg_path, is_two_wheel=True)
world.reset()


is_simulation_run = True
exit_flag = False
is_first = True

while (simulation_app.is_running() and (not exit_flag)):
    world.step(render=True)
    wc.UpdateData()
    if(is_first):
        world.reset()
        wc.UpdateData()
        print("Initial q : " )
        print(wc.getPose()) 
        is_first = False
        wc.initPosition()
    
    if keyboard.is_pressed("shift+i"):
        time.sleep(0.1)
        wc.setMode("init")
    elif keyboard.is_pressed("shift+1"):
        time.sleep(0.1)
        wc.setMode("Tracking Circle")
    elif keyboard.is_pressed("shift+2"):
        time.sleep(0.1)
        wc.setMode("Tracking Square")
    elif keyboard.is_pressed("shift+3"):
        time.sleep(0.1)
        wc.setMode("Tracking Eight")
    elif keyboard.is_pressed("shift+4"):
        time.sleep(0.1)
        wc.setMode("Velocity Command")
    elif keyboard.is_pressed("shift+p"):
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
    elif keyboard.is_pressed("shift+q"):
        time.sleep(0.1)
        is_simulation_run = False
        exit_flag = True
        
    if is_simulation_run:
        wc.compute()
        wc.write()
wc.closeFile()
simulation_app.close()
