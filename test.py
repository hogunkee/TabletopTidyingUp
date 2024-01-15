import time
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI) # non-graphical version
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setTimeStep(1 / 240)

# set the plane and table        
planeID = p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])

p.loadURDF("./YCB_dataset/vinegar_000/vinegar_000.urdf", basePosition=[0.0,0.0,0.7], baseOrientation=[0.0,0.0,0.7071,0.7071],globalScaling=0.01)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)