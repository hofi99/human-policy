import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from pathlib import Path
import sys

# Add parent directory to path to import cet modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cet.sim_mujoco import MujocoSim

def main():
    # Initialize simulator with G1 config
    config_files = ['g1_dex3_sim.yml']
    root_path = Path(__file__).parent.parent
    
    print("Initializing G1 simulator...")
    sim = MujocoSim(
        config_files=config_files,
        root_path=str(root_path),
        is_viewer=True,
        tasktype='test'  # Use 'test' scene for simple testing
    )
    
    print("Starting simulation with right arm motion test...")
    print("G1 right arm joint indices: 29-35 (7 DOF)")
    print("Joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw")
    
    # Get initial state (all zeros for standing still)
    dummy_action = np.zeros(128, dtype=np.float32)
    qpos, head_rot = sim.generate_qpos(dummy_action, init_state=True)
    
    # Set left arm (non-active) to neutral position
    # Left arm indices: 15-21 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw)
    qpos[15:22] = 0.0  # All joints at zero
    
    # Launch viewer
    with mj.viewer.launch_passive(sim.model, sim.data) as viewer:
        sim.setup_viewer(viewer)
        
        print("\nRunning test: Right arm oscillating motion")
        print("Left arm set to neutral position (all zeros)")
        print("Press Ctrl+C to stop\n")
        
        try:
            for t in range(10000):  # Run for a long time
                # Create oscillating motion for right arm
                # Right arm indices: 29-35
                # 29: right_shoulder_pitch
                # 30: right_shoulder_roll  
                # 31: right_shoulder_yaw
                # 32: right_elbow
                # 33: right_wrist_pitch
                # 34: right_wrist_roll
                # 35: right_wrist_yaw
                
                # Keep left arm in neutral position
                qpos[15:22] = 0.0  # All joints at zero
                
                # Simple oscillating motion where only the shoulder moves
                shoulder_angle = np.sin(t * 0.01) * 0.5  # Oscillates between -0.5 and 0.5
                
                # Modify right arm joints directly
                qpos[29] = shoulder_angle  # shoulder_pitch: oscillate
                qpos[30] = -0.5   # shoulder_roll: fixed
                
                # Step simulation
                sim.step(qpos, head_rot, viewer)
                
                if t % 100 == 0:
                    print(f"Step {t}: right_shoulder_pitch = {qpos[29]:.3f}")
                    
        except KeyboardInterrupt:
            print("\nTest stopped by user")
    
    print("Test completed")

if __name__ == "__main__":
    main()
