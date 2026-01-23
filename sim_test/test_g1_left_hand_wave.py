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
    
    print("Starting simulation with left hand waving test...")
    print("G1 left arm joint indices: 15-21 (7 DOF)")
    print("G1 left hand joint indices: 22-28 (7 DOF)")
    print("Joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw")
    
    # Get initial state (all zeros for standing still)
    dummy_action = np.zeros(128, dtype=np.float32)
    qpos, head_rot = sim.generate_qpos(dummy_action, init_state=True)
    
    # Set right arm (non-active) to neutral position
    # Right arm indices: 29-35 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw)
    qpos[29:36] = 0.0  # All joints at zero
    
    # Launch viewer
    with mj.viewer.launch_passive(sim.model, sim.data) as viewer:
        sim.setup_viewer(viewer)
        
        print("\nRunning test: Left hand waving motion")
        print("Right arm set to neutral position (all zeros)")
        print("Press Ctrl+C to stop\n")
        
        try:
            for t in range(10000):  # Run for a long time
                # Create waving motion for left arm/hand
                # Left arm indices: 15-21
                # 15: left_shoulder_pitch
                # 16: left_shoulder_roll  
                # 17: left_shoulder_yaw
                # 18: left_elbow
                # 19: left_wrist_pitch
                # 20: left_wrist_roll
                # 21: left_wrist_yaw
                
                # Keep right arm in neutral position
                qpos[29:36] = 0.0  # All joints at zero
                
                # Waving motion: arm moves side to side (shoulder_yaw) and up/down (shoulder_pitch)
                # Create a smooth waving pattern
                wave_cycle = t * 0.02  # Slower frequency for waving
                
                # Shoulder pitch: raise arm up (negative is up for most robots)
                shoulder_pitch = -0.3 + np.sin(wave_cycle) * 0.2  # Oscillates around -0.3
                
                # Shoulder yaw: move arm side to side (waving motion)
                shoulder_yaw = np.sin(wave_cycle * 2) * 0.6  # Wider side-to-side motion
                
                # Elbow: keep slightly bent for natural waving
                elbow = -0.8 + np.sin(wave_cycle * 1.5) * 0.2  # Slight variation
                
                # Wrist pitch: add some wrist movement for more natural wave
                wrist_pitch = np.sin(wave_cycle * 2.5) * 0.3
                
                # Modify left arm joints directly
                qpos[15] = shoulder_pitch  # shoulder_pitch: up/down motion
                qpos[16] = -0.3   # shoulder_roll: fixed at slight angle
                qpos[17] = shoulder_yaw  # shoulder_yaw: side-to-side waving
                qpos[18] = elbow  # elbow: slightly bent with variation
                qpos[19] = wrist_pitch  # wrist_pitch: wrist movement
                
                # Step simulation
                sim.step(qpos, head_rot, viewer)
                
                if t % 100 == 0:
                    print(f"Step {t}: left_shoulder_pitch = {qpos[15]:.3f}, left_shoulder_yaw = {qpos[17]:.3f}")
                    
        except KeyboardInterrupt:
            print("\nTest stopped by user")
    
    print("Test completed")

if __name__ == "__main__":
    main()
