"""
Forward Kinematics (FK) Calculator for Piper-X Robot Arm

This module provides a class to compute the end-effector (flange) pose 
from joint angles using the Modified Denavit-Hartenberg (MDH) model.

Usage:
    fk_calc = PiperXForwardKinematics()
    joint_angles = [0.0, 1.57, -1.57, 0.0, 0.0, 0.0]  # in radians
    pose = fk_calc.compute_fk(joint_angles)
    print(pose)  # [x, y, z, roll, pitch, yaw]
"""

import numpy as np
from typing import List, Tuple, Union


class PiperXForwardKinematics:
    """
    Forward Kinematics calculator for Piper-X robot arm using Modified DH parameters.
    
    The Piper-X is a 6-DOF collaborative robot arm. This class computes the 
    end-effector flange pose in the base frame given the joint angles.
    
    Coordinate System:
    - Base frame: origin at the base of the robot
    - Flange frame: mounted on the end-effector
    - Pose representation: [x, y, z, roll, pitch, yaw]
      - Position (x, y, z): in meters
      - Orientation (roll, pitch, yaw): in radians (ZYX RPY convention)
    """
    
    def __init__(self):
        """
        Initialize the Forward Kinematics calculator with Piper-X MDH parameters.
        
        Modified DH Parameters Format: (d, a, alpha, theta_offset)
        - d: Link offset (meters)
        - a: Link length (meters)  
        - alpha: Link twist (radians)
        - theta_offset: Joint angle offset (radians)
        """
        # Official pyAgxArm ROBOT_MDH_PRESET["piper_x"] (meters/radians).
        # Keep the SDK's d6=35.26 mm; the supplied URDF uses 35 mm instead.
        self.mdh_params = [
            (0.123, 0.0, 0.0, 3.141592653589793),
            (0.0, 0.0, 1.5707963267948966, 0.13578661580515886),
            (0.0, 0.28502999999999995, 0.0, 2.8380798966679794),
            (0.0, 0.27364, 0.0, 0.08063421144213803),
            (0.0, 0.07465999999999999, -1.5707963267948966, 1.5707963267948966),
            (0.03526, 0.0, 1.5707963267948966, 0.0),
        ]
        
        # Robot model name and number of joints
        self.model_name = "piper_x"
        self.num_joints = 6
        
    def _mdh_transform(self, 
                       d: float, 
                       a: float, 
                       alpha: float, 
                       theta: float) -> np.ndarray:
        """
        Compute the transformation matrix from Modified DH parameters.
        
        The transformation matrix represents the pose of frame i relative to frame i-1.
        
        Parameters
        ----------
        d : float
            Link offset (meters)
        a : float
            Link length (meters)
        alpha : float
            Link twist (radians)
        theta : float
            Joint angle (radians) - actual joint angle + theta_offset
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix
            
        Notes
        -----
        Modified DH convention: Rx(alpha) Tx(a) Rz(theta) Tz(d),
        matching pyAgxArm.utiles.mdh_kinematics.
        
        T = |    cos(θ)       -sin(θ)       0          a      |
            | sin(θ)cos(α)  cos(θ)cos(α)  -sin(α)  -d·sin(α) |
            | sin(θ)sin(α)  cos(θ)sin(α)   cos(α)   d·cos(α) |
            |       0             0          0          1      |
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Build the 4x4 transformation matrix
        T = np.array([
            [cos_theta, -sin_theta, 0.0, a],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d * sin_alpha],
            [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d * cos_alpha],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        return T
    
    def _rotation_matrix_to_rpy(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a 3x3 rotation matrix to Roll-Pitch-Yaw angles.
        
        Uses the ZYX (yaw-pitch-roll) convention as per the SDK specification.
        
        Parameters
        ----------
        R : np.ndarray
            3x3 rotation matrix
            
        Returns
        -------
        Tuple[float, float, float]
            (roll, pitch, yaw) in radians
            
        Notes
        -----
        ZYX Euler angle extraction:
        - Pitch angle is extracted from R[2, 0]
        - For pitch in range (-π/2, π/2), unique roll and yaw are computed
        - Singularity handling: when pitch ≈ ±π/2, gimbal lock occurs
        """
        # Extract pitch
        sin_pitch = -R[2, 0]
        # Clamp to [-1, 1] to avoid numerical issues with arcsin
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        
        # Check for gimbal lock (singularity at pitch = ±π/2)
        cos_pitch = np.cos(pitch)
        
        if abs(cos_pitch) < 1e-9:
            # Gimbal lock: set roll to 0 and solve for yaw
            roll = 0.0
            # The same expression applies at both signs of pitch.
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        else:
            # Normal case: extract roll and yaw
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return roll, pitch, yaw
    
    def compute_fk(self, 
                   joint_angles: Union[List[float], np.ndarray]) -> List[float]:
        """
        Compute the forward kinematics for the Piper-X robot.
        
        Calculates the end-effector (flange) pose in the base coordinate frame
        given the joint angles using the modified DH model.
        
        Parameters
        ----------
        joint_angles : Union[List[float], np.ndarray]
            Joint angles in radians. Should be a list or array of length 6:
            [j1, j2, j3, j4, j5, j6]
            
        Returns
        -------
        List[float]
            End-effector pose in the base frame:
            [x, y, z, roll, pitch, yaw]
            - Position (x, y, z): in meters
            - Orientation (roll, pitch, yaw): in radians
            
        Raises
        ------
        ValueError
            If joint_angles length is not 6
            
        Examples
        --------
        >>> fk = PiperXForwardKinematics()
        >>> joint_angles = [0.0, 1.57, -1.57, 0.0, 0.0, 0.0]
        >>> pose = fk.compute_fk(joint_angles)
        >>> print(f"Position: x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}")
        >>> print(f"Orientation: roll={pose[3]:.3f}, pitch={pose[4]:.3f}, yaw={pose[5]:.3f}")
        """
        # Convert to numpy array if necessary
        joint_angles = np.array(joint_angles, dtype=np.float64)
        
        # Validate input
        if len(joint_angles) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joint angles, got {len(joint_angles)}"
            )
        
        # Initialize transformation matrix to identity (base frame)
        T = np.eye(4, dtype=np.float64)
        
        # Compute cumulative transformation for each joint
        for i in range(self.num_joints):
            d, a, alpha, theta_offset = self.mdh_params[i]
            
            # Apply joint angle with offset
            theta = joint_angles[i] + theta_offset
            
            # Compute transformation matrix for this joint
            T_i = self._mdh_transform(d, a, alpha, theta)
            
            # Accumulate transformation
            T = T @ T_i
        
        # Extract position from transformation matrix
        position = T[:3, 3]
        
        # Extract rotation matrix
        rotation_matrix = T[:3, :3]
        
        # Convert rotation matrix to RPY angles
        roll, pitch, yaw = self._rotation_matrix_to_rpy(rotation_matrix)
        
        # Combine position and orientation
        pose = [
            float(position[0]),  # x
            float(position[1]),  # y
            float(position[2]),  # z
            float(roll),         # roll (radians)
            float(pitch),        # pitch (radians)
            float(yaw)           # yaw (radians)
        ]
        
        return pose
    
    def compute_fk_matrix(self, 
                          joint_angles: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute the forward kinematics and return the full transformation matrix.
        
        Parameters
        ----------
        joint_angles : Union[List[float], np.ndarray]
            Joint angles in radians [j1, j2, j3, j4, j5, j6]
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix representing the end-effector
            pose relative to the base frame
            
        Examples
        --------
        >>> fk = PiperXForwardKinematics()
        >>> joint_angles = [0.0, 1.57, -1.57, 0.0, 0.0, 0.0]
        >>> T = fk.compute_fk_matrix(joint_angles)
        >>> print(T)
        """
        joint_angles = np.array(joint_angles, dtype=np.float64)
        
        if len(joint_angles) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joint angles, got {len(joint_angles)}"
            )
        
        T = np.eye(4, dtype=np.float64)
        
        for i in range(self.num_joints):
            d, a, alpha, theta_offset = self.mdh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self._mdh_transform(d, a, alpha, theta)
            T = T @ T_i
        
        return T
    
    def get_mdh_parameters(self) -> List[Tuple[float, float, float, float]]:
        """
        Get the MDH parameters used for this robot model.
        
        Returns
        -------
        List[Tuple[float, float, float, float]]
            List of MDH parameters for each joint: (d, a, alpha, theta_offset)
        """
        return self.mdh_params.copy()


def degrees_to_radians(angles_deg: List[float]) -> List[float]:
    """
    Convert joint angles from degrees to radians.
    
    Parameters
    ----------
    angles_deg : List[float]
        Angles in degrees
        
    Returns
    -------
    List[float]
        Angles in radians
    """
    return [angle * np.pi / 180.0 for angle in angles_deg]


def radians_to_degrees(angles_rad: List[float]) -> List[float]:
    """
    Convert joint angles from radians to degrees.
    
    Parameters
    ----------
    angles_rad : List[float]
        Angles in radians
        
    Returns
    -------
    List[float]
        Angles in degrees
    """
    return [angle * 180.0 / np.pi for angle in angles_rad]


if __name__ == "__main__":
    """
    Example usage and testing of the Forward Kinematics calculator.
    """
    
    # Create FK calculator instance
    fk = PiperXForwardKinematics()
    
    print("=" * 70)
    print("Piper-X Forward Kinematics Calculator")
    print("=" * 70)
    
    # Example 1: Home position (all zeros)
    print("\n--- Example 1: Home Position (all zeros) ---")
    joint_angles_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pose_1 = fk.compute_fk(joint_angles_1)
    print(f"Joint Angles (rad): {joint_angles_1}")
    print(f"Flange Pose: x={pose_1[0]:.4f}, y={pose_1[1]:.4f}, z={pose_1[2]:.4f}")
    print(f"Orientation:  roll={pose_1[3]:.4f}, pitch={pose_1[4]:.4f}, yaw={pose_1[5]:.4f}")
    
    # Example 2: Arbitrary joint configuration
    print("\n--- Example 2: Arbitrary Configuration ---")
    joint_angles_2 = [0.5, 1.57, -1.57, 0.3, 0.2, 0.1]
    pose_2 = fk.compute_fk(joint_angles_2)
    print(f"Joint Angles (rad): {joint_angles_2}")
    print(f"Flange Pose: x={pose_2[0]:.4f}, y={pose_2[1]:.4f}, z={pose_2[2]:.4f}")
    print(f"Orientation:  roll={pose_2[3]:.4f}, pitch={pose_2[4]:.4f}, yaw={pose_2[5]:.4f}")
    
    # Example 3: Using degrees as input
    print("\n--- Example 3: Using Degrees as Input ---")
    joint_angles_deg = [30.0, 90.0, -90.0, 15.0, 10.0, 5.0]
    joint_angles_rad = degrees_to_radians(joint_angles_deg)
    pose_3 = fk.compute_fk(joint_angles_rad)
    print(f"Joint Angles (deg): {joint_angles_deg}")
    print(f"Joint Angles (rad): {[f'{x:.4f}' for x in joint_angles_rad]}")
    print(f"Flange Pose: x={pose_3[0]:.4f}, y={pose_3[1]:.4f}, z={pose_3[2]:.4f}")
    print(f"Orientation:  roll={pose_3[3]:.4f}, pitch={pose_3[4]:.4f}, yaw={pose_3[5]:.4f}")
    
    # Example 4: Get transformation matrix
    print("\n--- Example 4: Full Transformation Matrix ---")
    T = fk.compute_fk_matrix([0.0, 1.57, -1.57, 0.0, 0.0, 0.0])
    print("Transformation Matrix T (base to flange):")
    print(T)
    
    # Example 5: MDH Parameters
    print("\n--- Example 5: MDH Parameters ---")
    mdh = fk.get_mdh_parameters()
    print("Modified DH Parameters (d, a, alpha, theta_offset):")
    for i, (d, a, alpha, theta_offset) in enumerate(mdh):
        print(f"Joint {i+1}: d={d:.6f}, a={a:.6f}, alpha={alpha:.6f}, theta_offset={theta_offset:.6f}")
    
    print("\n" + "=" * 70)
