""" Test Dual Quaternion related utility functions. """
from __future__ import annotations

import numpy as np
from dqrobotics import DQ
from spatialmath import Quaternion, UnitDualQuaternion

from screwmpc_experiments.experiments.screwmpc import (
    dq_sclerp,
    dq_to_pose,
    generate_intermediate_waypoints,
    generate_random_poses,
    pose_to_dq,
)

# setup boundaries for robot EE pose.
dp = [0.1, 0.3, 0.3]
dr = [45, 45, 45]

min_pose_bounds = np.array(
    [
        0.307 - dp[0],
        0 - dp[1],
        0.487 + 0.1034 - dp[2],
        np.pi - dr[0],
        0 - dr[1],
        0.25 * np.pi - dr[2],
    ]
)

max_pose_bounds = np.array(
    [
        0.307 + dp[0],
        0 + dp[1],
        0.487 + 0.1034 + dp[2],
        np.pi + dr[0],
        0 + dr[1],
        0.25 * np.pi + dr[2],
    ]
)

def test_dq_sclerp() -> None:
    """Test scLERP for unit Dual Quternions"""

    rng = np.random.RandomState()

    initial_pose, goal_pose = (
        pose_to_dq(pose)
        for pose in generate_random_poses(2, min_pose_bounds, max_pose_bounds, rng)
    )
    inter_pose = dq_sclerp(initial_pose, goal_pose, 0)
    assert inter_pose == initial_pose
    inter_pose = dq_sclerp(initial_pose, goal_pose, 1)
    assert inter_pose == goal_pose


def test_generate_waypoints() -> None:
    """Test the generated waypoints."""

    rng = np.random.RandomState()

    initial_pose, goal_pose = (
        pose_to_dq(pose)
        for pose in generate_random_poses(2, min_pose_bounds, max_pose_bounds, rng)
    )
    waypoints = generate_intermediate_waypoints(initial_pose, goal_pose, 10)

    assert len(waypoints) == 10

    steps = np.linspace(0, 1, len(waypoints))

    for step, waypoint in zip(steps, waypoints):
        inter_pose = dq_sclerp(initial_pose, goal_pose, step)
        assert inter_pose == waypoint


def test_dq_to_pose() -> None:
    """Test retrieved poses"""

    rng = np.random.RandomState()
    pose = generate_random_poses(1, min_pose_bounds, max_pose_bounds, rng)[0]
    primal = Quaternion(pose[1])
    dual = 0.5 * (Quaternion(0, pose[0]) * primal)
    dq = DQ(UnitDualQuaternion(primal, dual).vec)

    _pose = dq_to_pose(dq)

    # numerical issues can lead to inconsistencies.
    np.testing.assert_array_almost_equal(pose[1], _pose[1])
    np.testing.assert_array_almost_equal(pose[0], _pose[0])
