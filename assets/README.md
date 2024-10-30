# Climber
## Description
`climber_v0.xml` is modified from `humanoid_v5.xml` to include climbing shoes and fingers.

It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
The goal of the environment is to walk forward as fast as possible without falling over.

## Action Space
The action space is a `Box(-0.4, 0.4, (17,), float32)`. An action represents the torques applied at the hinge joints.
| Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
| --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
| 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
| 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
| 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
| 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
| 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
| 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
| 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
| 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
| 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
| 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
| 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
| 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
| 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
| 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
| 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |

## Observation Space
The observation space consists of the following parts (in order)

- *qpos (22 elements by default):* The position values of the robot's body parts.
- *qvel (23 elements):* The velocities of these individual body parts (their derivatives).
- *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
(this is an intermediate result of the transition).
It has shape 13*10 (*nbody * 10*).
(cinert - inertia matrix and body mass offset and body mass)
- *cvel (78 elements):* Center of mass based velocity.
It has shape 13 * 6 (*nbody * 6*).
(com velocity - velocity x, y, z and angular velocity x, y, z)
- *qfrc_actuator (17 elements):* Constraint force generated as the actuator force at each joint.
This has shape `(17,)`  *(nv * 1)*.
- *cfrc_ext (78 elements):* This is the center of mass based external force on the body parts.
It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
(external forces - force x, y, z and torque x, y, z)

where *nbody* is the number of bodies in the robot,
and *nv* is the number of degrees of freedom (*= dim(qvel)*).

By default, the observation does not include the x- and y-coordinates of the torso.
These can be included by passing `exclude_current_positions_from_observation=False` during construction.
In this case, the observation space will be a `Box(-Inf, Inf, (350,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

By default, however, the observation space is a `Box(-Inf, Inf, (348,), float64)`, where the position and velocity elements are as follows:

| Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
| --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
| 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
| 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
| 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
| 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
| 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
| 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
| 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
| 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
| 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
| 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
| 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
| 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
| 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
| 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
| 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
| 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
| 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
| 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
| 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | angular velocity (rad/s)   |
| 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | angular velocity (rad/s)   |
| 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | angular velocity (rad/s)   |
| 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | angular velocity (rad/s)   |
| 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | angular velocity (rad/s)   |
| 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | angular velocity (rad/s)   |
| 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | angular velocity (rad/s)   |
| 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | angular velocity (rad/s)   |
| 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | angular velocity (rad/s)   |
| 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | angular velocity (rad/s)   |
| 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | angular velocity (rad/s)   |
| 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | angular velocity (rad/s)   |
| 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | angular velocity (rad/s)   |
| 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | angular velocity (rad/s)   |
| 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | angular velocity (rad/s)   |
| 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | angular velocity (rad/s)   |
| 44  | angular velocity of the angle between left upper arm and left_lower_arm   


## Body parts
| body part       | id (for `v2`, `v3`, `v4)`) | id (for `v5`) |
|  -------------  |  ---   |  ---  |
| torso           |1  | 0      |
| lwaist          |2  | 1      |
| pelvis          |3  | 2      |
| right_thigh     |4  | 3      |
| right_sin       |5  | 4      |
| right_foot      |6  | 5      |
| left_thigh      |7  | 6      |
| left_sin        |8  | 7      |
| left_foot       |9  | 8      |
| right_upper_arm |10 | 9      |
| right_lower_arm |11 | 10     |
| left_upper_arm  |12 | 11     |
| left_lower_arm  |13 | 12     |


## Joints
| joint           | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
|  -------------  |  ---   |  ---  |
| root (note: all values are constant 0) | 0  |excluded|
| root (note: all values are constant 0) | 1  |excluded|
| root (note: all values are constant 0) | 2  |excluded|
| root (note: all values are constant 0) | 3  |excluded|
| root (note: all values are constant 0) | 4  |excluded|
| root (note: all values are constant 0) | 5  |excluded|
| abdomen_z       | 6  | 0      |
| abdomen_y       | 7  | 1      |
| abdomen_x       | 8  | 2      |
| right_hip_x     | 9  | 3      |
| right_hip_z     | 10 | 4      |
| right_hip_y     | 11 | 5      |
| right_knee      | 12 | 6      |
| left_hip_x      | 13 | 7      |
| left_hiz_z      | 14 | 8      |
| left_hip_y      | 15 | 9      |
| left_knee       | 16 | 10     |
| right_shoulder1 | 17 | 11     |
| right_shoulder2 | 18 | 12     |
| right_elbow     | 19 | 13     |
| left_shoulder1  | 20 | 14     |
| left_shoulder2  | 21 | 15     |
| left_elfbow     | 22 | 16     |

The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

## Rewards
The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.

- *healthy_reward*:
Every timestep that the Humanoid is alive (see definition in section "Episode End"),
it gets a reward of fixed value `healthy_reward` (default is $5$).
- *forward_reward*:
A reward for moving forward,
this reward would be positive if the Humanoid moves forward (in the positive $x$ direction / in the right direction).
$w_{forward} \times \frac{dx}{dt}$, where
$dx$ is the displacement of the center of mass ($x_{after-action} - x_{before-action}$),
$dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
and `frametime` which is $0.001$ - so the default is $dt = 5 \times 0.003 = 0.015$,
$w_{forward}$ is the `forward_reward_weight` (default is $1.25$).
- *ctrl_cost*:
A negative reward to penalize the Humanoid for taking actions that are too large.
$w_{control} \times \|action\|_2^2$,
where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
- *contact_cost*:
A negative reward to penalize the Humanoid if the external contact forces are too large.
$w_{contact} \times clamp(contact\_cost\_range, \|F_{contact}\|_2^2)$, where
$w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-7}$),
$F_{contact}$ are the external contact forces (see `cfrc_ext` section on observation).

`info` contains the individual reward terms.


## Starting State
The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{24}, reset\_noise\_scale \times I_{24}]}$.
The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{23}, reset\_noise\_scale \times I_{23}]}$.

where $\mathcal{U}$ is the multivariate uniform continuous distribution.

Note that the z- and x-coordinates are non-zero so that the humanoid can immediately stand up and face forward (x-axis).


## Episode End
### Termination
If `terminate_when_unhealthy is True` (the default), the environment terminates when the Humanoid is unhealthy.
The Humanoid is said to be unhealthy if any of the following happens:

1. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[1.0, 2.0]$).

### Truncation
The default duration of an episode is 1000 timesteps.