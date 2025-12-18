# shadow_hand_open_bottle
Migrate shadow hand open lit task from IsaacGym (https://github.com/RobotEmperor/SR-Tac2Motion/tree/ver_1) to IsaacLab based on tutorial in (https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)

# General info
Project is created with:
* Isaac Lab (https://isaac-sim.github.io/IsaacLab/main/index.html)
* SKRL 2.0.0 (https://skrl.readthedocs.io/en/latest/)
  
# Setup


# Main changes
1. **`create_sim`** change to **`_setup_scene`**.
2. **`pre_physics_step`** change to **`pre_physics_step`** and **`_apply_action`**.
3. **`post_physics_step`** removed and includes **`_get_dones`**, **`_reset_idx`**, **`_get_rewards`** and **_get_observations`**.
4. No need to refresh states.
5. **`compute_observations`** change to **`_get_observations`** and **`compute_observations`**.
6. **`compute_rewards`** change to **`_get_rewards`**, and removed early termination conditions to new function **`_get_dones`**.
7. **`reset_idx`** change to **`_get_dones`** and **`_reset_idx`**.
8. **`initializer`** and **`randomizer`** are removed and setup in the env_config as **`eventCFG`**, **`rewarder`** is kept.
9. Shadow hand asset is replaced with the one in IsaacLab, all assets should be convert to usd fomat.
10. Isaac Lab no longer requires pre-initialization of buffers through the **acquire_*** API.
11. The **progress_buf** variable has also been renamed to **episode_length_buf**.


