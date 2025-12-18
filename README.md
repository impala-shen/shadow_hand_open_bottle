# shadow_hand_open_bottle
Migrate shadow hand open lit task from IsaacGym tn IsaacLab based on tutorial in (https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)

# General info
Project is created with:
* Isaac Lab (https://isaac-sim.github.io/IsaacLab/main/index.html)
* SKRL 2.0.0 (https://skrl.readthedocs.io/en/latest/)
  
# Setup


# Main changes
1. **`create_sim`** change to **`_setup_scene`**
2. **`pre_physics_step`** change to **`pre_physics_step`** and **`_apply_action`**
3. **`post_physics_step`** removed
4. No need to refresh states
5. **`compute_observations`** change to **`_get_observations`** and **`compute_observations`**
6. **`compute_rewards`** change to **`_get_rewards`**, and removed early termination conditions to new function **`_get_dones`**
7. **`reset_idx`** change to **`_get_dones`** and **`idx`**
8. **`initializer`** and **`randomizer`** are removed and setup in the env_config, **`rewarder`** is kept.
9. Shadow hand asset is replaced with the one in IsaacLab


