"""
Centralized feature definitions and normalization strategies for the Boost and Broadside project.
"""
from boost_and_broadside.core.constants import StateFeature, TargetFeature

# Ego State: [HEALTH, POWER, VX, VY, ANG_VEL]
EGO_STATE_RESOURCES = ["State_HEALTH", "State_POWER"]
EGO_STATE_DYNAMICS = ["State_VX", "State_VY", "State_ANG_VEL"]

EGO_STATE_FIELDS = [
    ("State_HEALTH", "Min-Max"),
    ("State_POWER", "Min-Max"),
    ("State_VX", "Scale"),
    ("State_VY", "Scale"),
    ("State_ANG_VEL", "Scale"),
]

# Targets: [DX, DY, DVX, DVY, DHEALTH, DPOWER, DANG_VEL]
TARGET_FIELDS = [
    ("Target_DX", "Scale"),
    ("Target_DY", "Scale"),
    ("Target_DVX", "Scale"),
    ("Target_DVY", "Scale"),
    ("Target_DHEALTH", "Scale"),
    ("Target_DPOWER", "Scale"),
    ("Target_DANG_VEL", "Scale"),
]

# Relational: [dx, dy, dvx, dvy, dist, inv_dist, rel_speed, closing, dir_x, dir_y, log_dist, tti, cos_ata, sin_ata, cos_aa, sin_aa, cos_hca, sin_hca]
RELATIONAL_FIELDS = [
    ("Relational_dx", "Scale"),
    ("Relational_dy", "Scale"),
    ("Relational_dvx", "Scale"),
    ("Relational_dvy", "Scale"),
    ("Relational_dist", "Scale"),
    ("Relational_inv_dist", "Scale"),
    ("Relational_rel_speed", "Scale"),
    ("Relational_closing", "Scale"),
    ("Relational_dir_x", "Identity"),
    ("Relational_dir_y", "Identity"),
    ("Relational_log_dist", "Z-Score"),
    ("Relational_tti", "Identity"),
    ("Relational_cos_ata", "Identity"),
    ("Relational_sin_ata", "Identity"),
    ("Relational_cos_aa", "Identity"),
    ("Relational_sin_aa", "Identity"),
    ("Relational_cos_hca", "Identity"),
    ("Relational_sin_hca", "Identity"),
]
