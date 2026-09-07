# Autoregressive Rollout Report

This report compares the ground truth simulation with closed-loop (forced actions) and open-loop (imagined actions) autoregressive rollouts.

## 2D Trajectory Map (All Ships)
![Trajectory Map](2d_map.png)

## 2D Trajectory Map (Featured Ships, Centered)
![Trajectory Map Ship 0](2d_map_ship0.png)

## 2D Velocity Space Map
![Velocity Space](2d_vel_map.png)

## Error Metrics Over Time
Calculated only while both the ground truth and rollout ships are alive.

### Position Error (Toroidal L2)
![Position Error (Toroidal L2)](mae_position.png)

### Velocity Error (L2)
![Velocity Error (L2)](mae_velocity.png)

### Pos+Vel 4D Error (L2)
![Pos+Vel 4D Error (L2)](mae_pos_vel_4d.png)

### Attitude Error (MAE)
![Attitude Error (MAE)](mae_attitude.png)

### Health Error (MAE)
![Health Error (MAE)](mae_health.png)

### Power Error (MAE)
![Power Error (MAE)](mae_power.png)

## Feature Divergence
### Position X
![Position X](feature_position_x.png)

### Velocity X
![Velocity X](feature_velocity_x.png)

### Angle (cos)
![Angle (cos)](feature_angle_cos.png)

### Angular Vel
![Angular Vel](feature_angular_vel.png)

### Angular Vel (Scaled to GT)
![Angular Vel (Scaled to GT)](feature_angular_vel_scaled.png)

### Health
![Health](feature_health.png)

### Power
![Power](feature_power.png)

### Cooldown
![Cooldown](feature_cooldown.png)

### Alive Prob
![Alive Prob](feature_alive.png)
