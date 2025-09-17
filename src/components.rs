use bevy::prelude::*;

// Ship parameters from requirements.md
#[derive(Component, Debug, Clone)]
pub struct ShipParameters {
    // Physics parameters
    pub thrust_base: f32,
    pub forward_boost_multiplier: f32,
    pub backward_boost_multiplier: f32,
    pub max_boost_energy: f32,
    pub base_energy_cost: f32,
    pub forward_energy_cost: f32,
    pub backward_energy_cost: f32,

    // Maneuvering parameters
    pub no_turn_drag_coefficient: f32,
    pub normal_turn_drag_coefficient: f32,
    pub sharp_turn_drag_coefficient: f32,
    pub normal_turn_lift_coefficient: f32,
    pub sharp_turn_lift_coefficient: f32,
    pub normal_turn_angle: f32,    // in radians
    pub sharp_turn_angle: f32,     // in radians

    // Physical properties
    pub collision_radius: f32,
    pub max_health: f32,
}

impl Default for ShipParameters {
    fn default() -> Self {
        Self {
            // Physics parameters (default values from requirements.md)
            thrust_base: 10.0,
            forward_boost_multiplier: 5.0,
            backward_boost_multiplier: 0.5,
            max_boost_energy: 100.0,
            base_energy_cost: -10.0,    // negative = regeneration
            forward_energy_cost: 50.0,
            backward_energy_cost: -20.0,

            // Maneuvering parameters (default values)
            no_turn_drag_coefficient: 8e-4,
            normal_turn_drag_coefficient: 1e-3,
            sharp_turn_drag_coefficient: 3e-3,
            normal_turn_lift_coefficient: 15e-3,
            sharp_turn_lift_coefficient: 30e-3,
            normal_turn_angle: 5.0_f32.to_radians(),   // 5 degrees
            sharp_turn_angle: 15.0_f32.to_radians(),   // 15 degrees

            // Physical properties
            collision_radius: 10.0,
            max_health: 100.0,
        }
    }
}

// Ship state component
#[derive(Component, Debug)]
pub struct Ship {
    pub id: u32,
    pub parameters: ShipParameters,
    
    // Physics state
    pub velocity: Vec2,
    pub turn_offset: f32,    // radians - key arcade maneuvering feature
    
    // Resources
    pub energy: f32,
    pub health: f32,
    pub ammo: f32,
    
    // Action hold system (10Hz -> 5 frames at 50Hz)
    pub current_actions: ShipActions,
    pub action_hold_frames_remaining: u32,
}

// Ship actions - 6 binary actions as per requirements
#[derive(Debug, Clone, Default)]
pub struct ShipActions {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub sharp_turn: bool,
    pub shoot: bool,
}

impl Ship {
    pub fn new(id: u32) -> Self {
        let parameters = ShipParameters::default();
        Self {
            id,
            parameters: parameters.clone(),
            velocity: Vec2::ZERO,
            turn_offset: 0.0,
            energy: parameters.max_boost_energy,
            health: parameters.max_health,
            ammo: 32.0,    // max_ammo from requirements.md
            current_actions: ShipActions::default(),
            action_hold_frames_remaining: 0,
        }
    }
    
    pub fn update_actions(&mut self, actions: ShipActions, hold_frames: u32) {
        self.current_actions = actions;
        self.action_hold_frames_remaining = hold_frames;
    }
    
    pub fn tick_action_hold(&mut self) {
        if self.action_hold_frames_remaining > 0 {
            self.action_hold_frames_remaining -= 1;
        }
        
        // Clear actions when hold expires
        if self.action_hold_frames_remaining == 0 {
            self.current_actions = ShipActions::default();
        }
    }
}

// Projectile parameters from requirements.md
#[derive(Component, Debug)]
pub struct Projectile {
    pub owner_id: u32,
    pub velocity: Vec2,
    pub lifetime_frames_remaining: u32,
    pub damage: f32,
}

impl Projectile {
    pub fn new(owner_id: u32, _position: Vec2, direction: Vec2) -> Self {
        let speed = 500.0;  // projectile_speed from requirements.md
        let lifetime_frames = 50;  // 1.0 second * 50 Hz from requirements.md
        let damage = 20.0;  // projectile_damage from requirements.md
        
        Self {
            owner_id,
            velocity: direction.normalize() * speed,
            lifetime_frames_remaining: lifetime_frames,
            damage,
        }
    }
    
    pub fn tick_lifetime(&mut self) -> bool {
        if self.lifetime_frames_remaining > 0 {
            self.lifetime_frames_remaining -= 1;
            true  // still alive
        } else {
            false // expired
        }
    }
}

// Resource for managing game state
#[derive(Resource, Debug)]
pub struct GameState {
    pub frame_count: u64,
    pub agent_decision_frame_interval: u32,  // 5 frames = 100ms at 50Hz
    
    // World parameters
    pub world_width: f32,
    pub world_height: f32,
    
    // Ship management
    pub next_ship_id: u32,
    
    // Projectile pooling
    pub max_projectiles_per_ship: u32,
    pub projectile_pools: Vec<Vec<Entity>>,  // Pool per ship
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            frame_count: 0,
            agent_decision_frame_interval: 5,  // 10Hz decisions = every 5 frames at 50Hz
            world_width: 1200.0,
            world_height: 800.0,
            next_ship_id: 0,
            max_projectiles_per_ship: 16,  // from requirements.md
            projectile_pools: Vec::new(),
        }
    }
}

impl GameState {
    pub fn is_agent_decision_frame(&self) -> bool {
        self.frame_count % (self.agent_decision_frame_interval as u64) == 0
    }
    
    pub fn increment_frame(&mut self) {
        self.frame_count += 1;
    }
    
    pub fn allocate_ship_id(&mut self) -> u32 {
        let id = self.next_ship_id;
        self.next_ship_id += 1;
        
        // Ensure projectile pool exists for this ship
        while self.projectile_pools.len() <= id as usize {
            self.projectile_pools.push(Vec::new());
        }
        
        id
    }
}

// Marker component for controllable ships (only ship 0 for Phase 1)
#[derive(Component)]
pub struct PlayerControlled;

// Marker component for rendering
#[derive(Component)]
pub struct Renderable {
    pub color: Color,
    pub size: f32,
}

impl Renderable {
    pub fn ship(id: u32) -> Self {
        let colors = [
            Color::srgb(0.0, 0.0, 1.0),  // Blue
            Color::srgb(1.0, 0.0, 0.0),  // Red
            Color::srgb(0.0, 1.0, 0.0),  // Green
            Color::srgb(1.0, 1.0, 0.0),  // Yellow
        ];
        Self {
            color: colors[id as usize % colors.len()],
            size: 20.0,
        }
    }
    
    pub fn projectile() -> Self {
        Self {
            color: Color::srgb(1.0, 1.0, 1.0),  // White
            size: 2.0,
        }
    }
}