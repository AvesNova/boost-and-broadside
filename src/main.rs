use bevy::prelude::*;

mod components;
use components::*;

// Constants for dual-rate system
const PHYSICS_TIMESTEP: f64 = 1.0 / 50.0; // 50 Hz physics (20ms)
const AGENT_TIMESTEP: f64 = 1.0 / 10.0; // 10 Hz agent decisions (100ms)
const ACTION_HOLD_FRAMES: u32 = 5; // Hold actions for 5 physics frames

// World parameters
const WORLD_WIDTH: f32 = 1200.0;
const WORLD_HEIGHT: f32 = 800.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Boost & Broadside - Naval Combat Simulation".into(),
                resolution: (WORLD_WIDTH, WORLD_HEIGHT).into(),
                resizable: false,
                ..default()
            }),
            ..default()
        }))
        .init_resource::<GameState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            // All systems for now - we'll add fixed timestep later
            physics_system,
            collision_system,
            projectile_system,
            render_system,
        ))
        .run();
}

fn setup(mut commands: Commands, mut game_state: ResMut<GameState>) {
    // Setup camera
    commands.spawn(Camera2dBundle::default());
    
    // Spawn 4 ships at different positions
    let ship_positions = [
        Vec3::new(-300.0, 0.0, 0.0),    // Ship 0 - Player controlled
        Vec3::new(300.0, 0.0, 0.0),     // Ship 1 - AFK
        Vec3::new(0.0, -200.0, 0.0),    // Ship 2 - AFK
        Vec3::new(0.0, 200.0, 0.0),     // Ship 3 - AFK
    ];
    
    for (i, position) in ship_positions.iter().enumerate() {
        let ship_id = game_state.allocate_ship_id();
        let ship = Ship::new(ship_id);
        
        let mut entity = commands.spawn((
            ship,
            Transform::from_translation(*position),
            Renderable::ship(ship_id),
        ));
        
        // Only ship 0 is player controlled
        if i == 0 {
            entity.insert(PlayerControlled);
            println!("Spawned player-controlled ship {} at {:?}", ship_id, position);
        } else {
            println!("Spawned AFK ship {} at {:?}", ship_id, position);
        }
    }
    
    println!("Boost & Broadside - Phase 1 Core Simulation initialized");
    println!("Physics running at 50 Hz, Agent decisions simulated at 10 Hz");
    println!("World size: {}x{}", WORLD_WIDTH, WORLD_HEIGHT);
    println!("Spawned {} ships total", ship_positions.len());
}

// Placeholder systems - will be implemented in next steps
fn physics_system() {
    // TODO: Ship physics updates at 50 Hz
}

fn collision_system() {
    // TODO: Collision detection at 50 Hz
}

fn projectile_system() {
    // TODO: Projectile updates at 50 Hz
}

fn render_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    ships_query: Query<(Entity, &Ship, &Transform, &Renderable), (Without<Handle<Mesh>>, Without<Handle<ColorMaterial>>)>,
) {
    // Render ships as actual triangular meshes using MaterialMesh2dBundle
    // This system only runs when ships don't have rendering components yet
    for (entity, _ship, transform, renderable) in ships_query.iter() {
        // Create triangle mesh
        let triangle_mesh = create_triangle_mesh(renderable.size);
        let mesh_handle = meshes.add(triangle_mesh);
        let material_handle = materials.add(ColorMaterial::from(renderable.color));
        
        // Use MaterialMesh2dBundle for 2D rendering
        commands.entity(entity).insert(bevy::sprite::MaterialMesh2dBundle {
            mesh: mesh_handle.into(),
            material: material_handle,
            transform: *transform,
            ..default()
        });
        
        println!("Rendered triangular ship at {:?} with color {:?}", transform.translation, renderable.color);
    }
}

// Create a proper triangle mesh
fn create_triangle_mesh(size: f32) -> Mesh {
    let half_size = size * 0.5;
    let width = half_size * 0.8;
    
    // Triangle vertices (pointing upward like Asteroids)
    let vertices = vec![
        [0.0, half_size, 0.0],       // Top point (nose)
        [-width, -half_size, 0.0],   // Bottom left
        [width, -half_size, 0.0],    // Bottom right
    ];
    
    // Triangle indices
    let indices = vec![0u32, 1, 2];
    
    // UV coordinates for texture mapping
    let uvs = vec![
        [0.5, 1.0], // Top
        [0.0, 0.0], // Bottom left  
        [1.0, 0.0], // Bottom right
    ];
    
    // Create the mesh
    use bevy::render::mesh::{Indices, PrimitiveTopology};
    use bevy::render::render_asset::RenderAssetUsages;
    
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    
    mesh
}
