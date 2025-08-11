//! Meshlet rendering for dense high-poly scenes (experimental).

// Note: This example showcases the meshlet API, but is not the type of scene that would benefit from using meshlets.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    anti_aliasing::taa::{TemporalAntiAliasPlugin, TemporalAntiAliasing},
    pbr::experimental::meshlet::{
        MeshletMesh, MeshletMesh3d, MeshletPlugin,
        MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR,
    },
    prelude::*,
    render::render_resource::AsBindGroup,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::{f32::consts::PI, path::Path, process::ExitCode};

const ASSET_URL: &str =
    "https://raw.githubusercontent.com/atlv24/assets/69bb39164fd35aadf863f6009520d4981eafcea0/bunny.meshlet_mesh";

fn main() -> ExitCode {
    if !Path::new("./assets/external/models/bunny.meshlet_mesh").exists() {
        eprintln!("ERROR: Asset at path <bevy>/assets/external/models/bunny.meshlet_mesh is missing. Please download it from {ASSET_URL}");
        return ExitCode::FAILURE;
    }

    App::new()
        .add_plugins((
            DefaultPlugins,
            MeshletPlugin {
                cluster_buffer_slots: 1 << 25,
            },
            MaterialPlugin::<MeshletDebugMaterial>::default(),
            CameraControllerPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, plop_meshes)
        .run();

    ExitCode::SUCCESS
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(1.8, 0.4, -0.1)).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 150.0,
            ..default()
        },
        CameraController::default(),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, PI * -0.15, PI * -0.15)),
    ));

    // A custom file format storing a [`bevy_render::mesh::Mesh`]
    // that has been converted to a [`bevy_pbr::meshlet::MeshletMesh`]
    // using [`bevy_pbr::meshlet::MeshletMesh::from_mesh`], which is
    // a function only available when the `meshlet_processor` cargo feature is enabled.
    let dragon = asset_server.load("external/models/dragon.gltf#Mesh0/Primitive0");
    commands.insert_resource(Dummy(Some(dragon)));
}

#[derive(Resource)]
struct Dummy(Option<Handle<Mesh>>);

fn plop_meshes(
    mut commands: Commands,
    mut run: ResMut<Dummy>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut m_meshes: ResMut<Assets<MeshletMesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut debug_materials: ResMut<Assets<MeshletDebugMaterial>>,
) {
    if let Some(mesh) = run.0.as_ref()
        && let Some(mesh) = meshes.get(mesh)
    {
        run.0 = None;
        let mesh =
            MeshletMesh::from_mesh(mesh, MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR)
                .unwrap();
        let handle = m_meshes.add(mesh);
        let debug_material = debug_materials.add(MeshletDebugMaterial::default());

        let mats: [_; 5] = std::array::from_fn(|i| {
            standard_materials.add(StandardMaterial {
                base_color: match i {
                    0 => Srgba::hex("#dc2626").unwrap().into(),
                    1 => Srgba::hex("#ea580c").unwrap().into(),
                    2 => Srgba::hex("#facc15").unwrap().into(),
                    3 => Srgba::hex("#16a34a").unwrap().into(),
                    4 => Srgba::hex("#0284c7").unwrap().into(),
                    _ => unreachable!(),
                },
                ..default()
            })
        });

        for x in -25i32..=25 {
            for y in -25i32..=25 {
                for z in -25i32..=25 {
                    if (x & 1) == 0 {
                        commands.spawn((
                            MeshletMesh3d(handle.clone()),
                            MeshMaterial3d(debug_material.clone()),
                            Transform::default()
                                .with_translation(Vec3::new(x as f32, y as f32, z as f32))
                                .with_rotation(Quat::from_rotation_x(PI / 2.0))
                                .with_scale(Vec3::splat(3.0)),
                        ));
                    } else {
                        commands.spawn((
                            MeshletMesh3d(handle.clone()),
                            MeshMaterial3d(mats[(z.abs() % 5) as usize].clone()),
                            Transform::default()
                                .with_translation(Vec3::new(x as f32, y as f32, z as f32))
                                .with_rotation(Quat::from_rotation_x(PI / 2.0))
                                .with_scale(Vec3::splat(3.0)),
                        ));
                    }
                }
            }
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct MeshletDebugMaterial {
    _dummy: (),
}

impl Material for MeshletDebugMaterial {}
