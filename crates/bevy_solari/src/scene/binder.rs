use crate::scene::scene::NormalCone;

use super::{extract::StandardMaterialAssets, scene::SceneManager, RaytracingMesh3d};
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    resource::Resource,
    system::{Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{ops::cos, Mat4, Vec3, Vec3A, Vec4, Vec4Swizzles};
use bevy_pbr::{ExtractedDirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy_platform::{collections::HashMap, hash::FixedHasher};
use bevy_render::{
    mesh::allocator::MeshAllocator,
    primitives::Aabb,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use bevy_transform::components::GlobalTransform;
use core::{f32::consts::TAU, hash::Hash, num::NonZeroU32, ops::Deref};
use std::f32::consts::PI;

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

/// Average angular diameter of the sun as seen from earth.
/// <https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy>
const SUN_ANGULAR_DIAMETER_RADIANS: f32 = 0.00930842;

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayout,
    previous_frame_light_entities: Vec<Entity>,
}

pub fn prepare_raytracing_scene_bindings(
    instances_query: Query<(
        Entity,
        &RaytracingMesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
    )>,
    directional_lights_query: Query<(Entity, &ExtractedDirectionalLight)>,
    mesh_allocator: Res<MeshAllocator>,
    scene_manager: Res<SceneManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut raytracing_scene_bindings: ResMut<RaytracingSceneBindings>,
) {
    raytracing_scene_bindings.bind_group = None;

    let mut this_frame_entity_to_light_id = EntityHashMap::<u32>::default();
    let previous_frame_light_entities: Vec<_> = raytracing_scene_bindings
        .previous_frame_light_entities
        .drain(..)
        .collect();

    if instances_query.iter().len() == 0 {
        return;
    }

    let mut vertex_buffers = CachedBindingArray::new();
    let mut index_buffers = CachedBindingArray::new();
    let mut textures = CachedBindingArray::new();
    let mut samplers = Vec::new();
    let mut materials = StorageBufferList::<GpuMaterial>::default();
    let mut tlas = TlasPackage::new(render_device.wgpu_device().create_tlas(
        &CreateTlasDescriptor {
            label: Some("tlas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: instances_query.iter().len() as u32,
        },
    ));
    let mut transforms = StorageBufferList::<Mat4>::default();
    let mut geometry_ids = StorageBufferList::<GpuInstanceGeometryIds>::default();
    let mut material_ids = StorageBufferList::<u32>::default();
    let mut light_tree = StorageBufferList::<GpuLightTreeNode>::default();
    let mut light_sources = StorageBufferList::<GpuLightSource>::default();
    let mut directional_lights = StorageBufferList::<GpuDirectionalLight>::default();
    let mut previous_frame_light_id_translations = StorageBufferList::<u32>::default();

    let mut material_id_map: HashMap<AssetId<StandardMaterial>, u32, FixedHasher> =
        HashMap::default();
    let mut material_id = 0;
    let mut process_texture = |texture_handle: &Option<Handle<_>>| -> Option<u32> {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) => {
                    let (texture_id, is_new) =
                        textures.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    if is_new {
                        samplers.push(texture.sampler.deref());
                    }
                    Some(texture_id)
                }
                None => None,
            },
            None => Some(TEXTURE_MAP_NONE),
        }
    };
    for (asset_id, material) in material_assets.iter() {
        let Some(base_color_texture_id) = process_texture(&material.base_color_texture) else {
            continue;
        };
        let Some(normal_map_texture_id) = process_texture(&material.normal_map_texture) else {
            continue;
        };
        let Some(emissive_texture_id) = process_texture(&material.emissive_texture) else {
            continue;
        };

        materials.get_mut().push(GpuMaterial {
            base_color: material.base_color.to_linear(),
            emissive: material.emissive,
            base_color_texture_id,
            normal_map_texture_id,
            emissive_texture_id,
            _padding: Default::default(),
        });

        material_id_map.insert(*asset_id, material_id);
        material_id += 1;
    }

    if material_id == 0 {
        return;
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    let mut instance_id = 0;
    let mut lights = Vec::new();
    let mut vmf_normals = Vec::new();
    for (entity, mesh, material, transform) in &instances_query {
        let average_emissive = scene_manager
            .get_material(&material.id())
            .expect("RaytracingMesh3d material not found")
            .average_emissive;

        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&mesh.id()) else {
            continue;
        };
        let Some(index_slice) = mesh_allocator.mesh_index_slice(&mesh.id()) else {
            continue;
        };
        let Some(mesh) = scene_manager.get_mesh(&mesh.id()) else {
            continue;
        };

        if average_emissive != Vec3::ZERO {
            let light_id = light_sources.get().len() as u32;
            vmf_normals.push(mesh.normal_average);
            lights.push(transform_light(
                transform,
                BuildLight {
                    aabb: mesh.aabb,
                    power: average_emissive * mesh.area * PI,
                    cone: mesh.normal_cone,
                    left: u32::MAX,
                    right: light_id,
                },
            ));

            this_frame_entity_to_light_id.insert(entity, light_id);
            light_sources
                .get_mut()
                .push(GpuLightSource::new_emissive_mesh_light(
                    instance_id as u32,
                    (index_slice.range.len() / 3) as u32,
                ));

            raytracing_scene_bindings
                .previous_frame_light_entities
                .push(entity);
        }

        let Some(material_id) = material_id_map.get(&material.id()).copied() else {
            continue;
        };

        let transform = transform.to_matrix();
        *tlas.get_mut_single(instance_id as usize).unwrap() = Some(TlasInstance::new(
            &mesh.blas,
            tlas_transform(&transform),
            Default::default(),
            0xFF,
        ));

        transforms.get_mut().push(transform);

        let (vertex_buffer_id, _) = vertex_buffers.push_if_absent(
            vertex_slice.buffer.as_entire_buffer_binding(),
            vertex_slice.buffer.id(),
        );
        let (index_buffer_id, _) = index_buffers.push_if_absent(
            index_slice.buffer.as_entire_buffer_binding(),
            index_slice.buffer.id(),
        );

        geometry_ids.get_mut().push(GpuInstanceGeometryIds {
            vertex_buffer_id,
            vertex_buffer_offset: vertex_slice.range.start,
            index_buffer_id,
            index_buffer_offset: index_slice.range.start,
        });

        material_ids.get_mut().push(material_id);

        instance_id += 1;
    }

    if instance_id == 0 {
        return;
    }

    if lights.len() == 0 {
        light_tree.get_mut().push(GpuLightTreeNode {
            left: GpuSgLight::default(),
            right: GpuSgLight::default(),
            left_index: u32::MAX,
            right_index: u32::MAX,
            _padding: [0; 2],
        });
    } else if lights.len() == 1 {
        let light = lights[0];
        light_tree.get_mut().push(GpuLightTreeNode {
            left: leaf_sg_light(light, &vmf_normals),
            right: GpuSgLight::default(),
            left_index: light.right | (1 << 31),
            right_index: u32::MAX,
            _padding: [0; 2],
        });
    } else {
        let mut indices: Vec<_> = (0..lights.len() as u32).collect();
        let root = build_bvh(&mut lights, &mut indices);
        let root = build_gpu_bvh(light_tree.get_mut(), &lights, &vmf_normals, root);
        assert_eq!(root, 0, "Root of the light BVH should be 0");
    }

    // TODO: Use this when we need PDFs for MIS.
    // let mut mesh_paths = Vec::with_capacity(vmf_normals.len());
    // mesh_paths.resize(vmf_normals.len(), 0);
    // emissive_paths(&mut mesh_paths, light_tree.get(), 0, 0, 0);

    for (entity, directional_light) in &directional_lights_query {
        let directional_lights = directional_lights.get_mut();
        let directional_light_id = directional_lights.len() as u32;

        directional_lights.push(GpuDirectionalLight::new(directional_light));

        light_sources
            .get_mut()
            .push(GpuLightSource::new_directional_light(directional_light_id));

        this_frame_entity_to_light_id.insert(entity, light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_entities
            .push(entity);
    }

    for previous_frame_light_entity in previous_frame_light_entities {
        let current_frame_index = this_frame_entity_to_light_id
            .get(&previous_frame_light_entity)
            .copied()
            .unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);
        previous_frame_light_id_translations
            .get_mut()
            .push(current_frame_index);
    }

    materials.write_buffer(&render_device, &render_queue);
    transforms.write_buffer(&render_device, &render_queue);
    geometry_ids.write_buffer(&render_device, &render_queue);
    material_ids.write_buffer(&render_device, &render_queue);
    light_tree.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);
    directional_lights.write_buffer(&render_device, &render_queue);
    previous_frame_light_id_translations.write_buffer(&render_device, &render_queue);

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_tlas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&[], [&tlas]);
    render_queue.submit([command_encoder.finish()]);

    raytracing_scene_bindings.bind_group = Some(render_device.create_bind_group(
        "raytracing_scene_bind_group",
        &raytracing_scene_bindings.bind_group_layout,
        &BindGroupEntries::sequential((
            vertex_buffers.as_slice(),
            index_buffers.as_slice(),
            textures.as_slice(),
            samplers.as_slice(),
            materials.binding().unwrap(),
            tlas.as_binding(),
            transforms.binding().unwrap(),
            geometry_ids.binding().unwrap(),
            material_ids.binding().unwrap(),
            light_tree.binding().unwrap(),
            light_sources.binding().unwrap(),
            directional_lights.binding().unwrap(),
            previous_frame_light_id_translations.binding().unwrap(),
        )),
    ));
}

impl FromWorld for RaytracingSceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            bind_group: None,
            bind_group_layout: render_device.create_bind_group_layout(
                "raytracing_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT),
                        storage_buffer_read_only_sized(false, None),
                        acceleration_structure(),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
            previous_frame_light_entities: Vec::new(),
        }
    }
}

struct CachedBindingArray<T, I: Eq + Hash> {
    map: HashMap<I, u32>,
    vec: Vec<T>,
}

impl<T, I: Eq + Hash> CachedBindingArray<T, I> {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            vec: Vec::default(),
        }
    }

    fn push_if_absent(&mut self, item: T, item_id: I) -> (u32, bool) {
        let mut is_new = false;
        let i = *self.map.entry(item_id).or_insert_with(|| {
            is_new = true;
            let i = self.vec.len() as u32;
            self.vec.push(item);
            i
        });
        (i, is_new)
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
}

type StorageBufferList<T> = StorageBuffer<Vec<T>>;

#[derive(ShaderType)]
struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
}

#[derive(ShaderType)]
struct GpuMaterial {
    base_color: LinearRgba,
    emissive: LinearRgba,
    base_color_texture_id: u32,
    normal_map_texture_id: u32,
    emissive_texture_id: u32,
    _padding: u32,
}

#[derive(ShaderType)]
struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        Self {
            kind: triangle_count << 1,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }
}

#[derive(ShaderType, Default)]
struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(SUN_ANGULAR_DIAMETER_RADIANS / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}

#[derive(Copy, Clone, Debug)]
struct BuildLight {
    aabb: Aabb,
    power: Vec3,
    cone: NormalCone,
    // If nodes, both are indices into the list.
    // If leaves, left is u32::MAX, and right is mesh light id.
    left: u32,
    right: u32,
}

fn transform_light(t: &GlobalTransform, mut light: BuildLight) -> BuildLight {
    let mat = t.to_matrix();
    let min = light.aabb.min();
    let max = light.aabb.max();
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for c in corners {
        let c = (mat * Vec4::new(c.x, c.y, c.z, 1.0)).xyz();
        min = min.min(c);
        max = max.max(c);
    }
    light.aabb = Aabb::from_min_max(min, max);
    let a = light.cone.axis;
    light.cone.axis = (mat * Vec4::new(a.x, a.y, a.z, 0.0)).xyz();
    light
}

fn merge_lights(left: BuildLight, right: BuildLight) -> BuildLight {
    BuildLight {
        aabb: Aabb::from_min_max(
            left.aabb.min().min(right.aabb.min()).into(),
            left.aabb.max().max(right.aabb.max()).into(),
        ),
        power: left.power + right.power,
        cone: left.cone.merge(right.cone),
        left: left.left,
        right: right.right,
    }
}

// SAOH, from [Importance Sampling of Many Lights with Adaptive Tree Splitting, Kulla and Conty,
// 2018]
fn m_omega(cone: NormalCone) -> f32 {
    let theta_o = cone.theta_o;
    let theta_e = cone.theta_e;
    let theta_w = (theta_o + theta_e).min(PI);
    let (sin_o, cos_o) = theta_o.sin_cos();
    let w_2 = theta_w * 2.0;
    2.0 * PI * (1.0 - cos_o)
        + 0.5 * PI * (w_2 * sin_o - (theta_o - w_2).cos() - 2.0 * theta_o * sin_o + cos_o)
}

fn surface_area(extents: Vec3A) -> f32 {
    2.0 * (extents.x * extents.y + extents.x * extents.z + extents.y * extents.z)
}

fn luminance(x: Vec3) -> f32 {
    0.2126 * x.x + 0.7152 * x.y + 0.0722 * x.z
}

fn saoh(curr: BuildLight, left: BuildLight, right: BuildLight, axis: u32) -> f32 {
    let extents = curr.aabb.half_extents * 2.0;
    let k_r = extents.max_element() / extents[axis as usize];

    let curr_omega = m_omega(curr.cone);
    let curr_sa = surface_area(extents);

    let left_omega = m_omega(left.cone);
    let left_sa = surface_area(left.aabb.half_extents * 2.0);
    let right_omega = m_omega(right.cone);
    let right_sa = surface_area(right.aabb.half_extents * 2.0);

    k_r * (luminance(left.power) * left_omega * left_sa
        + luminance(right.power) * right_omega * right_sa)
        / (curr_omega * curr_sa)
}

fn build_bvh(nodes: &mut Vec<BuildLight>, indices: &mut [u32]) -> u32 {
    let count = indices.len();
    if count == 1 {
        indices[0]
    } else if count == 2 {
        let i = nodes.len();
        let left = indices[0];
        let right = indices[1];
        nodes.push(BuildLight {
            left,
            right,
            ..merge_lights(nodes[left as usize], nodes[right as usize])
        });
        i as _
    } else {
        let merged = indices
            .iter()
            .map(|&i| nodes[i as usize])
            .reduce(|a, b| merge_lights(a, b))
            .unwrap();
        let p_40 = (count as f32 * 0.4) as usize;
        let p_60 = (count as f32 * 0.6) as usize;
        let mut cost = f32::INFINITY;
        let mut axis = 0;
        let mut split = 0;
        let key = |x, ax| nodes[x as usize].aabb.center[ax];
        for ax in 0..3 {
            indices.sort_unstable_by(|&x, &y| key(x, ax).partial_cmp(&key(y, ax)).unwrap());
            for s in p_40..=p_60 {
                let (left, right) = indices.split_at_mut(s);
                let left_merged = left
                    .iter()
                    .map(|&i| nodes[i as usize])
                    .reduce(|a, b| merge_lights(a, b))
                    .unwrap();
                let right_merged = right
                    .iter()
                    .map(|&i| nodes[i as usize])
                    .reduce(|a, b| merge_lights(a, b))
                    .unwrap();
                let c = saoh(merged, left_merged, right_merged, ax as _);
                if c < cost {
                    cost = c;
                    axis = ax;
                    split = s;
                }
            }
        }
        if axis != 2 {
            indices.sort_unstable_by(|&x, &y| key(x, axis).partial_cmp(&key(y, axis)).unwrap());
        }

        let (left, right) = indices.split_at_mut(split);
        let left = build_bvh(nodes, left);
        let right = build_bvh(nodes, right);
        let i = nodes.len() as u32;
        nodes.push(BuildLight {
            left,
            right,
            ..merge_lights(nodes[left as usize], nodes[right as usize])
        });
        i as _
    }
}

#[derive(ShaderType, Default)]
struct GpuSgLight {
    pos: Vec3,
    variance: f32,
    intensity: Vec3,
    _padding: u32,
    axis: Vec3,
    sharpness: f32,
}

#[derive(ShaderType, Default)]
struct GpuLightTreeNode {
    left: GpuSgLight,
    right: GpuSgLight,
    // If indices are u32::MAX they are invalid.
    // If indices have their MSB set, then the node is a leaf, with the index being the light id.
    left_index: u32,
    right_index: u32,
    _padding: [u32; 2],
}

// [Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting, AMD, 2024]
fn axis_to_vmf(axis: Vec3) -> (Vec3, f32) {
    let len = axis.length().clamp(0.0, 1.0);
    let len2 = len * len;
    let len3 = len2 * len;
    let sharpness = ((3.0 * len - len3) / (1.0 - len2)).min(2.19902e+12);
    let axis = axis.normalize_or_zero();
    (axis, sharpness)
}

fn expm1_over_x(x: f32) -> f32 {
    let u = x.exp();
    if u == 1.0 {
        return 1.0;
    }
    let y = u - 1.0;
    if x.abs() < 1.0 {
        return y / u.ln();
    }

    y / x
}

fn sg_integral(sharpness: f32) -> f32 {
    4.0 * PI * expm1_over_x(-2.0 * sharpness)
}

fn power_to_intensity(power: Vec3, sharpness: f32) -> Vec3 {
    power / (2.0 * PI * sg_integral(sharpness))
}

fn leaf_sg_light(light: BuildLight, meshes: &[Vec3]) -> GpuSgLight {
    debug_assert_eq!(light.left, u32::MAX);
    let (axis, sharpness) = axis_to_vmf(meshes[light.right as usize]);
    let variance = 0.5 * light.aabb.half_extents.length_squared();
    GpuSgLight {
        pos: light.aabb.center.into(),
        variance,
        intensity: power_to_intensity(light.power, sharpness),
        _padding: 0,
        axis,
        sharpness,
    }
}

fn handle_child(
    out: &mut Vec<GpuLightTreeNode>,
    bvh: &[BuildLight],
    meshes: &[Vec3],
    node: u32,
) -> (GpuSgLight, u32) {
    let n = bvh[node as usize];
    if n.left == u32::MAX {
        (leaf_sg_light(n, meshes), n.right | (1 << 31))
    } else {
        let child_id = build_gpu_bvh(out, bvh, meshes, node);
        let child = &out[child_id as usize];

        let left_lum = luminance(bvh[n.left as usize].power);
        let right_lum = luminance(bvh[n.right as usize].power);
        let w_left = left_lum / (left_lum + right_lum);
        let w_right = 1.0 - w_left;

        let variance = child.left.variance * w_left
            + child.right.variance * w_right
            + w_left * w_right * (child.left.pos - child.right.pos).length_squared();

        let axis_avg = child.left.axis * w_left + child.right.axis * w_right;
        let (axis, sharpness) = axis_to_vmf(axis_avg);

        (
            GpuSgLight {
                pos: child.left.pos * w_left + child.right.pos * w_right,
                variance,
                intensity: power_to_intensity(n.power, sharpness),
                _padding: 0,
                axis,
                sharpness,
            },
            child_id,
        )
    }
}

fn build_gpu_bvh(
    out: &mut Vec<GpuLightTreeNode>,
    bvh: &[BuildLight],
    meshes: &[Vec3],
    node: u32,
) -> u32 {
    let us_index = out.len();
    let us = &bvh[node as usize];
    out.push(GpuLightTreeNode::default());
    let left = handle_child(out, bvh, meshes, us.left);
    let right = handle_child(out, bvh, meshes, us.right);
    let out = &mut out[us_index];
    (out.left, out.left_index) = left;
    (out.right, out.right_index) = right;
    us_index as _
}

fn emissive_paths(out: &mut Vec<u32>, bvh: &[GpuLightTreeNode], node: u32, depth: u32, path: u32) {
    let node = &bvh[node as usize];
    let left_path = path;
    let right_path = path | (1 << depth);
    if (node.left_index >> 31) & 1 == 1 {
        if node.left_index != u32::MAX {
            let i = node.left_index & !(1 << 31);
            out[i as usize] = left_path;
        }
    } else {
        emissive_paths(out, bvh, node.left_index, depth + 1, left_path);
    }
    if (node.right_index >> 31) & 1 == 1 {
        if node.right_index != u32::MAX {
            let i = node.right_index & !(1 << 31);
            out[i as usize] = right_path;
        }
    } else {
        emissive_paths(out, bvh, node.right_index, depth + 1, right_path);
    }
}
