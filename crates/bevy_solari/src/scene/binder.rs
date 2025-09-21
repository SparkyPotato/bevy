use super::{blas::BlasManager, extract::StandardMaterialAssets, RaytracingMesh3d};
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    resource::Resource,
    system::{Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{ops::cos, Mat4, Vec3};
use bevy_pbr::{ExtractedDirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy_platform::{collections::HashMap, hash::FixedHasher};
use bevy_render::{
    mesh::allocator::MeshAllocator,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use bevy_transform::components::GlobalTransform;
use core::{f32::consts::TAU, hash::Hash, num::NonZeroU32, ops::Deref};

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

/// Amount of entries in the light cache (must be a power of 2, and >= 2^10)
pub const LIGHT_CACHE_SIZE: u64 = 2u64.pow(20);
/// This should match the size of `LightCacheCellRead` and `LightCacheCellWrite` in
/// `light_cache.wgsl`!
const LIGHT_CACHE_CELL_SIZE: u64 = size_of::<[u32; 17]>() as _;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayout,
    light_cache_checksums: [Buffer; 2],
    light_cache_cells: [Buffer; 2],
    previous_frame_light_entities: Vec<Entity>,
    read_index: usize,
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
    blas_manager: Res<BlasManager>,
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
    let mut tlas = render_device
        .wgpu_device()
        .create_tlas(&CreateTlasDescriptor {
            label: Some("tlas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: instances_query.iter().len() as u32,
        });
    let mut transforms = StorageBufferList::<Mat4>::default();
    let mut geometry_ids = StorageBufferList::<GpuInstanceGeometryIds>::default();
    let mut material_ids = StorageBufferList::<u32>::default();
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
        let Some(metallic_roughness_texture_id) =
            process_texture(&material.metallic_roughness_texture)
        else {
            continue;
        };

        materials.get_mut().push(GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: material.emissive.to_vec3(),
            metallic: material.metallic,
            reflectance: LinearRgba::from(material.specular_tint).to_vec3() * material.reflectance,
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
    for (entity, mesh, material, transform) in &instances_query {
        let Some(blas) = blas_manager.get(&mesh.id()) else {
            continue;
        };
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&mesh.id()) else {
            continue;
        };
        let Some(index_slice) = mesh_allocator.mesh_index_slice(&mesh.id()) else {
            continue;
        };
        let Some(material_id) = material_id_map.get(&material.id()).copied() else {
            continue;
        };
        let Some(material) = materials.get().get(material_id as usize) else {
            continue;
        };

        let transform = transform.to_matrix();
        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
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
            triangle_count: (index_slice.range.len() / 3) as u32,
        });

        material_ids.get_mut().push(material_id);

        if material.emissive != Vec3::ZERO {
            light_sources
                .get_mut()
                .push(GpuLightSource::new_emissive_mesh_light(
                    instance_id as u32,
                    (index_slice.range.len() / 3) as u32,
                ));

            this_frame_entity_to_light_id.insert(entity, light_sources.get().len() as u32 - 1);
            raytracing_scene_bindings
                .previous_frame_light_entities
                .push(entity);
        }

        instance_id += 1;
    }

    if instance_id == 0 {
        return;
    }

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

    if light_sources.get().len() > u16::MAX as usize {
        panic!("Too many light sources in the scene, maximum is 65536.");
    }

    materials.write_buffer(&render_device, &render_queue);
    transforms.write_buffer(&render_device, &render_queue);
    geometry_ids.write_buffer(&render_device, &render_queue);
    material_ids.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);
    directional_lights.write_buffer(&render_device, &render_queue);
    previous_frame_light_id_translations.write_buffer(&render_device, &render_queue);

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_tlas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&[], [&tlas]);
    render_queue.submit([command_encoder.finish()]);

    raytracing_scene_bindings.bind_group = Some(
        render_device.create_bind_group(
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
                light_sources.binding().unwrap(),
                directional_lights.binding().unwrap(),
                previous_frame_light_id_translations.binding().unwrap(),
                raytracing_scene_bindings.light_cache_checksums
                    [raytracing_scene_bindings.read_index]
                    .as_entire_binding(),
                raytracing_scene_bindings.light_cache_checksums
                    [raytracing_scene_bindings.read_index ^ 1]
                    .as_entire_binding(),
                raytracing_scene_bindings.light_cache_cells[raytracing_scene_bindings.read_index]
                    .as_entire_binding(),
                raytracing_scene_bindings.light_cache_cells
                    [raytracing_scene_bindings.read_index ^ 1]
                    .as_entire_binding(),
            )),
        ),
    );

    raytracing_scene_bindings.read_index ^= 1;
}

impl FromWorld for RaytracingSceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let buffer = |label: &str, cell_size: u64| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: LIGHT_CACHE_SIZE * cell_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

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
                        storage_buffer_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                    ),
                ),
            ),
            light_cache_checksums: [
                buffer("raytracing_light_cache_checksums_a", size_of::<u32>() as _),
                buffer("raytracing_light_cache_checksums_b", size_of::<u32>() as _),
            ],
            light_cache_cells: [
                buffer("raytracing_light_cache_cells_a", LIGHT_CACHE_CELL_SIZE),
                buffer("raytracing_light_cache_cells_b", LIGHT_CACHE_CELL_SIZE),
            ],
            read_index: 0,
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
    triangle_count: u32,
}

#[derive(ShaderType)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    reflectance: Vec3,
    _padding: f32,
}

#[derive(ShaderType)]
struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        if triangle_count > u16::MAX as u32 {
            panic!("Too many triangles ({triangle_count}) in an emissive mesh, maximum is 65535.");
        }

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
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
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
