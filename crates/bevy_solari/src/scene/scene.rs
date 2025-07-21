use std::f32::consts::PI;

use bevy_asset::AssetId;
use bevy_color::ColorToComponents;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{Mat3, Vec3};
use bevy_mesh::{Indices, Mesh, VertexAttributeValues};
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::HashMap;
use bevy_render::{
    mesh::{
        allocator::{MeshAllocator, MeshBufferSlice},
        RenderMesh,
    },
    primitives::{Aabb, MeshAabb},
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};

use crate::scene::extract::StandardMaterialAssets;

#[derive(Copy, Clone, Debug, Default)]
pub struct NormalCone {
    pub axis: Vec3,
    pub theta_o: f32,
    pub theta_e: f32,
}

impl NormalCone {
    pub fn merge(self, other: Self) -> Self {
        if self.axis == Vec3::ZERO {
            return other;
        }
        if other.axis == Vec3::ZERO {
            return self;
        }

        let a = self;
        let b = other;

        let theta_e = a.theta_e.min(b.theta_e);
        let theta_d = a.axis.dot(b.axis).clamp(-1.0, 1.0).acos();
        if PI.min(theta_d + b.theta_o) <= a.theta_o {
            return Self {
                axis: a.axis,
                theta_o: a.theta_o,
                theta_e,
            };
        }
        if PI.min(theta_d + a.theta_o) <= b.theta_o {
            return Self {
                axis: b.axis,
                theta_o: b.theta_o,
                theta_e,
            };
        }
        let theta_o = (a.theta_o + b.theta_o + theta_d) / 2.0;
        if theta_o >= PI {
            return Self {
                axis: a.axis,
                theta_o: PI,
                theta_e: PI / 2.0,
            };
        }

        let theta_r = theta_o - a.theta_o;
        let w_r = a.axis.cross(b.axis);
        if w_r.length_squared() < 0.00001 {
            return Self {
                axis: a.axis,
                theta_o: PI,
                theta_e: PI / 2.0,
            };
        }
        let axis = Mat3::from_axis_angle(w_r, theta_r) * a.axis;

        Self {
            axis,
            theta_o,
            theta_e,
        }
    }
}

pub struct MeshData {
    pub blas: Blas,
    pub aabb: Aabb,
    pub normal_average: Vec3,
    pub normal_cone: NormalCone,
    pub area: f32,
}

pub struct MaterialData {
    pub average_emissive: Vec3,
}

#[derive(Resource, Default)]
pub struct SceneManager {
    meshes: HashMap<AssetId<Mesh>, MeshData>,
    images: HashMap<AssetId<Image>, Vec3>,
    materials: HashMap<AssetId<StandardMaterial>, MaterialData>,
}

impl SceneManager {
    pub fn get_mesh(&self, mesh: &AssetId<Mesh>) -> Option<&MeshData> {
        self.meshes.get(mesh)
    }

    pub fn get_blas(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.get_mesh(mesh).map(|data| &data.blas)
    }

    pub fn get_material(&self, material: &AssetId<StandardMaterial>) -> Option<&MaterialData> {
        self.materials.get(material)
    }
}

pub fn prepare_raytracing_meshes(
    mut scene_manager: ResMut<SceneManager>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    extracted_images: Res<ExtractedAssets<GpuImage>>,
    scene_materials: Res<StandardMaterialAssets>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let SceneManager {
        meshes,
        images,
        materials,
    } = &mut *scene_manager;

    for asset_id in extracted_images
        .removed
        .iter()
        .chain(extracted_images.modified.iter())
    {
        images.remove(asset_id);
    }
    // TODO: We really shouldn't have to calculate averages for every image, but there's no better
    // way to do it with the current API
    for (asset_id, image) in extracted_images.extracted.iter() {
        let is_rgba8 = image.texture_descriptor.format == TextureFormat::Rgba8Unorm
            || image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb;
        if is_rgba8 && let Some(data) = image.data.as_ref() {
            let is_srgb = image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb;
            let average = data
                .chunks_exact(4)
                .map(|x| {
                    let raw = Vec3::new(
                        x[0] as f32 / 255.0,
                        x[1] as f32 / 255.0,
                        x[2] as f32 / 255.0,
                    );
                    if is_srgb {
                        raw.powf(2.2)
                    } else {
                        raw
                    }
                })
                .sum::<Vec3>()
                / (data.len() as f32 / 4.0);
            images.insert(*asset_id, average);
        }
    }

    // Sync material data
    for (asset_id, material) in scene_materials.iter() {
        let average_emissive = material.emissive_texture.as_ref().map_or(Vec3::ONE, |e| {
            *images
                .get(&e.id())
                .expect("Emissive texture wasn't cached previously")
        }) * material.emissive.to_vec3();
        materials.insert(*asset_id, MaterialData { average_emissive });
    }

    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        meshes.remove(asset_id);
    }

    if extracted_meshes.extracted.is_empty() {
        return;
    }

    // Create new BLAS for added or changed meshes
    let blas_resources = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
        .map(|(asset_id, mesh)| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();

            let (data, blas_size) = allocate_blas_and_compute_mesh_data(
                mesh,
                &vertex_slice,
                &index_slice,
                asset_id,
                &render_device,
            );

            meshes.insert(*asset_id, data);

            (*asset_id, vertex_slice, index_slice, blas_size)
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, vertex_slice, index_slice, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertex_slice.buffer,
                first_vertex: vertex_slice.range.start,
                vertex_stride: 48,
                index_buffer: Some(index_slice.buffer),
                first_index: Some(index_slice.range.start),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &meshes[asset_id].blas,
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_blas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    render_queue.submit([command_encoder.finish()]);
}

fn allocate_blas_and_compute_mesh_data(
    mesh: &Mesh,
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    asset_id: &AssetId<Mesh>,
    render_device: &RenderDevice,
) -> (MeshData, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_slice.range.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&asset_id.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    let aabb = mesh.compute_aabb().unwrap_or_default();
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        unreachable!(); // Already tested in `is_mesh_raytracing_compatible`
    };
    let Some(Indices::U32(indices)) = mesh.indices() else {
        unreachable!(); // Already tested in `is_mesh_raytracing_compatible`
    };
    let (normal_cone, normal_sum, area) = indices.chunks_exact(3).fold(
        (NormalCone::default(), Vec3::ZERO, 0.0),
        |(cone, normal_sum, area), tri| {
            let v0 = Vec3::from(positions[tri[0] as usize]);
            let v1 = Vec3::from(positions[tri[1] as usize]);
            let v2 = Vec3::from(positions[tri[2] as usize]);
            let normal = (v1 - v0).cross(v2 - v0);
            let len = normal.length();
            let area = area + len * 0.5;
            let normal = normal.normalize_or_zero();
            let c = NormalCone {
                axis: normal,
                theta_o: 0.0,
                theta_e: PI * 0.5,
            };
            (cone.merge(c), normal_sum + normal, area)
        },
    );
    let normal_average = normal_sum / (indices.len() / 3) as f32;

    (
        MeshData {
            blas,
            aabb,
            normal_average,
            normal_cone,
            area,
        },
        blas_size,
    )
}

fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let vertex_attributes = mesh.attributes().map(|(attribute, _)| attribute.id).eq([
        Mesh::ATTRIBUTE_POSITION.id,
        Mesh::ATTRIBUTE_NORMAL.id,
        Mesh::ATTRIBUTE_UV_0.id,
        Mesh::ATTRIBUTE_TANGENT.id,
    ]);
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
    mesh.enable_raytracing && triangle_list && vertex_attributes && indexed_32
}
