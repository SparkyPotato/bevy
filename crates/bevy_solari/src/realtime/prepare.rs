use super::SolariLighting;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_aliasing::dlss::{
    Dlss, DlssRayReconstructionFeature, ViewDlssRayReconstructionTextures,
};
use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::{core_3d::CORE_3D_DEPTH_FORMAT, deferred::DEFERRED_PREPASS_FORMAT};
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, Texture, TextureDescriptor, TextureDimension,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_render::{render_resource::TextureFormat, texture::CachedTexture};

/// Size of the GI `Reservoir` shader struct in bytes.
const GI_RESERVOIR_STRUCT_SIZE: u64 = 48;

/// Internal rendering resources used for Solari lighting.
#[derive(Component)]
pub struct SolariLightingResources {
    pub gi_reservoirs_a: Buffer,
    pub gi_reservoirs_b: Buffer,
    pub previous_gbuffer: (Texture, TextureView),
    pub previous_depth: (Texture, TextureView),
    pub view_size: UVec2,
}

pub fn prepare_solari_lighting_resources(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&SolariLightingResources>,
            Option<&MainPassResolutionOverride>,
        ),
        With<SolariLighting>,
    >,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&SolariLightingResources>,
            Option<&MainPassResolutionOverride>,
            Has<Dlss<DlssRayReconstructionFeature>>,
        ),
        With<SolariLighting>,
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for query_item in &query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (entity, camera, solari_lighting_resources, resolution_override) = query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (entity, camera, solari_lighting_resources, resolution_override, has_dlss_rr) =
            query_item;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        if solari_lighting_resources.map(|r| r.view_size) == Some(view_size) {
            continue;
        }

        let gi_reservoirs = |name| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: (view_size.x * view_size.y) as u64 * GI_RESERVOIR_STRUCT_SIZE,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let gi_reservoirs_a = gi_reservoirs("solari_lighting_gi_reservoirs_a");
        let gi_reservoirs_b = gi_reservoirs("solari_lighting_gi_reservoirs_b");

        let previous_gbuffer = render_device.create_texture(&TextureDescriptor {
            label: Some("solari_lighting_previous_gbuffer"),
            size: view_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DEFERRED_PREPASS_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let previous_gbuffer_view = previous_gbuffer.create_view(&TextureViewDescriptor::default());

        let previous_depth = render_device.create_texture(&TextureDescriptor {
            label: Some("solari_lighting_previous_depth"),
            size: view_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let previous_depth_view = previous_depth.create_view(&TextureViewDescriptor::default());

        commands.entity(entity).insert(SolariLightingResources {
            gi_reservoirs_a,
            gi_reservoirs_b,
            previous_gbuffer: (previous_gbuffer, previous_gbuffer_view),
            previous_depth: (previous_depth, previous_depth_view),
            view_size,
        });

        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        if has_dlss_rr {
            let diffuse_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_diffuse_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let diffuse_albedo_view = diffuse_albedo.create_view(&TextureViewDescriptor::default());

            let specular_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_albedo_view =
                specular_albedo.create_view(&TextureViewDescriptor::default());

            let normal_roughness = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_normal_roughness"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let normal_roughness_view =
                normal_roughness.create_view(&TextureViewDescriptor::default());

            let specular_motion_vectors = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_motion_vectors"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_motion_vectors_view =
                specular_motion_vectors.create_view(&TextureViewDescriptor::default());

            commands
                .entity(entity)
                .insert(ViewDlssRayReconstructionTextures {
                    diffuse_albedo: CachedTexture {
                        texture: diffuse_albedo,
                        default_view: diffuse_albedo_view,
                    },
                    specular_albedo: CachedTexture {
                        texture: specular_albedo,
                        default_view: specular_albedo_view,
                    },
                    normal_roughness: CachedTexture {
                        texture: normal_roughness,
                        default_view: normal_roughness_view,
                    },
                    specular_motion_vectors: CachedTexture {
                        texture: specular_motion_vectors,
                        default_view: specular_motion_vectors_view,
                    },
                });
        }
    }
}
