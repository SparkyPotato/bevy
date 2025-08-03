// https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf

#define_import_path bevy_solari::presample_light_tiles

#import bevy_pbr::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::utils::{rand_range_u, octahedral_encode, octahedral_decode}
#import bevy_render::view::View
#import bevy_solari::sampling::{generate_light_tree_sample, LightSample, ResolvedLightSample}

@group(1) @binding(1) var<storage, read_write> light_tile_samples: array<LightSample>;
@group(1) @binding(2) var<storage, read_write> light_tile_resolved_samples: array<ResolvedLightSamplePacked>;
@group(1) @binding(7) var gbuffer: texture_2d<u32>;
@group(1) @binding(8) var depth_buffer: texture_depth_2d;
@group(1) @binding(12) var<uniform> view: View;
struct PushConstants { frame_index: u32, reset: u32 }
var<push_constant> constants: PushConstants;

const SAMPLES_PER_TILE = 512u;
const TILE_SIZE_BITS = 5u;
const TILE_SIZE = 32u;

@compute @workgroup_size(SAMPLES_PER_TILE, 1, 1)
fn presample_light_tiles(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) sample_index: u32) {
    let light_tile_count_x = (u32(view.viewport.z) + TILE_SIZE - 1u) >> TILE_SIZE_BITS;
    let tile_id = workgroup_id.y * light_tile_count_x + workgroup_id.x;
    var rng = (tile_id * 5782582u) + sample_index + constants.frame_index;
    
    let x = rand_range_u(TILE_SIZE, &rng);
    let y = rand_range_u(TILE_SIZE, &rng);
    let pixel = min(workgroup_id.xy * vec2(TILE_SIZE) + vec2(x, y), vec2u(view.viewport.zw - 1.0));

    let depth = textureLoad(depth_buffer, pixel, 0);
    if depth == 0.0 {
        return;
    }
    let gpixel = textureLoad(gbuffer, pixel, 0);
    let world_position = reconstruct_world_position(pixel, depth);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    let sample = generate_light_tree_sample(&rng, world_position, world_normal);

    let i = tile_id * SAMPLES_PER_TILE + sample_index;
    light_tile_samples[i] = sample.light_sample;
    light_tile_resolved_samples[i] = pack_resolved_light_sample(sample.resolved_light_sample);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view.viewport.zw;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = view.world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

struct ResolvedLightSamplePacked {
    world_position_x: f32,
    world_position_y: f32,
    world_position_z: f32,
    world_normal: u32,
    radiance: u32,
    inverse_pdf: f32,
}

fn pack_resolved_light_sample(sample: ResolvedLightSample) -> ResolvedLightSamplePacked {
    return ResolvedLightSamplePacked(
        sample.world_position.x,
        sample.world_position.y,
        sample.world_position.z,
        pack2x16unorm(octahedral_encode(sample.world_normal)),
        vec3_to_rgb9e5_(sample.radiance * view.exposure),
        sample.inverse_pdf * select(1.0, -1.0, sample.world_position.w == 0.0),
    );
}

fn unpack_resolved_light_sample(packed: ResolvedLightSamplePacked, exposure: f32) -> ResolvedLightSample {
    return ResolvedLightSample(
        vec4(packed.world_position_x, packed.world_position_y, packed.world_position_z, select(1.0, 0.0, packed.inverse_pdf < 0.0)),
        octahedral_decode(unpack2x16unorm(packed.world_normal)),
        rgb9e5_to_vec3_(packed.radiance) / exposure,
        abs(packed.inverse_pdf),
    );
}
