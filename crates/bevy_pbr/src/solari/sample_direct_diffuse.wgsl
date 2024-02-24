#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_pbr::solari::bindings::{trace_ray, resolve_ray_hit, light_sources, sample_light_sources, trace_light_source, RAY_T_MIN, RAY_T_MAX}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::utils::{rand_f, rand_range_u, octahedral_decode}
#import bevy_core_pipeline::tonemapping::tonemapping_luminance

@group(2) @binding(0) var direct_diffuse: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var gbuffer: texture_2d<u32>;
@group(2) @binding(3) var depth_buffer: texture_depth_2d;
@group(2) @binding(4) var<uniform> view: View;
@group(2) @binding(5) var<uniform> globals: Globals;

struct Reservoir {
    light_id: u32,
    light_rng: u32,
    light_weight: f32,
    weight_sum: f32,
    sample_count: u32
}

fn update_reservoir(reservoir: ptr<function, Reservoir>, light_id: u32, light_rng: u32, light_weight: f32, rng: ptr<function, u32>) {
    (*reservoir).weight_sum += light_weight;
    (*reservoir).sample_count += 1u;
    if rand_f(rng) < light_weight / (*reservoir).weight_sum {
        (*reservoir).light_id = light_id;
        (*reservoir).light_rng = light_rng;
    }
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view.viewport.zw;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = view.inverse_view_proj * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

// TODO: p_hat should be based on irradiance, not radiance, i.e. account for cos(theta) and BRDF terms

@compute @workgroup_size(8, 8, 1)
fn sample_direct_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    let frame_index = globals.frame_count * 5782582u;
    var rng = pixel_index + frame_index;

    let gpixel = textureLoad(gbuffer, global_id.xy, 0i);
    let depth = textureLoad(depth_buffer, global_id.xy, 0i);
    let world_position = reconstruct_world_position(global_id.xy, depth);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    var reservoir = Reservoir(0u, 0u, 0.0, 0.0, 0u);
    let light_count = arrayLength(&light_sources);
    for (var i = 0u; i < 32u; i++) {
        let light_id = rand_range_u(light_count, &rng);
        let light = light_sources[light_id];

        let light_rng = rng;
        let sample = sample_light_sources(light_id, light_count, world_position, world_normal, &rng);
        let p_hat = tonemapping_luminance(sample.radiance);
        let light_weight = p_hat / sample.pdf;

        update_reservoir(&reservoir, light_id, light_rng, light_weight, &rng);
    }

    rng = reservoir.light_rng;
    var radiance = trace_light_source(reservoir.light_id, world_position, world_normal, &rng);

    let p_hat = tonemapping_luminance(radiance);
    let w = reservoir.weight_sum / (p_hat * f32(reservoir.sample_count));
    reservoir.light_weight = select(0.0, w, p_hat > 0.0);

    radiance *= reservoir.light_weight;
    radiance *= view.exposure;

    textureStore(direct_diffuse, global_id.xy, vec4(radiance, 1.0));
    textureStore(view_output, global_id.xy, vec4(radiance, 1.0));
}