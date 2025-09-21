// Somewhat inspired by https://advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf

#define_import_path bevy_solari::light_cache

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::rand_f
#import bevy_solari::brdf::evaluate_brdf
#import bevy_solari::sampling::{light_contribution_no_trace, select_random_light, select_random_light_inverse_pdf, trace_light_visibility}
#import bevy_solari::scene_bindings::ResolvedMaterial

/// Maximum amount of attempts to find a cache entry after a hash collision
const LIGHT_CACHE_MAX_SEARCH_STEPS: u32 = 3u;
/// Maximum lights stored in each cache cell
const LIGHT_CACHE_CELL_LIGHT_COUNT: u32 = 8u;
/// Lights searched that aren't in the cell
const LIGHT_CACHE_NEW_LIGHTS_SEARCH_COUNT: u32 = 2u;

/// The size of a cache cell at the lowest LOD in meters
const LIGHT_CACHE_POSITION_BASE_CELL_SIZE: f32 = 0.25;
/// How fast the world cache transitions between LODs as a function of distance to the camera
const LIGHT_CACHE_POSITION_LOD_SCALE: f32 = 30.0;

/// Marker value for an empty cell
const LIGHT_CACHE_EMPTY_CELL: u32 = 0u;

struct LightData {
    light: u32,
    weight: f32,
}

// The size of these structs should match `LIGHT_CACHE_CELL_SIZE` in `scene/binder.rs`!
struct LightCacheCellRead {
    visible_light_count: u32,
    visible_lights: array<LightData, LIGHT_CACHE_CELL_LIGHT_COUNT>,
}

struct LightCacheCellWrite {
    visible_light_count: atomic<u32>,
    visible_lights: array<u64, LIGHT_CACHE_CELL_LIGHT_COUNT>,
}

@group(0) @binding(12) var<storage, read> light_cache_checksums_read: array<u32, #{LIGHT_CACHE_SIZE}>;
@group(0) @binding(13) var<storage, read_write> light_cache_checksums_write: array<atomic<u32>, #{LIGHT_CACHE_SIZE}>;
@group(0) @binding(14) var<storage, read> light_cache_cells_read: array<LightCacheCellRead, #{LIGHT_CACHE_SIZE}>;
@group(0) @binding(15) var<storage, read_write> light_cache_cells_write: array<LightCacheCellWrite, #{LIGHT_CACHE_SIZE}>;

fn query_light_cache(world_position: vec3<f32>, view_position: vec3<f32>) -> LightCacheCellRead {
    let cell_size = get_cell_size(world_position, view_position);
    let world_position_quantized = bitcast<vec3<u32>>(quantize_position(world_position, cell_size));
    var key = compute_key(world_position_quantized);
    let checksum = compute_checksum(world_position_quantized);

    for (var i = 0u; i < LIGHT_CACHE_MAX_SEARCH_STEPS; i++) {
        let existing_checksum = light_cache_checksums_read[key];
        if existing_checksum == checksum {
            return light_cache_cells_read[key];
        } else if existing_checksum == LIGHT_CACHE_EMPTY_CELL {
            return LightCacheCellRead(0, array<LightData, LIGHT_CACHE_CELL_LIGHT_COUNT>());
        } else {
            // Collision - jump to another entry
            key = wrap_key(pcg_hash(key));
        }
    }

    return LightCacheCellRead(0, array<LightData, LIGHT_CACHE_CELL_LIGHT_COUNT>());
}

fn write_light_cache(data: LightCacheCellRead, world_position: vec3<f32>, view_position: vec3<f32>) {
    let cell_size = get_cell_size(world_position, view_position);
    let world_position_quantized = bitcast<vec3<u32>>(quantize_position(world_position, cell_size));
    var key = compute_key(world_position_quantized);
    let checksum = compute_checksum(world_position_quantized);

    for (var i = 0u; i < LIGHT_CACHE_MAX_SEARCH_STEPS; i++) {
        let existing_checksum = atomicCompareExchangeWeak(&light_cache_checksums_write[key], LIGHT_CACHE_EMPTY_CELL, checksum).old_value;
        if existing_checksum == checksum || existing_checksum == LIGHT_CACHE_EMPTY_CELL {
        } else {
            // Collision - jump to another entry
            key = wrap_key(pcg_hash(key));
        }
    }
}

struct EvaluatedLighting {
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

fn evaluate_lighting_from_cache(
    rng: ptr<function, u32>, 
    cell: ptr<function, LightCacheCellRead>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    wo: vec3<f32>,
    material: ResolvedMaterial,
    exposure: f32,
) -> EvaluatedLighting {
    let cell_selected_light = select_light_from_cache_cell(rng, cell, world_position, world_normal, wo, material);
    let cell_selected_weight = cell_selected_light.weight + log2(exposure);
    let cell_confidence = smoothstep(0.1, 0.3, cell_selected_weight);

    // Sample more lights if our cell has bad lights
    let random_sample_count = u32(round(mix(8.0, 2.0, cell_confidence)));
    let random_selected_light = select_light_randomly(rng, world_position, world_normal, wo, material, random_sample_count);
    let p_random_candidate = p_wrs(random_selected_light) * f32(random_sample_count) / select_random_light_inverse_pdf(random_selected_light.light);

    let p_cell_selection = select(p_wrs(cell_selected_light), 0.0, cell_selected_light.weight_sum < 0.0001);
    let p_random_selection = select(p_random_candidate, 0.0, random_selected_light.weight_sum < 0.0001);
    let p_random_selection_clamped = min(mix(1.0, 0.25 * p_cell_selection, cell_confidence), p_random_selection);

    let weight_sum = p_cell_selection + p_random_selection_clamped;
    if weight_sum < 0.0001 {
        return EvaluatedLighting(vec3(0.0), 0.0);
    }

    let p_should_choose_cell = p_cell_selection / weight_sum;
    let p_should_choose_random = 1.0 - p_should_choose_cell;
    var sel: u32;
    var pdf: f32;
    if rand_f(rng) < p_should_choose_cell {
        sel = cell_selected_light.light;
        pdf = p_should_choose_cell * p_cell_selection;
    } else {
        sel = random_selected_light.light;
        pdf = p_should_choose_random * p_random_selection;
    }
    
    // TODO: reuse the eval that we did for light selection somehow
    let direct_lighting = light_contribution_no_trace(rng, sel, world_position, world_normal);
    let brdf = evaluate_brdf(world_normal, wo, direct_lighting.wi, material);
    let radiance = direct_lighting.radiance * brdf * trace_light_visibility(world_position, direct_lighting.world_position);
    let inverse_pdf = direct_lighting.inverse_pdf / pdf;
    return EvaluatedLighting(radiance, inverse_pdf);
}

struct SelectedLight {
    light: u32,
    weight: f32,
    weight_sum: f32,
}

fn p_wrs(selection: SelectedLight) -> f32 {
    return selection.weight / selection.weight_sum;
}

fn select_light_from_cache_cell(
    rng: ptr<function, u32>, 
    cell: ptr<function, LightCacheCellRead>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    wo: vec3<f32>,
    material: ResolvedMaterial
) -> SelectedLight {
    var p = rand_f(rng);
    
    var selected = 0u;
    var selected_weight = 0.0;
    var weight_sum = 0.0;
    // WRS to select the light based on unshadowed contribution
    for (var i = 0u; i < (*cell).visible_light_count; i++) {
        let light_id = (*cell).visible_lights[i].light;
        let direct_lighting = light_contribution_no_trace(rng, light_id, world_position, world_normal);
        let brdf = evaluate_brdf(world_normal, wo, direct_lighting.wi, material);
        // Weight by inverse_pdf to bias towards larger triangles
        let radiance = direct_lighting.radiance * direct_lighting.inverse_pdf * brdf;

        let weight = log2(luminance(radiance) + 1.0);
        cell.visible_lights[i].weight = weight;
        weight_sum += weight;

        let prob = weight / weight_sum;
        if p < prob {
            selected = light_id;
            selected_weight = weight;
            p /= prob;
        } else {
            p = (p - prob) / (1.0 - prob);
        }
    }
    return SelectedLight(selected, selected_weight, weight_sum);
}

fn select_light_randomly(
    rng: ptr<function, u32>, 
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    wo: vec3<f32>,
    material: ResolvedMaterial,
    samples: u32,
) -> SelectedLight { 
    var p = rand_f(rng);

    var selected = 0u;
    var selected_weight = 0.0;
    var weight_sum = 0.0;
    for (var i = 0u; i < samples; i++) {
        let light_id = select_random_light(rng);
        let direct_lighting = light_contribution_no_trace(rng, light_id, world_position, world_normal);
        let brdf = evaluate_brdf(world_normal, wo, direct_lighting.wi, material);
        let radiance = direct_lighting.radiance * direct_lighting.inverse_pdf * brdf;

        let weight = log2(luminance(radiance) + 1.0);
        weight_sum += weight;

        let prob = weight / weight_sum;
        if p < prob {
            selected = light_id;
            selected_weight = weight;
            p /= prob;
        } else {
            p = (p - prob) / (1.0 - prob);
        }
    }
    return SelectedLight(selected, selected_weight, weight_sum);
}

fn get_cell_size(world_position: vec3<f32>, view_position: vec3<f32>) -> f32 {
    let camera_distance = distance(view_position, world_position) / LIGHT_CACHE_POSITION_LOD_SCALE;
    let lod = exp2(floor(log2(1.0 + camera_distance)));
    return LIGHT_CACHE_POSITION_BASE_CELL_SIZE * lod;
}

fn quantize_position(world_position: vec3<f32>, quantization_factor: f32) -> vec3<f32> {
    return floor(world_position / quantization_factor + 0.0001);
}

// TODO: Clustering
fn compute_key(world_position: vec3<u32>) -> u32 {
    var key = pcg_hash(world_position.x);
    key = pcg_hash(key + world_position.y);
    key = pcg_hash(key + world_position.z);
    return wrap_key(key);
}

fn compute_checksum(world_position: vec3<u32>) -> u32 {
    var key = iqint_hash(world_position.x);
    key = iqint_hash(key + world_position.y);
    key = iqint_hash(key + world_position.z);
    return key;
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn iqint_hash(input: u32) -> u32 {
    let n = (input << 13u) ^ input;
    return n * (n * n * 15731u + 789221u) + 1376312589u;
}

fn wrap_key(key: u32) -> u32 {
    return key & (#{LIGHT_CACHE_SIZE} - 1u);
}
