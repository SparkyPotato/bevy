#define_import_path bevy_solari::sampling

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI, PI_2, orthonormalize}
#import bevy_solari::scene_bindings::{trace_ray, RAY_T_MIN, RAY_T_MAX, light_tree, light_sources, directional_lights, SgLight, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, resolve_triangle_data_full}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec28%3A303
fn sample_cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = 1.0 - 2.0 * rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = normal.x + sin_theta * cos(phi);
    let y = normal.y + sin_theta * sin(phi);
    let z = normal.z + cos_theta;
    return vec3(x, y, z);
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformlySamplingaHemisphere
fn sample_uniform_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = cos_theta;
    return orthonormalize(normal) * vec3(x, y, z);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec19%3A294
fn sample_disk(disk_radius: f32, rng: ptr<function, u32>) -> vec2<f32> {
    let ab = 2.0 * rand_vec2f(rng) - 1.0;
    let a = ab.x;
    var b = ab.y;
    if (b == 0.0) { b = 1.0; }

    var phi: f32;
    var r: f32;
    if (a * a > b * b) {
        r = disk_radius * a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = disk_radius * b;
        phi = (PI / 2.0) - (PI / 4.0) * (a / b);
    }

    let x = r * cos(phi);
    let y = r * sin(phi);
    return vec2(x, y);
}

struct LightSample {
    light_id: u32,
    seed: u32,
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
}

struct LightContributionNoPdf {
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

fn sample_with_light_tree(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_light_tree_sample(rng, ray_origin, origin_world_normal);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

fn generate_light_tree_sample(rng: ptr<function, u32>, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> GenerateRandomLightSampleResult {
    let light_tree_sample = sample_light_tree(rand_f(rng), ray_origin, origin_world_normal);
    let light_id = light_tree_sample.light_id;
    let seed = rand_u(rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = seed % triangle_count;
    }

    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= light_tree_sample.inverse_pdf;

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let light_count = arrayLength(&light_sources);
    let light_id = rand_range_u(light_count, rng);
    let seed = rand_u(rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = seed % triangle_count;
    }

    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(light_count);

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

#ifdef DIRECTIONAL_LIGHT_SOFT_SHADOWS
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        var direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
#else
        let direction_to_light = directional_light.direction_to_light;
#endif

        return ResolvedLightSample(
            vec4(direction_to_light, 0.0),
            -direction_to_light,
            directional_light.luminance,
            directional_light.inverse_pdf,
        );
    } else {
        let triangle_count = light_source.kind >> 1u;
        let triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    let light_distance = length(ray);
    let wi = ray / light_distance;

    let cos_theta_origin = saturate(dot(wi, origin_world_normal));
    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    let radiance = resolved_light_sample.radiance * cos_theta_origin * (cos_theta_light / light_distance_squared);

    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi);
}

fn resolve_and_calculate_light_contribution(light_sample: LightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContributionNoPdf {
    let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_sample.light_id >> 16u]);
    let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    return LightContributionNoPdf(light_contribution.radiance, light_contribution.wi);
}

fn trace_light_visibility(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> f32 {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w == 1.0 {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_point_visibility(ray_origin: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - ray_origin;
    let dist = length(ray);
    let ray_direction = ray / dist;

    let ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}

struct LightTreeSample {
    light_id: u32,
    inverse_pdf: f32,
}

fn sample_light_tree(rng_sample: f32, ray_origin: vec3<f32>, ray_origin_world_normal: vec3<f32>) -> LightTreeSample {
    var p = rng_sample;
    var node = light_tree[0];
    var inverse_pdf = 1.0;
    for (var i = 0u; i < 32u; i++) {
        let left_w = weight_light(node.left, ray_origin, ray_origin_world_normal);
        let right_w = weight_light(node.right, ray_origin, ray_origin_world_normal);
        let w_sum = left_w + right_w;
        if w_sum < 0.0001 {
            return LightTreeSample(0, 0.0);
        }
        let left_p = left_w / w_sum;
        let right_p = 1.0 - left_p;
        if p < left_p {
            inverse_pdf /= left_p;
            if node.left_index == ~0u {
                return LightTreeSample(0, 0.0);
            } else if node.left_index >> 31u == 1u {
                return LightTreeSample(node.left_index & ~(1u << 31u), inverse_pdf);
            }
            node = light_tree[node.left_index];
            p /= left_p;
        } else {
            inverse_pdf /= right_p;
            if node.right_index == ~0u {
                return LightTreeSample(0, 0.0);
            } else if node.right_index >> 31u == 1u {
                return LightTreeSample(node.right_index & ~(1u << 31u), inverse_pdf);
            }
            node = light_tree[node.right_index];
            p = (p - left_p) / right_p;
        }
    }
    return LightTreeSample(0, 0.0);
}

// https://github.com/yusuketokuyoshi/VSGL/blob/master/VSGL/Shaders/LightingPS.hlsl
fn weight_light(light: SgLight, ray_origin: vec3<f32>, ray_origin_world_normal: vec3<f32>) -> f32 {
	let dir = light.pos - ray_origin;
	let t2 = dot(dir, dir);
	let wi = dir / sqrt(t2);
    let variance = max(light.variance, t2 / 0x1.0p41);
	let emissive = light.intensity / variance;
	let sharpness = t2 / variance;
	let lobe = sg_product(light.axis, light.sharpness, wi, sharpness);

	let amplitude = exp(lobe.log_amplitude);
	let cosine = clamp(dot(lobe.axis, ray_origin_world_normal), -1.0, 1.0);
	let diffuse_illum = amplitude * sg_clamped_cosine_product_integral_over_pi(cosine, lobe.sharpness);
	return diffuse_illum * luminance(emissive);
}

struct SgLobe {
    axis: vec3<f32>,
    sharpness: f32,
    log_amplitude: f32,
}

// https://github.com/yusuketokuyoshi/VSGL/blob/master/VSGL/Shaders/SphericalGaussian.hlsli
fn sg_product(axis1: vec3<f32>, sharpness1: f32, axis2: vec3<f32>, sharpness2: f32) -> SgLobe {
    let axis = axis1 * sharpness1 + axis2 * sharpness2;
    let sharpness = length(axis);

    let d = axis1 - axis2;
    let len2 = dot(d, d);
    let log_amplitude = -sharpness1 * sharpness2 * len2 / max(sharpness + sharpness1 + sharpness2, 1.175494351e-38);
    return SgLobe(axis / max(sharpness, 1.175494351e-38), sharpness, log_amplitude);
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 7]
fn sg_clamped_cosine_product_integral_over_pi(cosine: f32, sharpness: f32) -> f32 {
	let A = 2.7360831611272558028247203765204;
	let B = 17.02129778174187535455530451145;
	let C = 4.0100826728510421403939290030394;
	let D = 15.219156263147210594866010069381;
	let E = 76.087896272360737270901154261082;
	let t = sharpness * sqrt(0.5 * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
	let tz = t * cosine;

	let INV_SQRTPI = 0.56418958354775628694807945156077;  // = 1/sqrt(pi).
	let CLAMPING_THRESHOLD = 0.5 * 1.192092896e-07;		  // Set zero if a precise erfc function is available.
	let lerpFactor =
		saturate(max(0.5 * (cosine * erfc(-tz) + erfc(t)) -
						 0.5 * INV_SQRTPI * exp(-tz * tz) * expm1(t * t * (cosine * cosine - 1.0)) / t,
					 CLAMPING_THRESHOLD));

	// Interpolation between lower and upper hemispherical integrals.
	let lowerIntegral = lower_sg_clamped_cosine_integral_over_two_pi(sharpness);
	let upperIntegral = upper_sg_clamped_cosine_integral_over_two_pi(sharpness);
	return 2.0 * mix(lowerIntegral, upperIntegral, lerpFactor);
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 5]
fn upper_sg_clamped_cosine_integral_over_two_pi(sharpness: f32) -> f32 {
	if (sharpness <= 0.5) {
		return (((((((-1.0 / 362880.0) * sharpness + 1.0 / 40320.0) * sharpness - 1.0 / 5040.0) * sharpness +
				   1.0 / 720.0) *
					  sharpness -
				  1.0 / 120.0) *
					 sharpness +
				 1.0 / 24.0) *
					sharpness -
				1.0 / 6.0) *
				   sharpness +
			   0.5;
	}

	return (expm1(-sharpness) + sharpness) / (sharpness * sharpness);
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary
// Document)" Listing. 6]
fn lower_sg_clamped_cosine_integral_over_two_pi(sharpness: f32) -> f32 {
	let e = exp(-sharpness);

	if (sharpness <= 0.5) {
		return e * (((((((((1.0 / 403200.0) * sharpness - 1.0 / 45360.0) * sharpness + 1.0 / 5760.0) * sharpness -
						 1.0 / 840.0) *
							sharpness +
						1.0 / 144.0) *
						   sharpness -
					   1.0 / 30.0) *
						  sharpness +
					  1.0 / 8.0) *
						 sharpness -
					 1.0 / 3.0) *
						sharpness +
					0.5);
	}

	return e * (-expm1(-sharpness) - sharpness * e) / (sharpness * sharpness);
}

fn mulsign(x: f32, y: f32) -> f32 {
	return bitcast<f32>((bitcast<u32>(y) & 0x80000000) ^ bitcast<u32>(x));
}

// exp(x) - 1 with cancellation of rounding errors.
// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p.19]
fn expm1(x: f32) -> f32 {
	let u = exp(x);

	if (u == 1.0) {
		return x;
	}

	let y = u - 1.0;

	if (abs(x) < 1.0) {
		return y * x / log(u);
	}

	return y;
}

fn erf(x: f32) -> f32 {
	// Early return for large |x|.
	if (abs(x) >= 4.0) {
		return mulsign(1.0, x);
	}

	// Polynomial approximation based on the approximation posted in
	// https://forums.developer.nvidia.com/t/optimized-version-of-single-precision-error-function-erff/40977
	if (abs(x) > 1.0) {
		// The maximum error is smaller than the approximation described in Abramowitz and Stegun [1964 "Handbook of
		// Mathematical Functions with Formulas, Graphs, and Mathematical Tables", 7.1.26, p.299].
		let A1 = 1.628459513;
		let A2 = 9.15674746e-1;
		let A3 = 1.54329389e-1;
		let A4 = -3.51759829e-2;
		let A5 = 5.66795561e-3;
		let A6 = -5.64874616e-4;
		let A7 = 2.58907676e-5;
		let a = abs(x);
		let y = 1.0 - exp2(-(((((((A7 * a + A6) * a + A5) * a + A4) * a + A3) * a + A2) * a + A1) * a));

		return mulsign(y, x);
	}

	// The maximum error is smaller than the 6th order Taylor polynomial.
	let A1 = 1.128379121;
	let A2 = -3.76123011e-1;
	let A3 = 1.12799220e-1;
	let A4 = -2.67030653e-2;
	let A5 = 4.90735564e-3;
	let A6 = -5.58853149e-4;
	let x2 = x * x;

	return (((((A6 * x2 + A5) * x2 + A4) * x2 + A3) * x2 + A2) * x2 + A1) * x;
}

// Complementary error function erfc(x) = 1 - erf(x).
// This implementation can have a numerical error for large x.
fn erfc(x: f32) -> f32 {
	return 1.0 - erf(x);
}
