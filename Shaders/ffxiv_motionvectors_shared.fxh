/*=============================================================================
This work is licensed under the 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License
https://creativecommons.org/licenses/by-nc/4.0/	  

Original developer: Jak0bPCoder
Optimization by : Marty McFly
Compatibility by : MJ_Ehsan

alex:
- WS normals as features for ME and upscaling
- Velocity as ME guidance and filter
=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#define UI_ME_MAX_ITERATIONS_PER_LEVEL                  2
#define UI_ME_SAMPLES_PER_ITERATION                     5

uniform int UI_ME_LAYER_MAX <
	ui_type = "combo";
	ui_items = "Full Resolution\0Half Resolution\0Quarter Resolution\0";
	ui_min = 0;
	ui_max = 2;	
> = 1;

uniform float UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;	
> = 4.0;

uniform bool SHOWME <
	ui_label = "Debug Output";	
> = false;

/*=============================================================================
	UI Uniforms
=============================================================================*/
/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);
*/
/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

#include "ReShade.fxh"
#include "ffxiv_common.fxh"

//integer divide, rounding up
#define CEIL_DIV(num, denom) (((num - 1) / denom) + 1)
#define PI 3.14159265
uniform uint FRAME_COUNT < source = "framecount"; >;
uniform float4x4 matPrevViewProj < source = "mat_PrevViewProj"; >;

#define M_PI 3.1415926535

#ifndef PRE_BLOCK_SIZE_2_TO_7
 #define PRE_BLOCK_SIZE_2_TO_7	3   //[2 - 7]     
#endif

#ifndef WS_NORMAL_FEATURES
 #define WS_NORMAL_FEATURES 0 //[0 - 1]
#endif

#define BLOCK_SIZE (PRE_BLOCK_SIZE_2_TO_7) //4

//NEVER change these!!!
#define BLOCK_SIZE_HALF (BLOCK_SIZE * 0.5 - 0.5)//(int(BLOCK_SIZE / 2)) //2
#define BLOCK_AREA 		(BLOCK_SIZE * BLOCK_SIZE) //16

#undef NUM_MIPS
#if (BUFFER_HEIGHT >> 9) > (PRE_BLOCK_SIZE_2_TO_7)
#define NUM_MIPS 10
#elif (BUFFER_HEIGHT >> 8) > (PRE_BLOCK_SIZE_2_TO_7)
#define NUM_MIPS 9
#else
#define NUM_MIPS 8
#endif

//smpG samplers are .r = Grayscale, g = depth
//smgM samplers are .r = motion x, .g = motion y, .b = feature level, .a = loss;

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

/*=============================================================================
	Functions
=============================================================================*/

float GetDepth(float2 texcoords)
{
	return tex2Dlod(ReShade::DepthBuffer, float4(texcoords, 0, 0)).x;
}

float LDepth(float2 texcoord)
{
	return ReShade::GetLinearizedDepth(texcoord);
}

float2 GetMotionVector(float3 position, float d)
{
	float4 previousPositionUV = mul(matPrevViewProj, float4(position, 1));
	float4 currentPositionUV = mul(FFXIV::matViewProj, float4(position, 1));
	
	previousPositionUV *= rcp(previousPositionUV.w);
	currentPositionUV *= rcp(currentPositionUV.w);
	
	return (previousPositionUV.xy - currentPositionUV.xy) * float2(0.5, -0.5);
}

float noise(float2 co)
{
  return frac(sin(dot(co.xy ,float2(1.0,73))) * 437580.5453);
}

//uint fp16_ieee_from_fp32_value(uint x) {
//  uint x_sgn = x & 0x80000000U;
//  uint x_exp = x & 0x7f800000U;
//  x_exp = (x_exp < 0x38800000U) ? 0x38800000U : x_exp;
//  x_exp += 15U << 23;
//  x &= 0x7fffffffU;
//
//  uint f = x;
//  uint magic = x_exp;
//
//  float inf = asfloat(0x77800000U);
//  float zero = asfloat(0x08800000U);
//  f = asuint((asfloat(f) * inf) * zero);
//  f = asuint(asfloat(f) + asfloat(magic));
//  
//  uint h_exp = (f >> 13) & 0x7c00U;
//  uint h_sig = f & 0x0fffU;
//  h_sig = (x > 0x7f800000U) ? 0x0200U : h_sig;
//  return (x_sgn >> 16) + h_exp + h_sig;
//}

#if WS_NORMAL_FEATURES == 0
float4 CalcMotionLayer(VSOUT i, int mip_gcurr, float2 searchStart, sampler sFeatureCurr, sampler sFeaturePrev)
{
	//uint xx = fp16_ieee_from_fp32_value(0);
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float2 localBlock[BLOCK_AREA]; 
	
	float2 moments_local_s = 0;
	float2 moments_search_b = 0;
	float2 moments_search_s = 0;
	float2 moments_c = 0;
	
	float2 vel_moments_local_s = 0;
	float2 vel_moments_search_b = 0;
	float2 vel_moments_search_s = 0;
	float2 vel_moments_c = 0;
	
	float curDepth = GetDepth(i.uv);
	float2 vel = GetMotionVector(FFXIV::get_world_position_from_uv(i.uv - texelsize * 0.5, curDepth), curDepth);

	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float2 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr)).rg;
		float2 t_search = tex2Dlod(sFeaturePrev, float4(tuv + searchStart, 0, mip_gcurr)).rg;
		float2 vel_t_search = tex2Dlod(sFeaturePrev, float4(tuv + vel, 0, mip_gcurr)).rg;
		localBlock[k] = t_local; 

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
		
		vel_moments_local_s += t_local * t_local;
		vel_moments_search_b += vel_t_search;
		vel_moments_search_s += vel_t_search * vel_t_search;
		vel_moments_c += t_local * vel_t_search;
	}

	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;
	
	vel_moments_local_s /= BLOCK_AREA;
	vel_moments_search_b /= BLOCK_AREA;
	vel_moments_search_s /= BLOCK_AREA;
	vel_moments_c /= BLOCK_AREA;
	
	float2 vel_cossim = vel_moments_c * rsqrt(vel_moments_local_s * vel_moments_search_s);
	float vel_best_sim = saturate(min(vel_cossim.x, vel_cossim.y));

	float vel_best_features = abs(vel_moments_search_b.x * vel_moments_search_b.x - vel_moments_search_s.x);

	float2 cossim = moments_c * rsqrt(moments_local_s * moments_search_s);
	float best_sim = saturate(min(cossim.x, cossim.y));

	float best_features = abs(moments_search_b.x * moments_search_b.x - moments_search_s.x);

	float2 bestMotion = 0;
	float2 searchCenter = searchStart;

	float randseed = noise(i.uv);
	randseed = frac(randseed + (FRAME_COUNT % 16) * UI_ME_SAMPLES_PER_ITERATION * UI_ME_MAX_ITERATIONS_PER_LEVEL * 0.6180339887498);
	float2 randdir; sincos(randseed * 6.283, randdir.x, randdir.y); //yo dawg, I heard you like golden ratios
	float2 scale = texelsize;

	[loop]
	for(int j = 0; j < UI_ME_MAX_ITERATIONS_PER_LEVEL && best_sim < 0.999999; j++)
	{
		[loop]
		for (int s = 0; s < UI_ME_SAMPLES_PER_ITERATION && best_sim < 0.999999; s++) 
		{
			randdir = mul(randdir, float2x2(-0.7373688, 0.6754903, -0.6754903, -0.7373688));//rotate by larger golden angle			
			float2 pixelOffset = randdir * scale;
			float2 samplePos = i.uv + searchCenter + pixelOffset;			 

			float2 moments_candidate_b = 0;
			float2 moments_candidate_s = 0;
			moments_c = 0;

			[loop]
			for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
			{
				float2 t = tex2Dlod(sFeaturePrev, float4(samplePos + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize, 0, mip_gcurr)).rg;
				moments_candidate_b += t;
				moments_candidate_s += t * t;
				moments_c += t * localBlock[k];
			}
			
			moments_candidate_b /= BLOCK_AREA;
			moments_candidate_s /= BLOCK_AREA;
			moments_c /= BLOCK_AREA;
			
			cossim = moments_c * rsqrt(moments_local_s * moments_candidate_s);
			float candidate_similarity = saturate(min(cossim.x, cossim.y));

			[flatten]
			if(candidate_similarity > best_sim)					
			{
				best_sim = candidate_similarity;
				bestMotion = pixelOffset;
				best_features = abs(moments_candidate_b.x * moments_candidate_b.x - moments_candidate_s.x);
			}			
		}
		searchCenter += bestMotion;
		bestMotion = 0;
		scale *= 0.5;
	}

	if(vel_best_sim > best_sim)
	{
		return float4(vel, sqrt(vel_best_features), vel_best_sim * vel_best_sim * vel_best_sim * vel_best_sim);
	}
	
	return float4(searchCenter, sqrt(best_features), best_sim * best_sim * best_sim * best_sim);  //delayed sqrt for variance -> stddev, cossim^4 for filter
}

float4 atrous_upscale(VSOUT i, int mip_gcurr, sampler sMotionLow, sampler sFeatureCurr)
{	
    float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr + 1));	
	float rand = frac(mip_gcurr * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS;
	const float3 gauss = float3(1, 0.85, 0.65);

	float center_n = tex2Dlod(sFeatureCurr, float4(i.uv, 0, mip_gcurr)).y;

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;

	for(int x = -1; x <= 1; x++)
	for(int y = -1; y <= 1; y++)
	{
		float2 offs = mul(float2(x, y), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;

		float sample_n = tex2Dlod(sFeatureCurr, float4(sample_uv, 0, mip_gcurr + 1)).y;
		float4 sample_gbuf = tex2Dlod(sMotionLow, float4(sample_uv, 0, 0));

		float wn = abs(sample_n - center_n) * 256.0;
		//float wns = (wn.x * wn.y * wn.z) * 128.0;
		float ws = saturate(1.0 - sample_gbuf.w * 4.0);
		float wf = saturate(1.0 - sample_gbuf.z * 4.0);
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4.0;

		float weight = exp2(-(wn + ws + wm + wf)) * gauss[abs(x)] * gauss[abs(y)];
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;		
	}

	return gbuffer_sum / wsum;	
}
#else
float4 CalcMotionLayer(VSOUT i, int mip_gcurr, float2 searchStart, sampler sFeatureCurr, sampler sFeaturePrev)
{	
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float4 localBlock[BLOCK_AREA]; 
	
	float4 moments_local_s = 0;
	float4 moments_search_b = 0;
	float4 moments_search_s = 0;
	float4 moments_c = 0;
	
	float4 vel_moments_local_s = 0;
	float4 vel_moments_search_b = 0;
	float4 vel_moments_search_s = 0;
	float4 vel_moments_c = 0;
	
	float curDepth = GetDepth(i.uv);
	float2 vel = GetMotionVector(FFXIV::get_world_position_from_uv(i.uv - texelsize * 0.5, curDepth), curDepth);

	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float4 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr));
		float4 t_search = tex2Dlod(sFeaturePrev, float4(tuv + searchStart, 0, mip_gcurr));
		float4 vel_t_search = tex2Dlod(sFeaturePrev, float4(tuv + vel, 0, mip_gcurr));
		localBlock[k] = t_local; 

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
		
		vel_moments_local_s += t_local * t_local;
		vel_moments_search_b += vel_t_search;
		vel_moments_search_s += vel_t_search * vel_t_search;
		vel_moments_c += t_local * vel_t_search;
	}

	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;
	
	vel_moments_local_s /= BLOCK_AREA;
	vel_moments_search_b /= BLOCK_AREA;
	vel_moments_search_s /= BLOCK_AREA;
	vel_moments_c /= BLOCK_AREA;
	
	float4 vel_cossim = vel_moments_c * rsqrt(vel_moments_local_s * vel_moments_search_s);
	float vel_best_sim = saturate(min(vel_cossim.x, vel_cossim.y * vel_cossim.z * vel_cossim.w));

	float vel_best_features = abs(vel_moments_search_b.x * vel_moments_search_b.x - vel_moments_search_s.x);

	float4 cossim = moments_c * rsqrt(moments_local_s * moments_search_s);
	float best_sim = saturate(min(cossim.x, cossim.y * cossim.z * cossim.w)); //saturate(min(min(min(cossim.x, cossim.y), cossim.z), cossim.w));

	float best_features = abs(moments_search_b.x * moments_search_b.x - moments_search_s.x);

	float2 bestMotion = 0;
	float2 searchCenter = searchStart;    

	float randseed = noise(i.uv);
	randseed = frac(randseed + (FRAME_COUNT % 16) * UI_ME_SAMPLES_PER_ITERATION * UI_ME_MAX_ITERATIONS_PER_LEVEL * 0.6180339887498);
	float2 randdir; sincos(randseed * 6.283, randdir.x, randdir.y); //yo dawg, I heard you like golden ratios
	float2 scale = texelsize;

	[loop]
	for(int j = 0; j < UI_ME_MAX_ITERATIONS_PER_LEVEL && best_sim < 0.999999; j++)
	{
		[loop]
		for (int s = 0; s < UI_ME_SAMPLES_PER_ITERATION && best_sim < 0.999999; s++) 
		{
			randdir = mul(randdir, float2x2(-0.7373688, 0.6754903, -0.6754903, -0.7373688));//rotate by larger golden angle			
			float2 pixelOffset = randdir * scale;
			float2 samplePos = i.uv + searchCenter + pixelOffset;			 

			float4 moments_candidate_b = 0;
			float4 moments_candidate_s = 0;
			moments_c = 0;

			[loop]
			for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
			{
				float4 t = tex2Dlod(sFeaturePrev, float4(samplePos + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize, 0, mip_gcurr));
				moments_candidate_b += t;
				moments_candidate_s += t * t;
				moments_c += t * localBlock[k];
			}
			
			moments_candidate_b /= BLOCK_AREA;
			moments_candidate_s /= BLOCK_AREA;
			moments_c /= BLOCK_AREA;
			
			cossim = moments_c * rsqrt(moments_local_s * moments_candidate_s);
			float candidate_similarity = saturate(min(cossim.x, cossim.y * cossim.z * cossim.w));//saturate(min(min(min(cossim.x, cossim.y), cossim.z), cossim.w));

			[flatten]
			if(candidate_similarity > best_sim)					
			{
				best_sim = candidate_similarity;
				bestMotion = pixelOffset;
				best_features = abs(moments_candidate_b.x * moments_candidate_b.x - moments_candidate_s.x);
			}			
		}
		searchCenter += bestMotion;
		bestMotion = 0;
		scale *= 0.5;
	}
	
	if(vel_best_sim > best_sim)
	{
		return float4(vel, sqrt(vel_best_features), vel_best_sim * vel_best_sim * vel_best_sim * vel_best_sim);
	}

	return float4(searchCenter, sqrt(best_features), best_sim * best_sim * best_sim * best_sim);  //delayed sqrt for variance -> stddev, cossim^4 for filter
}

float4 atrous_upscale(VSOUT i, int mip_gcurr, sampler sMotionLow, sampler sFeatureCurr)
{	
    float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr + 1));	
	float rand = frac(mip_gcurr * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS;
	const float3 gauss = float3(1, 0.85, 0.65);

	float4 center_n = tex2Dlod(sFeatureCurr, float4(i.uv, 0, mip_gcurr));

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;

	for(int x = -1; x <= 1; x++)
	for(int y = -1; y <= 1; y++)
	{
		float2 offs = mul(float2(x, y), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;

		float4 sample_n = tex2Dlod(sFeatureCurr, float4(sample_uv, 0, mip_gcurr + 1));
		float4 sample_gbuf = tex2Dlod(sMotionLow, float4(sample_uv, 0, 0));

		float wn = dot(sample_n.yzw, center_n.yzw) * 4.0;
		//float wns = (wn.x * wn.y * wn.z) * 128.0;
		float ws = saturate(1 - sample_gbuf.w * 4.0);
		//float wf = saturate(1 - sample_gbuf.b * 4.0);
		float wf = abs(sample_n.x - center_n.x) * 128.0;
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4.0;

		//float weight = exp2(-(wns + ws + wm + wf)) * gauss[abs(x)] * gauss[abs(y)];
		float weight = exp2(-(wn + ws + wm + wf)) * gauss[abs(x)] * gauss[abs(y)];
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;		
	}

	return gbuffer_sum / wsum;	
}
#endif

/*=============================================================================
	Shader Entry Points
=============================================================================*/

#if WS_NORMAL_FEATURES == 0
void PSWriteColorAndDepth(in VSOUT i, out float2 o : SV_Target0)
{
	float luma = dot(tex2D(ReShade::BackBuffer, i.uv).rgb, float3(0.299, 0.587, 0.114));
    o = float2(luma, LDepth(i.uv));
}
#else
void PSWriteColorAndDepth(in VSOUT i, out float4 o : SV_Target0)
{    
	float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
	float3 onormal = FFXIV::get_normal(i.uv);
	float3 normal = mul(matV, normalize(onormal - 0.5));
	
	float luma = dot(tex2D(ReShade::BackBuffer, i.uv).rgb, float3(0.299, 0.587, 0.114));
    o = float4(luma, ((normal.rgb + 1.0) * 0.5));
}
#endif

float4 motion_estimation(in VSOUT i, sampler sMotionLow, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr)
{
	float4 upscaledLowerLayer = 0;

	upscaledLowerLayer = atrous_upscale(i, mip_gcurr, sMotionLow, sFeatureCurr);
	
	[branch]
	if(mip_gcurr >= UI_ME_LAYER_MAX)
	{
		upscaledLowerLayer = CalcMotionLayer(i, mip_gcurr, upscaledLowerLayer.xy, sFeatureCurr, sFeaturePrev);
	}
	
	return upscaledLowerLayer;
}

float4 motion_estimation_init(in VSOUT i, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr)
{
	return CalcMotionLayer(i, mip_gcurr, 0, sFeatureCurr, sFeaturePrev);
}
