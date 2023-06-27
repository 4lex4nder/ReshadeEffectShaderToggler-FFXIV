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
- Adjusted namespaces and texture names for iMMERSE compatibility
- Encoded normal output (octahedron encoding: https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html) for iMMERSE effects
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

uniform int UI_ME_LAYER_MIN <
	ui_min = 0; ui_max = 6; ui_step = 1;
	ui_type = "slider";
	ui_label = "Motion Estimation Range";
	ui_tooltip = "The lowest Layer / Resolution Motion Estimation is performed on. Actual range is (2^Range).\n\
Motion Estimation will break down and will produce inacurate Motion Vectors if a Pixel moves close to or more than (2^Range)\n\
LOW PERFORMANCE IMPACT";
> = 6;

uniform float UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;	
> = 4.0;

uniform bool SHOWME <
	ui_label = "Debug Output";	
> =false;

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

//#define BLOCK_SIZE (PRE_BLOCK_SIZE_2_TO_7) //4

//NEVER change these!!!
//#define BLOCK_SIZE_HALF (BLOCK_SIZE * 0.5 - 0.5)//(int(BLOCK_SIZE / 2)) //2
//#define BLOCK_AREA 		(BLOCK_SIZE * BLOCK_SIZE) //16

//smpG samplers are .r = Grayscale, g = depth
//smgM samplers are .r = motion x, .g = motion y, .b = feature level, .a = loss;

namespace FFXIV_Crashpad {

#if WS_NORMAL_FEATURES == 0
texture FeatureCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; MipLevels = 8; };
texture FeaturePrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; MipLevels = 8; };
#else
texture FeatureCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 8; };
texture FeaturePrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 8; };
#endif

texture MotionTexCur7               { Width = BUFFER_WIDTH >> 7; Height = BUFFER_HEIGHT >> 7; Format = RGBA16F; };
texture MotionTexCur6               { Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGBA16F; };
texture MotionTexCur5               { Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGBA16F; };
texture MotionTexCur4               { Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGBA16F; };
texture MotionTexCur3               { Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGBA16F; };
texture MotionTexCur2               { Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGBA16F; };
texture MotionTexCur1               { Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };
texture MotionTexCur0               { Width = BUFFER_WIDTH >> 0; Height = BUFFER_HEIGHT >> 0; Format = RGBA16F; };
}

sampler sFeatureCurr         { Texture = FFXIV_Crashpad::FeatureCurr; };
sampler sFeaturePrev         { Texture = FFXIV_Crashpad::FeaturePrev; };

namespace Deferred {
	texture MotionVectorsTex          { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
	sampler sMotionVectorsTex         { Texture = MotionVectorsTex;  };
	
	texture NormalsTex              { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG8; };
	sampler sNormalsTex             { Texture = NormalsTex; };
}

sampler sMotionTexCur7              { Texture = FFXIV_Crashpad::MotionTexCur7; };
sampler sMotionTexCur6              { Texture = FFXIV_Crashpad::MotionTexCur6; };
sampler sMotionTexCur5              { Texture = FFXIV_Crashpad::MotionTexCur5; };
sampler sMotionTexCur4              { Texture = FFXIV_Crashpad::MotionTexCur4; };
sampler sMotionTexCur3              { Texture = FFXIV_Crashpad::MotionTexCur3; };
sampler sMotionTexCur2              { Texture = FFXIV_Crashpad::MotionTexCur2; };
sampler sMotionTexCur1              { Texture = FFXIV_Crashpad::MotionTexCur1; };
sampler sMotionTexCur0              { Texture = FFXIV_Crashpad::MotionTexCur0; };

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
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

#if WS_NORMAL_FEATURES == 0
float4 CalcMotionLayer(VSOUT i, int mip_gcurr, float2 searchStart, sampler sFeatureCurr, sampler sFeaturePrev, uint BLOCK_SIZE)
{	
	//NEVER change these!!!
	float BLOCK_SIZE_HALF = (BLOCK_SIZE * 0.5 - 0.5);//(int(BLOCK_SIZE / 2)) //2
	uint BLOCK_AREA = 		(BLOCK_SIZE * BLOCK_SIZE); //16
	const uint BLOCK_AREA_ARR = 		(PRE_BLOCK_SIZE_2_TO_7 * PRE_BLOCK_SIZE_2_TO_7 * 8 * 8); //16

	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float2 localBlock[BLOCK_AREA_ARR]; 
	
	float2 moments_local_s = 0;
	float2 moments_search_b = 0;
	float2 moments_search_s = 0;
	float2 moments_c = 0;

	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float2 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr)).rg;
		float2 t_search = tex2Dlod(sFeaturePrev, float4(tuv + searchStart, 0, mip_gcurr)).rg;
		localBlock[k] = t_local; 

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
	}

	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;

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

	return float4(searchCenter, sqrt(best_features), best_sim * best_sim * best_sim * best_sim);  //delayed sqrt for variance -> stddev, cossim^4 for filter
}

float4 CalcVelocityLayer(in VSOUT i, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr, uint BLOCK_SIZE)
{
	float curDepth = GetDepth(i.uv);
	
	float2 vel = GetMotionVector(FFXIV::get_world_position_from_uv(i.uv, curDepth), curDepth);
	
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float BLOCK_SIZE_HALF = (BLOCK_SIZE * 0.5 - 0.5);//(int(BLOCK_SIZE / 2)) //2
	uint BLOCK_AREA = 		(BLOCK_SIZE * BLOCK_SIZE); //16
	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline
	
	float2 moments_local_s = 0;
	float2 moments_search_b = 0;
	float2 moments_search_s = 0;
	float2 moments_c = 0;

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float2 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr)).rg;
		float2 t_search = tex2Dlod(sFeaturePrev, float4(tuv + vel, 0, mip_gcurr)).rg;

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
	}
	
	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;

	float2 cossim = moments_c * rsqrt(moments_local_s * moments_search_s);
	float best_sim = saturate(min(cossim.x, cossim.y));

	float best_features = abs(moments_search_b.x * moments_search_b.x - moments_search_s.x);
	
	return float4(vel, best_features, best_sim * best_sim * best_sim * best_sim);
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

		float wn = abs(sample_n - center_n) * 128.0;
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
float4 CalcMotionLayer(VSOUT i, int mip_gcurr, float2 searchStart, sampler sFeatureCurr, sampler sFeaturePrev, uint BLOCK_SIZE)
{	
	//NEVER change these!!!
	float BLOCK_SIZE_HALF = (BLOCK_SIZE * 0.5 - 0.5);//(int(BLOCK_SIZE / 2)) //2
	uint BLOCK_AREA = 		(BLOCK_SIZE * BLOCK_SIZE); //16
	const uint BLOCK_AREA_ARR = 		(PRE_BLOCK_SIZE_2_TO_7 * PRE_BLOCK_SIZE_2_TO_7 * 8 * 8); //16

	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float4 localBlock[BLOCK_AREA_ARR]; 
	
	float4 moments_local_s = 0;
	float4 moments_search_b = 0;
	float4 moments_search_s = 0;
	float4 moments_c = 0;

	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float4 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr));
		float4 t_search = tex2Dlod(sFeaturePrev, float4(tuv + searchStart, 0, mip_gcurr));
		localBlock[k] = t_local; 

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
	}

	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;

	float4 cossim = moments_c * rsqrt(moments_local_s * moments_search_s);
	float best_sim = saturate(min(min(min(cossim.x, cossim.y), cossim.z), cossim.w));

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
			float candidate_similarity = saturate(min(min(min(cossim.x, cossim.y), cossim.z), cossim.w));

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

	return float4(searchCenter, sqrt(best_features), best_sim * best_sim * best_sim * best_sim);  //delayed sqrt for variance -> stddev, cossim^4 for filter
}

float4 CalcVelocityLayer(in VSOUT i, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr, uint BLOCK_SIZE)
{
	float curDepth = GetDepth(i.uv);
	
	float2 vel = GetMotionVector(FFXIV::get_world_position_from_uv(i.uv, curDepth), curDepth);
	
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));
	float BLOCK_SIZE_HALF = (BLOCK_SIZE * 0.5 - 0.5);//(int(BLOCK_SIZE / 2)) //2
	uint BLOCK_AREA = 		(BLOCK_SIZE * BLOCK_SIZE); //16
	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline
	
	float4 moments_local_s = 0;
	float4 moments_search_b = 0;
	float4 moments_search_s = 0;
	float4 moments_c = 0;

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float4 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr));
		float4 t_search = tex2Dlod(sFeaturePrev, float4(tuv + vel, 0, mip_gcurr));

		moments_local_s += t_local * t_local;
		moments_search_b += t_search;
		moments_search_s += t_search * t_search;
		moments_c += t_local * t_search;
	}
	
	moments_local_s /= BLOCK_AREA;
	moments_search_b /= BLOCK_AREA;
	moments_search_s /= BLOCK_AREA;
	moments_c /= BLOCK_AREA;

	float4 cossim = moments_c * rsqrt(moments_local_s * moments_search_s);
	float best_sim = saturate(min(min(min(cossim.x, cossim.y), cossim.z), cossim.w));

	float best_features = abs(moments_search_b.x * moments_search_b.x - moments_search_s.x);
	
	return float4(vel, best_features, best_sim * best_sim * best_sim * best_sim);
}

float4 atrous_upscale(VSOUT i, int mip_gcurr, sampler sMotionLow, sampler sFeatureCurr)
{	
    float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr + 1));	
	float rand = frac(mip_gcurr * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS;
	const float3 gauss = float3(1, 0.85, 0.65);

	float3 center_n = tex2Dlod(sFeatureCurr, float4(i.uv, 0, mip_gcurr)).yzw;

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;

	for(int x = -1; x <= 1; x++)
	for(int y = -1; y <= 1; y++)
	{
		float2 offs = mul(float2(x, y), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;

		float3 sample_n = tex2Dlod(sFeatureCurr, float4(sample_uv, 0, mip_gcurr + 1)).yzw;
		float4 sample_gbuf = tex2Dlod(sMotionLow, float4(sample_uv, 0, 0));

		float3 wn = abs(sample_n - center_n);
		float wns = (wn.x * wn.y * wn.z) * 128.0;
		float ws = saturate(1 - sample_gbuf.w * 4.0);
		float wf = saturate(1 - sample_gbuf.b * 4.0);
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4.0;

		float weight = exp2(-(wns + ws + wm + wf)) * gauss[abs(x)] * gauss[abs(y)];
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;		
	}

	return gbuffer_sum / wsum;	
}
#endif

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT VS_Main(in uint id : SV_VertexID)
{
    VSOUT o;
	PostProcessVS(id, o.vpos, o.uv);
    return o;
}

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
    o = float4(luma, normal);
}
#endif

float2 OctWrap( float2 v )
{
    //return ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );
    return float2((1.0 - abs( v.y ) ) * ( v.x >= 0.0 ? 1.0 : -1.0),
        (1.0 - abs( v.x ) ) * ( v.y >= 0.0 ? 1.0 : -1.0));
}

float2 Encode( float3 n )
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = n.z >= 0.0 ? n.xy : OctWrap( n.xy );
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

void PSWriteNormals(in VSOUT i, out float2 o : SV_Target0)
{
	float3 normals = FFXIV::get_normal(i.uv);
	normals.r = 1.0 - normals.r;
	o = Encode(normals - 0.5);
}

float4 motion_estimation(in VSOUT i, sampler sMotionLow, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr, uint BLOCK_SIZE)
{
	float4 upscaledLowerLayer = 0;

	[branch]
	if(mip_gcurr > UI_ME_LAYER_MIN) 
		return 0;

	[branch]
    if(mip_gcurr < UI_ME_LAYER_MIN)
	{
		upscaledLowerLayer = atrous_upscale(i, mip_gcurr, sMotionLow, sFeatureCurr);
	}
	
	[branch]
	if(mip_gcurr >= UI_ME_LAYER_MAX)
	{
		upscaledLowerLayer = CalcMotionLayer(i, mip_gcurr, upscaledLowerLayer.xy, sFeatureCurr, sFeaturePrev, BLOCK_SIZE);
		
		float4 vlayer = CalcVelocityLayer(i, sFeatureCurr, sFeaturePrev, mip_gcurr, BLOCK_SIZE);
		
		if(vlayer.w > upscaledLowerLayer.w)
			return vlayer;
	}
	
	return upscaledLowerLayer;
}

//void PSMotion7(in VSOUT i, out float4 o : SV_Target0){o = CalcVelocityLayer(i, sFeatureCurr, sFeaturePrev, 7, 1);}
void PSMotion6(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur7, sFeatureCurr, sFeaturePrev, 6, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion5(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur6, sFeatureCurr, sFeaturePrev, 5, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion4(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur5, sFeatureCurr, sFeaturePrev, 4, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion3(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur4, sFeatureCurr, sFeaturePrev, 3, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion2(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur3, sFeatureCurr, sFeaturePrev, 2, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion1(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur2, sFeatureCurr, sFeaturePrev, 1, PRE_BLOCK_SIZE_2_TO_7 * 1);}
void PSMotion0(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur1, sFeatureCurr, sFeaturePrev, 0, PRE_BLOCK_SIZE_2_TO_7 * 1);}


//Show motion vectors stuff
float3 HUEtoRGB(in float H)
{
	float R = abs(H * 6.f - 3.f) - 1.f;
	float G = 2 - abs(H * 6.f - 2.f);
	float B = 2 - abs(H * 6.f - 4.f);
	return saturate(float3(R,G,B));
}

float3 HSLtoRGB(in float3 HSL)
{
	float3 RGB = HUEtoRGB(HSL.x);
	float C = (1.f - abs(2.f * HSL.z - 1.f)) * HSL.y;
	return (RGB - 0.5f) * C + HSL.z;
}

float4 motionToLgbtq(float2 motion)
{
	float angle = degrees(atan2(motion.y, motion.x));
	float dist = length(motion);
	float3 rgb = HSLtoRGB(float3((angle / 360.f) + 0.5, saturate(dist * 100.0), 0.5));
	return float4(rgb.r, rgb.g, rgb.b, 0);
}

void PSOut(in VSOUT i, out float4 o : SV_Target0)
{
	if(!SHOWME) discard;
	o = motionToLgbtq(tex2D(Deferred::sMotionVectorsTex, i.uv).xy);
}
void PSWriteVectors(in VSOUT i, out float2 o : SV_Target0)
{
	o = tex2D(sMotionTexCur0, i.uv).xy;
}

/*=============================================================================
	Techniques
=============================================================================*/

technique FFXIV_Crashpad
{
    pass //update curr data RGB + depth
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteColorAndDepth; 
        RenderTarget = FFXIV_Crashpad::FeatureCurr; 

	}
    //mipmaps are being created :)
	//pass {VertexShader = VS_Main;PixelShader = PSMotion7;RenderTarget = FFXIV_Crashpad::MotionTexCur7;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit6;RenderTarget = MotionTexCur6;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion6;RenderTarget = FFXIV_Crashpad::MotionTexCur6;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit5;RenderTarget = MotionTexCur5;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion5;RenderTarget = FFXIV_Crashpad::MotionTexCur5;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit4;RenderTarget = MotionTexCur4;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion4;RenderTarget = FFXIV_Crashpad::MotionTexCur4;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit3;RenderTarget = MotionTexCur3;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion3;RenderTarget = FFXIV_Crashpad::MotionTexCur3;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit2;RenderTarget = MotionTexCur2;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion2;RenderTarget = FFXIV_Crashpad::MotionTexCur2;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit1;RenderTarget = MotionTexCur1;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion1;RenderTarget = FFXIV_Crashpad::MotionTexCur1;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit0;RenderTarget = MotionTexCur0;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion0;RenderTarget = FFXIV_Crashpad::MotionTexCur0;}

    pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteColorAndDepth; 
        RenderTarget0 = FFXIV_Crashpad::FeaturePrev; 
	}
	pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteVectors; 
		RenderTarget = Deferred::MotionVectorsTex;
	}
	pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteNormals; 
		RenderTarget = Deferred::NormalsTex;
	}

    pass 
	{
		VertexShader = VS_Main;
		PixelShader  = PSOut; 
	}     
}
