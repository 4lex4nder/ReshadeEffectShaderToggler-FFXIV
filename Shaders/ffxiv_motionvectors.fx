/*=============================================================================
This work is licensed under the 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License
https://creativecommons.org/licenses/by-nc/4.0/	  

Original developer: Jak0bPCoder
Optimization by : Marty McFly
Compatibility by : MJ_Ehsan
Messing around with stuff he barely understands: alex
 Added:
 -Use velocity buffer as an initial estimate, use world space normals for matching
 -Velocity buffer: http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html
 -Disocclusion checks: https://diharaw.github.io/post/adventures_in_hybrid_rendering/
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
> = 3;

uniform float UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;	
> = 4.0;

uniform bool VelocityEstimation <
	ui_label = "Use velocity as initial guess";
> = true;

uniform bool FireflyChecks <
	ui_label = "Perform disocclusion checks";
> = false;

uniform float NormalBias <
	ui_type = "drag";
	ui_label = "Normal disocclusion limit";
	ui_step = 0.01;
	ui_min = 0.0;
	ui_max = 1.0;	
> = 0.8;

uniform float PlaneBias <
	ui_type = "drag";
	ui_label = "Plane disocclusion limit";
	ui_step = 0.01;
	ui_min = 0.0;
	ui_max = 1.0;	
> = 0.5;

uniform float PlaneDistanceLimit <
	ui_type = "drag";
	ui_label = "Limit to perform plane disocclusion check on";
	ui_step = 1.0;
	ui_min = 0.0;
	ui_max = 1000.0;	
> = 100.0;

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

#define BLOCK_SIZE (PRE_BLOCK_SIZE_2_TO_7) //4

//NEVER change these!!!
#define BLOCK_SIZE_HALF (BLOCK_SIZE * 0.5 - 0.5)//(int(BLOCK_SIZE / 2)) //2
#define BLOCK_AREA 		(BLOCK_SIZE * BLOCK_SIZE) //16

//smpG samplers are .r = Grayscale, g = depth
//smgM samplers are .r = motion x, .g = motion y, .b = feature level, .a = loss;

texture FFFeatureCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 8; };
sampler sFeatureCurr         { Texture = FFFeatureCurr;  };
texture DepthCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F; MipLevels = 8; };
sampler sDepthCurr         { Texture = DepthCurr;   };
texture FFFeaturePrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 8; };
sampler sFeaturePrev         { Texture = FFFeaturePrev;   };
texture FFPosPrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
sampler sFFPosPrev         { Texture = FFPosPrev;   };

texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex         { Texture = texMotionVectors;  };

texture MotionTexCur7               { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = RGBA16F;  };
sampler sMotionTexCur7              { Texture = MotionTexCur7;};
texture MotionTexCur6               { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RGBA16F;  };
sampler sMotionTexCur6              { Texture = MotionTexCur6;};
texture MotionTexCur5               { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RGBA16F;  };
sampler sMotionTexCur5              { Texture = MotionTexCur5;};
texture MotionTexCur4               { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RGBA16F;  };
sampler sMotionTexCur4              { Texture = MotionTexCur4;};
texture MotionTexCur3               { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F;  };
sampler sMotionTexCur3              { Texture = MotionTexCur3;};
texture MotionTexCur2               { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;  };
sampler sMotionTexCur2              { Texture = MotionTexCur2;};
texture MotionTexCur1               { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;  };
sampler sMotionTexCur1              { Texture = MotionTexCur1;};
texture MotionTexCur0               { Width = BUFFER_WIDTH >> 0;   Height = BUFFER_HEIGHT >> 0;   Format = RGBA16F;  };
sampler sMotionTexCur0              { Texture = MotionTexCur0;};

texture2D texVelocityBuffer { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler sVelocityBuffer { Texture = texVelocityBuffer; };

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

float LDepth(float2 texcoord)
{
	return ReShade::GetLinearizedDepth(texcoord);
}

float GetDepth(float2 texcoords)
{
	return tex2Dlod(ReShade::DepthBuffer, float4(texcoords, 0, 0)).x;
}

bool normals_disocclusion_check(float3 current_normal, float3 history_normal)
{
    return (pow(abs(dot(current_normal, history_normal)), 2) > NormalBias);
}

bool plane_distance_disocclusion_check(float3 current_pos, float3 history_pos, float3 current_normal)
{
    float3 to_current = (current_pos) - (history_pos);
    
    // Limit plane occlusion check distance to void inaccuracies at z_far
    if(length(abs(current_pos - FFXIV::camPos)) > PlaneDistanceLimit && length(abs(history_pos - FFXIV::camPos)) > PlaneDistanceLimit)
        return true;
        
    float dist_to_plane = abs(dot(to_current, current_normal));

    return dist_to_plane < 1.0 - PlaneBias;
}

bool disocclusion_check(float2 texcoord, float2 motion)
{
	float3 hist_normal = tex2D(sFeaturePrev, texcoord + motion).rgb;
	float3 hist_pos = tex2D(sFFPosPrev, texcoord + motion).rgb;
	float3 cur_normal = tex2D(sFeatureCurr, texcoord).rgb;
	float3 cur_pos = FFXIV::get_world_position_from_uv(texcoord, GetDepth(texcoord));
	
	return (normals_disocclusion_check(cur_normal, hist_normal) && plane_distance_disocclusion_check(cur_pos, hist_pos, cur_normal));
}

float2 GetMotionVector(float3 position, float d)
{
	float4 previousPositionUV = mul(matPrevViewProj, float4(position, 1));
	float4 currentPositionUV = mul(FFXIV::matViewProj, float4(position, 1));
	
	previousPositionUV *= rcp(previousPositionUV.w);
	currentPositionUV *= rcp(currentPositionUV.w);
	
	return (previousPositionUV.xy - currentPositionUV.xy) * float2(0.5, -0.5);
}

float2 pixel_idx_to_uv(uint2 pos, float2 texture_size)
{
    float2 inv_texture_size = rcp(texture_size);
    return pos * inv_texture_size + 0.5 * inv_texture_size;
}

float noise(float2 co)
{
  return frac(sin(dot(co.xy ,float2(1.0,73))) * 437580.5453);
}

float4 CalcMotionLayer(VSOUT i, int mip_gcurr, float2 searchStart, sampler sFeatureCurr, sampler sFeaturePrev)
{	
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr));

	float3 moments_local = 0;
	float3 moments_search = 0;
	float2 moments_luma = 0;

	i.uv -= texelsize * BLOCK_SIZE_HALF; //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 tuv = i.uv + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize;
		float4 t_local = tex2Dlod(sFeatureCurr, float4(tuv, 0, mip_gcurr));
		float4 t_search = tex2Dlod(sFeaturePrev, float4(tuv + searchStart, 0, mip_gcurr));

		moments_local += t_local.rgb;
		moments_search += t_search.rgb;
		
		moments_luma += float2(t_search.w, t_search.w * t_search.w);
	}

	moments_local = normalize(moments_local / BLOCK_AREA);
	moments_search /= BLOCK_AREA;
	moments_luma /= BLOCK_AREA;

	float best_sim = dot(moments_local, normalize(moments_search));
	float best_features = abs(moments_luma.x * moments_luma.x - moments_luma.y);

	float2 bestMotion = 0;
	float2 searchCenter = searchStart;    

	float randseed = noise(i.uv);
	randseed = frac(randseed + (FRAME_COUNT % 16) * UI_ME_SAMPLES_PER_ITERATION * UI_ME_MAX_ITERATIONS_PER_LEVEL * 0.6180339887498);
	float2 randdir; sincos(randseed * 6.283, randdir.x, randdir.y); //yo dawg, I heard you like golden ratios
	float2 scale = texelsize;
	
	const float exit = 0.999;

	[loop]
	for(int j = 0; j < UI_ME_MAX_ITERATIONS_PER_LEVEL && best_sim < exit; j++)
	{
		[loop]
		for (int s = 1; s < UI_ME_SAMPLES_PER_ITERATION && best_sim < exit; s++) 
		{
			randdir = mul(randdir, float2x2(-0.7373688, 0.6754903, -0.6754903, -0.7373688));//rotate by larger golden angle			
			float2 pixelOffset = randdir * scale;
			float2 samplePos = i.uv + searchCenter + pixelOffset;			 

			float3 moments_candidate = 0;	
			float2 moments_candidate_luma = 0;

			[loop]
			for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
			{
				float4 t = tex2Dlod(sFeaturePrev, float4(samplePos + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize, 0, mip_gcurr));
				moments_candidate += t.rgb;
				moments_candidate_luma += float2(t.w, t.w * t.w);
			}
			moments_candidate /= BLOCK_AREA;
			moments_candidate_luma /= BLOCK_AREA;

			float candidate_similarity = dot(moments_local, normalize(moments_candidate));

			[flatten]
			if(candidate_similarity > best_sim + 0.001)					
			{
				best_sim = candidate_similarity;
				bestMotion = pixelOffset;
				best_features = abs(moments_candidate_luma.x * moments_candidate_luma.x - moments_candidate_luma.y);
			}			
		}
		searchCenter += bestMotion;
		bestMotion = 0;
		scale *= 0.5;
	}

	return float4(searchCenter, sqrt(best_features), best_sim * best_sim * best_sim * best_sim);  //delayed sqrt for variance -> stddev, cossim^4 for filter
}

float4 atrous_upscale(VSOUT i, int mip_gcurr, sampler sMotionLow, sampler sFeatureCurr)
{	
    float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(mip_gcurr + 1));	
	float rand = frac(mip_gcurr * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * UI_ME_PYRAMID_UPSCALE_FILTER_RADIUS;
	const float3 gauss = float3(1, 0.85, 0.65);

	float center_z = tex2Dlod(sDepthCurr, float4(i.uv, 0, mip_gcurr)).r;	

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;

	for(int x = -1; x <= 1; x++)
	for(int y = -1; y <= 1; y++)
	{
		float2 offs = mul(float2(x, y), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;

		float sample_z = tex2Dlod(sDepthCurr, float4(sample_uv, 0, mip_gcurr + 1)).r;
		float4 sample_gbuf = tex2Dlod(sMotionLow, float4(sample_uv, 0, 0));

		float wz = abs(sample_z - center_z) * 4;
		float ws = saturate(1 - sample_gbuf.w * 4);
		float wf = saturate(1 - sample_gbuf.z * 128.0);
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;

		float weight = exp2(-(wz + ws + wm + wf)) * gauss[abs(x)] * gauss[abs(y)];
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;		
	}

	return gbuffer_sum / wsum;	
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT VS_Main(in uint id : SV_VertexID)
{
    VSOUT o;
	PostProcessVS(id, o.vpos, o.uv);
    return o;
}

void PSWriteNormalsAndDepthCurr(in VSOUT i, out float4 o : SV_Target0, out float d : SV_Target1)
{    
	float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
	float3 onormal = FFXIV::get_normal(i.uv);
	float depth = LDepth(i.uv);
	float luma = dot(tex2D(ReShade::BackBuffer, i.uv).rgb, float3(0.299, 0.587, 0.114));
	o = float4(mul(matV, normalize(onormal - 0.5)), luma);
	
	d = depth;
}

void PSWriteNormalsAndPosPrev(in VSOUT i, out float4 o : SV_Target0, out float4 pos : SV_Target1)
{
	o = tex2D(sFeatureCurr, i.uv);
	pos = float4(FFXIV::get_world_position_from_uv(i.uv, GetDepth(i.uv)), 1);
}

float4 motion_estimation(in VSOUT i, sampler sMotionLow, sampler sFeatureCurr, sampler sFeaturePrev, int mip_gcurr)
{
	float4 upscaledLowerLayer = 0;

	[branch]
	if(mip_gcurr > UI_ME_LAYER_MIN) 
		return 0;
		
	[branch]
    if(mip_gcurr == UI_ME_LAYER_MIN && VelocityEstimation)
	{
		float curDepth = GetDepth(i.uv);
		upscaledLowerLayer = float4(GetMotionVector(FFXIV::get_world_position_from_uv(i.uv, curDepth), curDepth), 0, 0);
	}

	[branch]
    if(mip_gcurr < UI_ME_LAYER_MIN)
	{
		upscaledLowerLayer = atrous_upscale(i, mip_gcurr, sMotionLow, sFeatureCurr);
	}
	
	[branch]
	if(mip_gcurr >= UI_ME_LAYER_MAX)
	{
		upscaledLowerLayer = CalcMotionLayer(i, mip_gcurr, upscaledLowerLayer.xy, sFeatureCurr, sFeaturePrev);
	}
	
	[branch]
	if(mip_gcurr == UI_ME_LAYER_MAX && FireflyChecks && !disocclusion_check(i.uv, upscaledLowerLayer.xy))
	{
		return 0;
	}
	
	return upscaledLowerLayer;
}

void PSMotion6(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur7, sFeatureCurr, sFeaturePrev, 6);}
void PSMotion5(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur6, sFeatureCurr, sFeaturePrev, 5);}
void PSMotion4(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur5, sFeatureCurr, sFeaturePrev, 4);}
void PSMotion3(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur4, sFeatureCurr, sFeaturePrev, 3);}
void PSMotion2(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur3, sFeatureCurr, sFeaturePrev, 2);}
void PSMotion1(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur2, sFeatureCurr, sFeaturePrev, 1);}
void PSMotion0(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur1, sFeatureCurr, sFeaturePrev, 0);}


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
	o = motionToLgbtq(tex2D(sMotionVectorTex, i.uv).xy);
}
void PSWriteVectors(in VSOUT i, out float2 o : SV_Target0)
{
	o = tex2D(sMotionTexCur0, i.uv).xy;
}



/*=============================================================================
	Techniques
=============================================================================*/

technique FFXIV_MotionVectors
{
    pass //update curr data RGB + depth
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteNormalsAndDepthCurr; 
        RenderTarget0 = FFFeatureCurr; 
		RenderTarget1 = DepthCurr;

	}
    //mipmaps are being created :)
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit6;RenderTarget = MotionTexCur6;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion6;RenderTarget = MotionTexCur6;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit5;RenderTarget = MotionTexCur5;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion5;RenderTarget = MotionTexCur5;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit4;RenderTarget = MotionTexCur4;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion4;RenderTarget = MotionTexCur4;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit3;RenderTarget = MotionTexCur3;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion3;RenderTarget = MotionTexCur3;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit2;RenderTarget = MotionTexCur2;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion2;RenderTarget = MotionTexCur2;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit1;RenderTarget = MotionTexCur1;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion1;RenderTarget = MotionTexCur1;}
	//pass {VertexShader = VS_Main;PixelShader = PSMotionInit0;RenderTarget = MotionTexCur0;}
    pass {VertexShader = VS_Main;PixelShader = PSMotion0;RenderTarget = MotionTexCur0;}

    pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteNormalsAndPosPrev; 
        RenderTarget0 = FFFeaturePrev; 
		RenderTarget1 = FFPosPrev;
	}
	pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteVectors; 
		RenderTarget = texMotionVectors;
	}

    pass 
	{
		VertexShader = VS_Main;
		PixelShader  = PSOut; 
	}     
}
