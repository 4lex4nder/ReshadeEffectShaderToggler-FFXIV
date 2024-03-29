/*
    Simple velocity buffer
    Source: http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html

    Additional sources:
	- Motion visualization: DRME, Jakob Wapenhensch (https://github.com/JakobPCoder/ReshadeMotionEstimation, CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/)
	- Disocclusion checks: https://diharaw.github.io/post/adventures_in_hybrid_rendering/
*/

#include "Reshade.fxh"
#include "ffxiv_common.fxh"

uniform float4x4 matPrevViewProj < source = "mat_PrevViewProj"; >;

texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler SamplerMotionVectors { Texture = texMotionVectors; };

texture2D FFXIV_VB_PrevWSNormals { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler sFFXIV_VB_PrevWSNormals { Texture = FFXIV_VB_PrevWSNormals; };

texture2D FFXIV_VB_PrevPosition { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler sFFXIV_VB_PrevPosition { Texture = FFXIV_VB_PrevPosition; };

uniform int DebugMode <
	ui_type = "combo";
	ui_items = "None\0Motion\0";
	ui_min = 0;
	ui_max = 1;	
> = 0;

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

float GetDepth(float2 texcoords)
{
	return tex2Dlod(ReShade::DepthBuffer, float4(texcoords, 0, 0)).r;
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

    return dist_to_plane < PlaneBias;
}

float2 GetMotionVector(float3 position)
{
	float4 previousPositionUV = mul(matPrevViewProj, float4(position, 1));
	float4 currentPositionUV = mul(FFXIV::matViewProj, float4(position, 1));
	
	previousPositionUV *= rcp(previousPositionUV.w);
	currentPositionUV *= rcp(currentPositionUV.w);
	
	return (previousPositionUV.xy - currentPositionUV.xy) * float2(0.5, -0.5);
}

void PS_WritePrevFrameData(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 normal: SV_Target0, out float4 pos: SV_Target1)
{
	normal = tex2Dlod(FFXIV::NormalMap, float4(texcoord, 0, 0));
	float4 decal = tex2Dlod(FFXIV::DecalNormalMap, float4(texcoord, 0, 0));
	float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).r;
	
	normal.xyz = lerp(normal.xyz, float3(0.5, 0.5, 1), normal.w == 1);
	normal.xyz = lerp(normal.xyz, decal.xyz, decal.a > FFXIV::MAGIC_ALPHA);

	float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
	
	normal = float4(mul(matV, normalize(normal.xyz - 0.5)), 1);
	pos = FFXIV::get_world_position_from_uv(texcoord, depth);
}

void PS_CalcMotion(in float4 position : SV_Position, in float2 texcoord : TEXCOORD,  out float2 motion : SV_Target0, out float4 normals: SV_Target1)
{
	float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
	float3 cur_pos = FFXIV::get_world_position_from_uv(texcoord, GetDepth(texcoord));
	
	motion = GetMotionVector(cur_pos);
	
	float3 hist_wsnormal = tex2D(sFFXIV_VB_PrevWSNormals, texcoord + motion).rgb;
	float3 hist_pos = tex2D(sFFXIV_VB_PrevPosition, texcoord + motion).rgb;
	float3 cur_normal = FFXIV::get_normal(texcoord);
	float3 cur_wsnormal = float4(mul(matV, normalize(cur_normal - 0.5)), 1);
	
	[branch]
	if(!normals_disocclusion_check(cur_wsnormal, hist_wsnormal) || !plane_distance_disocclusion_check(cur_pos, hist_pos, cur_wsnormal))
	{
		motion = 0;
	}
}

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
	float angle = atan2(motion.y, motion.x) * 57.2957795131;
	float dist = length(motion);
	float3 rgb = HSLtoRGB(float3((angle / 360.f) + 0.5, saturate(dist * 50), 0.5));

	return float4(rgb.r, rgb.g, rgb.b, 0);
}

void PS_Out(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	output = 0;
	if(DebugMode > 0)
	{
		output = motionToLgbtq(tex2D(SamplerMotionVectors, texcoord).xy);
		return;
	}
	
	discard;
}

technique FFXIV_VELOCITYBUFFER
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_CalcMotion;
		RenderTarget0 = texMotionVectors;
	}
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_WritePrevFrameData;
		RenderTarget0 = FFXIV_VB_PrevWSNormals;
		RenderTarget1 = FFXIV_VB_PrevPosition;
	}
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_Out;
	}
}