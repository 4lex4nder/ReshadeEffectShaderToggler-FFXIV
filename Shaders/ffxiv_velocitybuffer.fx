/*
	Simple velocity buffer
    Source: http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html

    Additional sources:
	- Motion visualization: DRME, Jakob Wapenhensch (https://github.com/JakobPCoder/ReshadeMotionEstimation, CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/)
*/

#include "Reshade.fxh"
#include "ffxiv_common.fxh"

uniform float4x4 matPrevViewProj < source = "mat_PrevViewProj"; >;

texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler SamplerMotionVectors { Texture = texMotionVectors; };

texture2D FFXIV_VB_PrevFrameDepth<pooled = true;> { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R32F; };
sampler sFFXIV_VB_PrevFrameDepth { Texture = FFXIV_VB_PrevFrameDepth; };

uniform int DebugMode <
	ui_type = "combo";
	ui_items = "None\0Motion\0Depth Difference\0";
	ui_min = 0;
	ui_max = 2;	
> = 0;

uniform float DepthBias <
	ui_type = "drag";
	ui_label = "Depth displacement bias";
	ui_step = 1.0;
	ui_min = 0.0;
	ui_max = 1000.0;	
> = 100.0;

float GetDepth(float2 texcoords)
{
	return tex2Dlod(ReShade::DepthBuffer, float4(texcoords, 0, 0)).r;
}

float2 GetMotionVector(float3 position, float d)
{
	float4 previousPositionUV = mul(matPrevViewProj, float4(position, 1));
	float4 currentPositionUV = mul(FFXIV::matViewProj, float4(position, 1));
	
	previousPositionUV *= rcp(previousPositionUV.w);
	currentPositionUV *= rcp(currentPositionUV.w);
	
	return (previousPositionUV.xy - currentPositionUV.xy) * float2(0.5, -0.5);
}

void PS_WritePrevDepth(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float depth: SV_Target0)
{
	depth = tex2D(ReShade::DepthBuffer, texcoord).r;
}

void PS_CalcMotion(in float4 position : SV_Position, in float2 texcoord : TEXCOORD,  out float2 motion : SV_Target0)
{
	float curDepth = GetDepth(texcoord);
	
	motion = GetMotionVector(FFXIV::get_world_position_from_uv(texcoord, curDepth), curDepth);
		
	float prevDepth = tex2D(sFFXIV_VB_PrevFrameDepth, texcoord + motion).r;
	float diff = abs(curDepth - prevDepth);
	
	if(diff > (DepthBias / 100000))
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
		if(DebugMode == 1)
			output = motionToLgbtq(tex2D(SamplerMotionVectors, texcoord).xy);
		else
		{
			float2 mo = tex2D(SamplerMotionVectors, texcoord).xy;
			float curDepth = tex2D(ReShade::DepthBuffer, texcoord).r;
			float prevDepth = tex2D(sFFXIV_VB_PrevFrameDepth, texcoord + mo).r;
			float diff = abs(curDepth - prevDepth);
			
			output = lerp(float4(0, 0, 0, 1), float4(1, 0, 0, 1), diff * 1000 * (diff >= (DepthBias / 100000)));
		}
		
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
		PixelShader  = PS_WritePrevDepth;
		RenderTarget0 = FFXIV_VB_PrevFrameDepth;
	}
	
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_Out;
	}
}