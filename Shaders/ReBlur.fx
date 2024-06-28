/** 
ReBlur
 - Based on "A Reconstruction Filter for Plausible Motion Blur" by McGuire et al.
 - Pseudo random numbers: https://gamedev.stackexchange.com/questions/32681/random-number-hlsl

Based on:
 - Reshade Linear Motion Blur 
 - First published 2022 - Copyright, Jakob Wapenhensch
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License
- https://creativecommons.org/licenses/by-nc/4.0/
- https://creativecommons.org/licenses/by-nc/4.0/legalcode
# Human-readable summary of the License and not a substitute for https://creativecommons.org/licenses/by-nc/4.0/legalcode:
You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
- The licensor cannot revoke these freedoms as long as you follow the license terms.
Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
Notices:
- You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
- No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.
 */


//  Includes
#include "ReShadeUI.fxh"
#include "ReShade.fxh"

#define UI_FLOAT2(_category, _name, _label, _descr, _min, _max, _default) \
    uniform float2 _name < \
        ui_category = _category; \
        ui_category_closed = true; \
        ui_label = _label; \
        ui_min = _min; \
        ui_max = _max; \
        ui_tooltip = _descr; \
        ui_step = 0.001; \
        ui_type = "slider"; \
    > = _default;

#ifndef BLUR_FILTER_TAPS
	#define BLUR_FILTER_TAPS 15
#endif

#ifndef BLUR_USE_IMMERSE_LAUNCHPAD
	#define BLUR_USE_IMMERSE_LAUNCHPAD 0
#endif

#undef BLUR_COMPUTE_SHADER 
#if __RENDERER__ >= 0xb000
	#define BLUR_COMPUTE_SHADER 1
#else
	#define BLUR_COMPUTE_SHADER 0
#endif

#define CEILING_DIV(X, Y) (((X) + (Y) - 1) / (Y))
#define CS_DISPATCH_GROUPS 32 // Biggly big groups for better cache coherence

uniform float frametime < source = "frametime"; >;

// UI
uniform float  UI_BLUR_LENGTH < __UNIFORM_SLIDER_FLOAT1
	ui_min = 0; ui_max = 1.0; ui_step = 0.01;
	ui_tooltip = "";
	ui_label = "Blur Length";
	ui_category = "Motion Blur";
> = 0.33;

uniform float  UI_BLUR_Z_EXTENSION < __UNIFORM_SLIDER_FLOAT1
	ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
	ui_tooltip = "";
	ui_label = "Z Extension";
	ui_category = "Motion Blur";
> = 0.1;


UI_FLOAT2("Motion Blur", UI_MB_DebugLen, "Debug Length", "To disable debug, set both sliders to 0", 0, 1, 0)

namespace ReBlur
{
	//  Textures & Samplers
	texture2D texColor : COLOR;
	sampler samplerColor { Texture = texColor; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Linear; };
	
	texture texDilatedMotion { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
	sampler samplerDilatedMotion { Texture = texDilatedMotion; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };
	
#if BLUR_COMPUTE_SHADER > 0
	texture texBuf { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
	sampler samplerBuf { Texture = texBuf; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };
	storage2D stBuf { Texture = ReBlur::texBuf; };
#endif
	
	texture texLinearDepth { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; };
	sampler samplerLinearDepth { Texture = texLinearDepth; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Linear; };
}

#if BLUR_USE_IMMERSE_LAUNCHPAD <= 0
texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler SamplerMotionVectors2 { Texture = texMotionVectors; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };
#else
namespace Deferred
{
	texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
}
sampler SamplerMotionVectors2 { Texture = Deferred::MotionVectorsTex; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };
#endif

//Helper
float cone(float2 x, float2 y, float2 v)
{
	return clamp(1.0 - length(x - y) / length(v), 0.0, 1.0);
}

float cylinder(float2 x, float2 y, float2 v)
{
	return 1.0 - smoothstep(0.95 * length(v), 1.05 * length(v), length(x - y));
}

float softDepthCompare(float za, float zb)
{
	return clamp(1.0 - (za - zb) / UI_BLUR_Z_EXTENSION, 0.0, 1.0);
}

float rand(in float2 uv)
{
	float2 noise = (frac(sin(dot(uv, float2(12.9898,78.233) * 2.0)) * 43758.5453));
	return abs(noise.x + noise.y) * 0.5;
}

// Passes
void PSDilate(float4 position : SV_Position, float2 texcoord : TEXCOORD, out float2 dilatedMotion : SV_Target0, out float depth : SV_Target1)
{
	dilatedMotion = (tex2D(SamplerMotionVectors2, texcoord).xy / frametime) * 32 * UI_BLUR_LENGTH;
    
    // for debugging
    if(dot(UI_MB_DebugLen, 1) > 0) dilatedMotion = float2(UI_MB_DebugLen);
    
	depth = ReShade::GetLinearizedDepth(texcoord);
}

#if BLUR_COMPUTE_SHADER <= 0
void PSReconstruct(float4 position : SV_Position, float2 texcoord : TEXCOORD, out float4 color : SV_Target0)
{
	float2 X = texcoord;
	int2 Xi = texcoord * BUFFER_SCREEN_SIZE;
	float4 C = tex2D(ReBlur::samplerColor, X);
	float2 V = tex2D(ReBlur::samplerDilatedMotion, X).rg;
	float Z = -tex2D(ReBlur::samplerLinearDepth, X).r;
	float2 Vi = V * BUFFER_SCREEN_SIZE;
	
	if(all(V == 0))
	{
		color = C;
		return;
	}
	
	float weight = 1.0 / length(Vi);
	float3 sum = C.rgb * weight;
	
	float j = rand(X) - 0.5;
	
	for(uint i = 0; i < BLUR_FILTER_TAPS; i++)
	{
		if(i == (BLUR_FILTER_TAPS - 1) / 2)
			continue;
			
		float t = lerp(-1.0, 1.0, float(i + j + 1) / (BLUR_FILTER_TAPS + 1.0));
		float2 Y = X + V * t;
		float2 Yi = Y * BUFFER_SCREEN_SIZE;
		
		float Zy = -tex2D(ReBlur::samplerLinearDepth, Y).r;
		
		float f = softDepthCompare(Z, Zy);
		float b = softDepthCompare(Zy, Z);
		
		float2 Vy = tex2D(ReBlur::samplerDilatedMotion, Y).rg * BUFFER_SCREEN_SIZE;
		float ay = 
			f * cone(Yi, Xi, Vy) + 
			b * cone(Xi, Yi, Vi) +
			cylinder(Yi, Xi, Vy) * cylinder(Xi, Yi, Vi) * 2.0;
			
		float3 Cy = tex2D(ReBlur::samplerColor, Y).rgb;
		weight += ay;
		sum += ay * Cy;
	}
	
	color = float4(sum / weight, C.a);
}
#else
void CSReconstruct(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
	int2 X = id.xy;
	float4 C = tex2Dfetch(ReBlur::samplerColor, X);
	float2 V = tex2Dfetch(ReBlur::samplerDilatedMotion, X).rg * BUFFER_SCREEN_SIZE;
	float Z = -tex2Dfetch(ReBlur::samplerLinearDepth, X).r;
	
	if(all(V == 0))
	{
		tex2Dstore(ReBlur::stBuf, X, C);
		return;
	}
	
	float weight = 1.0 / length(V);
	float3 sum = C.rgb * weight;
	
	float j = rand(X) - 0.5;
	
	for(uint i = 0; i < BLUR_FILTER_TAPS; i++)
	{
		if(i == (BLUR_FILTER_TAPS - 1) / 2)
			continue;
			
		float t = lerp(-1.0, 1.0, (float(i) + j + 1.0) / (BLUR_FILTER_TAPS + 1.0));
		int2 Y = float2(X) + V * t + 0.5;
		
		float Zy = -tex2Dfetch(ReBlur::samplerLinearDepth, Y).r;
		
		float f = softDepthCompare(Z, Zy);
		float b = softDepthCompare(Zy, Z);
		
		float2 Vy = tex2Dfetch(ReBlur::samplerDilatedMotion, Y).rg * BUFFER_SCREEN_SIZE;
		float ay = 
			f * cone(Y, X, Vy) + 
			b * cone(X, Y, V) +
			cylinder(Y, X, Vy) * cylinder(X, Y, V) * 2.0;
			
		float3 Cy = tex2Dfetch(ReBlur::samplerColor, Y).rgb;
		
		weight += ay;
		sum += ay * Cy;
	}
	
	tex2Dstore(ReBlur::stBuf, X, float4(sum / weight, C.a));
}

void PSDraw(float4 position : SV_Position, float2 texcoord : TEXCOORD, out float4 color : SV_Target0)
{
	color = tex2D(ReBlur::samplerBuf, texcoord);
}
#endif

technique ReBlur
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = PSDilate;
		RenderTarget0 = ReBlur::texDilatedMotion;
		RenderTarget1 = ReBlur::texLinearDepth;
	}
	
#if BLUR_COMPUTE_SHADER <= 0
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = PSReconstruct;
	}
#else
	pass
	{
		ComputeShader = CSReconstruct<CS_DISPATCH_GROUPS, CS_DISPATCH_GROUPS>;
		DispatchSizeX = CEILING_DIV(BUFFER_WIDTH, CS_DISPATCH_GROUPS);
		DispatchSizeY = CEILING_DIV(BUFFER_HEIGHT, CS_DISPATCH_GROUPS);
	}
	
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = PSDraw;
	}
#endif
}