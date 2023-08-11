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

#include "ffxiv_motionvectors_shared.fxh"

#ifndef OUTPUT_GENERIC_MV_COPY
#define OUTPUT_GENERIC_MV_COPY 0 // [0, 1] Will write a copy in the old format as well, for usage with SSR,TFAA, etc.,...
#endif

namespace FFXIV_Crashpad {

#if WS_NORMAL_FEATURES == 0
texture FeatureCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; MipLevels = NUM_MIPS; };
texture FeaturePrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; MipLevels = NUM_MIPS; };
#else
texture FeatureCurr          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = NUM_MIPS; };
texture FeaturePrev          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = NUM_MIPS; };
#endif

#if NUM_MIPS >= 10
texture MotionTexCur9               { Width = BUFFER_WIDTH >> 9;   Height = BUFFER_HEIGHT >> 9;   Format = RGBA16F; };
#endif
#if NUM_MIPS >= 9
texture MotionTexCur8               { Width = BUFFER_WIDTH >> 8;   Height = BUFFER_HEIGHT >> 8;   Format = RGBA16F; };
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

#if OUTPUT_GENERIC_MV_COPY > 0
texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex         { Texture = texMotionVectors;  };
#endif

#if NUM_MIPS >= 10
sampler sMotionTexCur9              { Texture = FFXIV_Crashpad::MotionTexCur9; };
#endif
#if NUM_MIPS >= 9
sampler sMotionTexCur8              { Texture = FFXIV_Crashpad::MotionTexCur8; };
#endif
sampler sMotionTexCur7              { Texture = FFXIV_Crashpad::MotionTexCur7; };
sampler sMotionTexCur6              { Texture = FFXIV_Crashpad::MotionTexCur6; };
sampler sMotionTexCur5              { Texture = FFXIV_Crashpad::MotionTexCur5; };
sampler sMotionTexCur4              { Texture = FFXIV_Crashpad::MotionTexCur4; };
sampler sMotionTexCur3              { Texture = FFXIV_Crashpad::MotionTexCur3; };
sampler sMotionTexCur2              { Texture = FFXIV_Crashpad::MotionTexCur2; };
sampler sMotionTexCur1              { Texture = FFXIV_Crashpad::MotionTexCur1; };
sampler sMotionTexCur0              { Texture = FFXIV_Crashpad::MotionTexCur0; };

/*=============================================================================
	Functions
=============================================================================*/

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT VS_Main(in uint id : SV_VertexID)
{
    VSOUT o;
	PostProcessVS(id, o.vpos, o.uv);
    return o;
}

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
	o = Encode((normals - 0.5));
}

//void PSMotion7(in VSOUT i, out float4 o : SV_Target0){o = CalcVelocityLayer(i, sFeatureCurr, sFeaturePrev, 7, 1);}
#if NUM_MIPS == 10
void PSMotion9(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation_init(i, sFeatureCurr, sFeaturePrev, 9);}
void PSMotion8(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur9, sFeatureCurr, sFeaturePrev, 8);}
void PSMotion7(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur8, sFeatureCurr, sFeaturePrev, 7);}
#elif NUM_MIPS == 9
void PSMotion8(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation_init(i, sFeatureCurr, sFeaturePrev, 8);}
void PSMotion7(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation(i, sMotionTexCur8, sFeatureCurr, sFeaturePrev, 7);}
#else
void PSMotion7(in VSOUT i, out float4 o : SV_Target0){o = motion_estimation_init(i, sFeatureCurr, sFeaturePrev, 7);}
#endif
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
	o = motionToLgbtq(tex2D(Deferred::sMotionVectorsTex, i.uv).xy);
}

#if OUTPUT_GENERIC_MV_COPY > 0
void PSWriteVectors(in VSOUT i, out float2 o : SV_Target0, out float2 p : SV_Target1)
{
	o = tex2D(sMotionTexCur0, i.uv).xy;
	p = o;
}
#else
void PSWriteVectors(in VSOUT i, out float2 o : SV_Target0)
{
	o = tex2D(sMotionTexCur0, i.uv).xy;
}
#endif

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
	#if NUM_MIPS >= 10
	pass {VertexShader = VS_Main;PixelShader = PSMotion9;RenderTarget = FFXIV_Crashpad::MotionTexCur9;}
	#endif
	#if NUM_MIPS >= 9
	pass {VertexShader = VS_Main;PixelShader = PSMotion8;RenderTarget = FFXIV_Crashpad::MotionTexCur8;}
	#endif
	pass {VertexShader = VS_Main;PixelShader = PSMotion7;RenderTarget = FFXIV_Crashpad::MotionTexCur7;}
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
	#if OUTPUT_GENERIC_MV_COPY > 0
	pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteVectors; 
		RenderTarget0 = Deferred::MotionVectorsTex;
		RenderTarget1 = texMotionVectors;
	}
	#else
	pass  
	{
		VertexShader = VS_Main;
		PixelShader  = PSWriteVectors; 
		RenderTarget = Deferred::MotionVectorsTex;
	}
	#endif
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
