/*=============================================================================
This work is licensed under the 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License
https://creativecommons.org/licenses/by-nc/4.0/      

Original developer: Jak0bPCoder
Optimization by : Marty McFly
Compatibility by : MJ_Ehsan

alex: stripped out most things, lol
=============================================================================*/

/*=============================================================================
    Preprocessor settings
=============================================================================*/
#include "ReShade.fxh"
#include "ffxiv_common.fxh"
/*=============================================================================
    UI Uniforms
=============================================================================*/

uniform bool SHOWME <
    ui_label = "Debug Output";    
> = false;

/*=============================================================================
    Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

namespace Deferred {
    texture MotionVectorsTex { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
    sampler sMotionVectorsTex { Texture = MotionVectorsTex;  };
    
    texture NormalsTexV2 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16; };
    sampler sNormalsTex { Texture = NormalsTexV2; };
}

texture texMotionVectors { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex { Texture = texMotionVectors;  };

/*=============================================================================
    Functions
=============================================================================*/

/*=============================================================================
    Shader Entry Points
=============================================================================*/
struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

VSOUT VS_Main(in uint id : SV_VertexID)
{
    VSOUT o;
    PostProcessVS(id, o.vpos, o.uv);
    return o;
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
    float angle = degrees(atan2(motion.y, motion.x));
    float dist = length(motion);
    float3 rgb = HSLtoRGB(float3((angle / 360.f) + 0.5, saturate(dist * 100.0), 0.5));
    return float4(rgb.r, rgb.g, rgb.b, 0);
}

void PSOut(in VSOUT i, out float4 o : SV_Target0)
{
    if(!SHOWME) discard;
    float2 vec = tex2D(Deferred::sMotionVectorsTex, i.uv).rg;
    o = float4(motionToLgbtq(vec).rgb, 1);
}

void PSWriteVectors(in VSOUT i, out float2 o : SV_Target0, out float2 p : SV_Target1, out float4 q : SV_Target2)
{
    o = FFXIV::get_motion(i.uv).rg;
    p = o;
    
    // normals need to be reoriented
    float3 blah = FFXIV::get_normal(i.uv);
    blah.r = 1.0 - blah.r;
    float2 n = FFXIV::_encode(blah - 0.5);
    q = n.rgrg;
}

/*=============================================================================
    Techniques
=============================================================================*/

technique FFXIV_Crashpad
{
    pass  
    {
        VertexShader = VS_Main;
        PixelShader  = PSWriteVectors; 
        RenderTarget0 = texMotionVectors;
        RenderTarget1 = Deferred::MotionVectorsTex;
        RenderTarget2 = Deferred::NormalsTexV2;
    }

    pass 
    {
        VertexShader = VS_Main;
        PixelShader  = PSOut; 
    }     
}
