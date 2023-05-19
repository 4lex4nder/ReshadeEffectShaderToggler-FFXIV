/*
    Original article: GPU Pro 5, Hi-Z Screen-Space Cone-Traced Reflections by Yasin Uludag
    Based on implementation by Sugu Lee (https://sugulee.wordpress.com/2021/01/19/screen-space-reflections-implementation-and-optimization-part-2-hi-z-tracing-method/)
    
    Additional sources:
    - Information on Hi-Z SSR: https://www.gamedev.net/forums/topic/658702-help-with-gpu-pro-5-hi-z-screen-space-reflections/?page=1
    - UV Map: virtex.edge (https://virtexedge.design/shader-series-basic-screen-space-reflections/)
    - Normal reconstruction: bgolus (https://gist.github.com/bgolus/a07ed65602c009d5e2f753826e8078a0)
*/

#include "Reshade.fxh"
#include "ffxiv_common.fxh"
#include "ffxiv_hiz.fxh"

#ifndef FFXIV_REPROJECT_PREVIOUS_FRAME
 #define FFXIV_REPROJECT_PREVIOUS_FRAME 0
#endif

#define CEILING_DIV(X, Y) (((X)/(Y)) + ((X) % (Y) != 0))
#define CS_DISPATCH_GROUPS 8

texture ReflectionMaskTex : REFLECTIONS;
sampler ReflectionMaskInput { Texture = ReflectionMaskTex; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

texture FFXIV_SSR_Color : COLOR;
sampler sFFXIV_SSR_Color { Texture = FFXIV_SSR_Color; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

texture FFXIV_SSR_Depth : DEPTH;
sampler sFFXIV_SSR_Depth_Point { Texture = FFXIV_SSR_Depth; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

texture2D UVMap     { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler2D sUVMap    { Texture = UVMap; };
storage2D stUVMap   { Texture = UVMap; };

texture2D FFXIV_WaterDepth { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler2D sFFXIV_WaterDepth    { Texture = FFXIV_WaterDepth; };

texture2D FFXIV_Reflectivity { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler2D sFFXIV_Reflectivity    { Texture = FFXIV_Reflectivity; };

#if FFXIV_REPROJECT_PREVIOUS_FRAME > 0
texture2D FFXIV_PrevFrame { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler2D sFFXIV_PrevFrame { Texture = FFXIV_PrevFrame; };

texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler SamplerMotionVectors { Texture = texMotionVectors; };
#endif

//----------------

uniform float EDGE_CUT_OFF <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Edge cut off";
    ui_tooltip = "Edge fading factor";
> = 1;

uniform float INTENSITY <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Intensity";
    ui_tooltip = "Reflectivity multiplier";
> = 0.5;

uniform float BUMPINESS_GENERAL <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "General Bumpiness";
    ui_tooltip = "Blend factor between vertex and texture normals for everything that has no transparency (read: not water)";
> = 0.15;

uniform float BUMPINESS_DECAL <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Transparency Bumpiness";
    ui_tooltip = "Blend factor between vertex and texture normals for everything that has transparency (read: water)";
> = 0.05;

uniform uint MAX_ITERATION <
    ui_type = "drag";
    ui_min = 1; ui_max = 1000;
    ui_step = 1;
    ui_label = "Maximum number of search steps";
    ui_tooltip = "";
> = 300;

uniform float DEPTH_BIAS <
    ui_type = "drag";
    ui_min = 1; ui_max = 100;
    ui_step = 1;
    ui_label = "Maximum cut off depth thickness";
    ui_tooltip = "";
> = 10;

uniform float WATER_DEPTH_THRESHOLD <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Water Depth Threshold";
    ui_tooltip = "Only apply after a certain depth difference (broken shorelines!)";
    ui_category = "Water";
> = 0.01;

uniform float WATER_DEPTH_FACTOR <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 5.0;
    ui_step = 0.1;
    ui_label = "Water Depth Factor";
    ui_tooltip = "Water depth difference multiplier";
    ui_category = "Water";
> = 2.5;

//----------------

float GetDepth(float2 texcoords)
{
    return tex2Dlod(ReShade::DepthBuffer, float4(texcoords, 0, 0)).x;
}

float2 getMinimumDepthPlane(uint2 p, int mipLevel)
{
    return mipLevel > 0 ? tex2Dfetch(sFFXIV_Hi_Z, p, mipLevel - 1).xy : tex2Dfetch(sFFXIV_SSR_Depth_Point, p, 0).rr;
}

float3 NormalViewToWorld(float2 texcoord, float3 normal)
{
    float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
    return mul(matV, normal);
}

float3 ViewNormalAtPixelPosition(float2 uv)
{
    float3x2 taps = float3x2(uv + float2(0.0, 0.0) * ReShade::PixelSize, uv + float2(1.0, 0.0) * ReShade::PixelSize, uv + float2(0.0, 1.0) * ReShade::PixelSize);
    // get current pixel's view space position
    float3 viewSpacePos_c = FFXIV::get_position_from_uv(taps[0], GetDepth(taps[0]));

    // get view space position at 1 pixel offsets in each major direction
    float3 viewSpacePos_r = FFXIV::get_position_from_uv(taps[1], GetDepth(taps[1]));
    float3 viewSpacePos_u = FFXIV::get_position_from_uv(taps[2], GetDepth(taps[2]));

    // get the difference between the current and each offset position
    float3 hDeriv = viewSpacePos_r - viewSpacePos_c;
    float3 vDeriv = viewSpacePos_u - viewSpacePos_c;

    // get view space normal from the cross product of the diffs
    float3 viewNormal = normalize(cross(hDeriv, vDeriv));

    return viewNormal;
}

bool IsWater(float2 texcoord)
{
    float4 decal = tex2Dlod(FFXIV::DecalNormalMap, float4(texcoord, 0, 0));
    return (decal.a == FFXIV::MAGIC_WATER_ALPHA || decal.a == FFXIV::MAGIC_WATER_ALPHA2);
}

float3x3 rotmat(float3 axis, float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0 - c;
	return float3x3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s, 
	oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s, 
	oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

float3 get_normal(uint2 id, float vertex_blend_factor, float vertex_blend_water_factor)
{
    float2 texcoord = (float2(id) + 0.5) / ReShade::ScreenSize;
    
    float4 normal = tex2Dlod(FFXIV::NormalMap, float4(texcoord, 0, 0));
    float4 decal = tex2Dlod(FFXIV::DecalNormalMap, float4(texcoord, 0, 0));
    normal.rgb = lerp(normal.rgb, float3(0.5, 0.5, 1), normal.a == 1);
    
    decal.rgb = normalize(0.5 - decal.rgb);
    normal.rgb = normalize(0.5 - normal.rgb);
        
    float3x3 matV = float3x3(FFXIV::matViewInv[0].xyz, FFXIV::matViewInv[1].xyz, FFXIV::matViewInv[2].xyz);
    float3 cur_wsnormal = float4(mul(matV, decal.rgb), 1).rgb;
    float3 vNormal = ViewNormalAtPixelPosition(texcoord);
    
    normal.rgb = lerp(vNormal.rgb, normal.rgb, vertex_blend_factor);
    decal.rgb = lerp(vNormal.rgb, decal.rgb, vertex_blend_water_factor);
    
    // blend with decals (hair, water, etc.,...) while ignoring flat enough surfaces for shore lines
    //normal.rgb = lerp(normal.rgb, decal.rgb, decal.a > FFXIV::MAGIC_ALPHA && dot(cur_wsnormal, float3(0, -1, 0)) < 0.9999999); //0.999999
    normal.rgb = lerp(normal.rgb, decal.rgb, decal.a > FFXIV::MAGIC_ALPHA);
    
    // smooth out gbuffer inaccuracies by straightening out flat surfaces
    //normal.rgb = lerp(normal.rgb, vNormal.rgb, dot(normal.rgb, vNormal.rgb) > 0.999);
    
    return normal.rgb;
}

// Compute the position, the reflection direction, maxTraceDistance of the sample in texture space.
void ComputePosAndReflection(uint2 tid,
                             float3 vSampleNormalInVS,
                             out float3 outSamplePosInTS,
                             out float3 outReflDirInTS,
                             out float outMaxDistance,
                             out bool outTowards)
{
    float sampleDepth = getMinimumDepthPlane(tid, 0).r;
    float4 samplePosInCS =  float4(((float2(tid) + 0.5) / ReShade::ScreenSize) * 2 - 1.0f, sampleDepth, 1);
    samplePosInCS.y *= -1;

    float4 samplePosInVS = mul(FFXIV::matProjInv, samplePosInCS);
    samplePosInVS /= samplePosInVS.w;
    
    float3 vCamToSampleInVS = normalize(samplePosInVS.xyz);
    float4 vReflectionInVS = float4(reflect(vCamToSampleInVS.xyz, vSampleNormalInVS.xyz),0);
    
    float refAngle = dot(vCamToSampleInVS.xyz, vReflectionInVS.xyz);
    
    outTowards = false;
    if(refAngle < 0)
    {
        outTowards = true;
        return;
    }

    float4 vReflectionEndPosInVS = samplePosInVS + vReflectionInVS * FFXIV::z_far();
    float4 vReflectionEndPosInCS = mul(FFXIV::matProj, float4(vReflectionEndPosInVS.xyz, 1));
    vReflectionEndPosInCS /= vReflectionEndPosInCS.w;
    float3 vReflectionDir = normalize((vReflectionEndPosInCS - samplePosInCS).xyz);

    // Transform to texture space
    samplePosInCS.xy *= float2(0.5f, -0.5f);
    samplePosInCS.xy += float2(0.5f, 0.5f);
    
    vReflectionDir.xy *= float2(0.5f, -0.5f);
    
    outSamplePosInTS = samplePosInCS.xyz;
    outReflDirInTS = vReflectionDir;
    
    // Compute the maximum distance to trace before the ray goes outside of the visible area.
    outMaxDistance = outReflDirInTS.x>=0 ? (1 - outSamplePosInTS.x)/outReflDirInTS.x  : -outSamplePosInTS.x/outReflDirInTS.x;
    outMaxDistance = min(outMaxDistance, outReflDirInTS.y<0 ? (-outSamplePosInTS.y/outReflDirInTS.y) : ((1-outSamplePosInTS.y)/outReflDirInTS.y));
    outMaxDistance = min(outMaxDistance, outReflDirInTS.z<0 ? (-outSamplePosInTS.z/outReflDirInTS.z) : ((1-outSamplePosInTS.z)/outReflDirInTS.z));
}

float3 intersectDepthPlane(float3 o, float3 d, float z)
{
    return o + d * z;
}

float2 getCell(float2 pos, float2 cell_count)
{
    return float2(floor(pos*cell_count));
}

float3 intersectCellBoundary(float3 o, float3 d, float2 cell, float2 cell_count, float2 crossStep, float2 crossOffset)
{
    float3 intersection = 0;
    
    float2 index = cell + crossStep;
    float2 boundary = index / cell_count;
    boundary += crossOffset;
    
    float2 delta = boundary - o.xy;
    delta /= d.xy;
    float t = min(delta.x, delta.y);
    
    intersection = intersectDepthPlane(o, d, t);
    intersection.xy += (delta.x < delta.y) ? float2(crossOffset.x, 0.0) : float2(0.0, crossOffset.y);
    
    return intersection;
}

float2 getCellCount(int mipLevel)
{
    uint2 dim = uint2(BUFFER_WIDTH >> mipLevel, BUFFER_HEIGHT >> mipLevel);
    return float2(dim.x, dim.y);
}
 
bool crossedCellBoundary(float2 oldCellIndex, float2 newCellIndex)
{
    return (oldCellIndex.x != newCellIndex.x) || (oldCellIndex.y != newCellIndex.y);
}

bool FindIntersection_HiZ(float3 samplePosInTS, float3 vReflDirInTS, float maxTraceDistance, out float3 intersection)
{
    const int maxLevel = NUM_MIPMAPS;
    int startLevel = maxLevel / 2;
    int stopLevel = 0;
    
    float2 crossStep = float2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
    float2 crossOffset = crossStep / ReShade::ScreenSize / 1000;
    crossStep = saturate(crossStep);
    
    float3 ray = samplePosInTS.xyz;
    
    float minZ = ray.z;
    float maxZ = ray.z + vReflDirInTS.z * maxTraceDistance;
    float deltaZ = (maxZ - minZ);
    bool isBackwardRay = vReflDirInTS.z<0;
    float rayDir = isBackwardRay ? -1 : 1;

    float3 d = vReflDirInTS * maxTraceDistance;
    float3 o = ray;
    float MAX_THICKNESS = DEPTH_BIAS / 100000;
    
    float2 startCellCount = getCellCount(startLevel);
    float2 rayCell = getCell(ray.xy, startCellCount);
    
    ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset * 1000);
    
    int level = startLevel;
    uint iter = 0;
    
    [loop]
    while(level >= stopLevel && ray.z*rayDir <= maxZ*rayDir && iter<MAX_ITERATION)
    {
        const float2 cellCount = getCellCount(level);
        const float2 oldCellIdx = getCell(ray.xy, cellCount);
        
        float2 cell_minmaxZ = getMinimumDepthPlane(oldCellIdx, level);
        float thicknessG = (ray.z - cell_minmaxZ.g);
        float thicknessR = (cell_minmaxZ.r - ray.z);
        
        float3 tmpRay = ray;
        if((cell_minmaxZ.r > ray.z) && !isBackwardRay)
        {
            tmpRay = intersectDepthPlane(o, d, (cell_minmaxZ.r - minZ)/deltaZ);
        }
        else if((cell_minmaxZ.g < ray.z) && isBackwardRay)
        {
            tmpRay = intersectDepthPlane(o, d, (cell_minmaxZ.g - minZ)/deltaZ );
        }
        
        const float2 newCellIdx = getCell(tmpRay.xy, cellCount);

        [branch]
        if(crossedCellBoundary(oldCellIdx, newCellIdx) || (cell_minmaxZ.g < ray.z) && !isBackwardRay && thicknessG > MAX_THICKNESS || (cell_minmaxZ.r > ray.z) && isBackwardRay && thicknessR > MAX_THICKNESS)
        //if(crossedCellBoundary(oldCellIdx, newCellIdx) || (cell_minmaxZ.g < ray.z) && !isBackwardRay || (cell_minmaxZ.r > ray.z) && isBackwardRay)
        {
            tmpRay = intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset);
            level = min(maxLevel, level + 2);
        }
        
        ray = tmpRay;
        --level;
        
        ++iter;
    }
    
    intersection = ray;    
    
    return ray.x >= 0 && ray.x <= 1 && ray.y >= 0 && ray.y <= 1;
}

//----------------

float4 PS_WSPosBeforeWater(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target0
{
    return float4(FFXIV::get_world_position_from_uv(texcoord, GetDepth(texcoord)), 1);
}

float PS_Reflectivity(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target0
{
    float amount = tex2D(ReflectionMaskInput, texcoord).a;
    
    if(IsWater(texcoord))
    {
        float3 positionWorld = tex2D(sFFXIV_WaterDepth, texcoord).xyz;
        float3 positionWater = FFXIV::get_world_position_from_uv(texcoord, GetDepth(texcoord));
        float lenA = abs(positionWater.y - positionWorld.y);
        float waterFade = saturate((lenA - WATER_DEPTH_THRESHOLD) * WATER_DEPTH_FACTOR);
        
        return saturate(amount + waterFade);
    }
    
    return amount;
}

float4 PS_DoTheOtherThing(in float4 position : SV_Position, in float2 coord : TEXCOORD) : SV_Target0
{
    float4 orgColor = tex2D(ReShade::BackBuffer, coord);
    float4 refData = tex2Dlod(sUVMap, float4(coord.xy, 0, 0));
    
    #if FFXIV_REPROJECT_PREVIOUS_FRAME == 0
    float4 refColor = tex2D(sFFXIV_SSR_Color, refData.xy);
    #else
    float2 motion = tex2D(SamplerMotionVectors, refData.xy).xy;
    float4 refColor = tex2D(sFFXIV_PrevFrame, refData.xy + motion);
    #endif
    
    return float4(lerp(orgColor.rgb, refColor.rgb, refData.a), orgColor.a);
}


void CS_DoTheThing(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
    if(id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT)
        return;
    
    float amount = tex2Dfetch(sFFXIV_Reflectivity, id.xy).r;
    
    float3 normal = get_normal(id.xy, BUMPINESS_GENERAL, BUMPINESS_DECAL);
    
    // Find intersection in texture space by tracing the reflection ray
    float3 intersection = 0;

    if (amount > 0)
    {
        float3 samplePosInTS = 0;
        float3 vReflDirInTS = 0;
        float maxTraceDistance = 0;
        bool towardsCamera = false;
        // Compute the position, the reflection vector, maxTraceDistance of this sample in texture space.
        ComputePosAndReflection(id.xy, normal, samplePosInTS, vReflDirInTS, maxTraceDistance, towardsCamera);
        
        if(!towardsCamera)
        {
            bool intersected = FindIntersection_HiZ(samplePosInTS, vReflDirInTS, maxTraceDistance, intersection);
            
            if (intersected)
            {
                // Fade at edges
                if (intersection.y < EDGE_CUT_OFF * 2)
                    amount *= (intersection.y / EDGE_CUT_OFF / 2);
                    
                tex2Dstore(stUVMap, id.xy, float4(intersection.xy, 0, amount * INTENSITY));
                return;
            }
        }
    }

    // If it didn't hit anything, then just simply return nothing
    tex2Dstore(stUVMap, id.xy, float4(0, 0, 0, 0));
}

//----------------

technique FFXIV_SSR
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_Reflectivity;
        RenderTarget = FFXIV_Reflectivity;
    }

    pass
    {
        ComputeShader = CS_DoTheThing<CS_DISPATCH_GROUPS, CS_DISPATCH_GROUPS>;
        DispatchSizeX = CEILING_DIV(BUFFER_WIDTH, CS_DISPATCH_GROUPS);
        DispatchSizeY = CEILING_DIV(BUFFER_HEIGHT, CS_DISPATCH_GROUPS);
        GenerateMipMaps = false;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_DoTheOtherThing;
    }
}

technique FFXIV_SSR_BEFORE_WATER
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_WSPosBeforeWater;
        RenderTarget = FFXIV_WaterDepth;
    }
}