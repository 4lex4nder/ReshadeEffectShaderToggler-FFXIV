///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation 
// 
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion", 
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// 
// Implementation:  Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>         (\_/)
// Version:         (see XeGTAO.h)                                                                            (='.'=)
// Details:         https://github.com/GameTechDev/XeGTAO                                                     (")_(")
//
// Version history: see XeGTAO.h
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// - Visibility bitmask based on "Screen Space Indirect Lighting with Visibility Bitmask" (https://arxiv.org/abs/2301.11376)
// - Clamping hint: YASSGI by Pentalimbed (https://github.com/Pentalimbed/YASSGI/)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#if defined( XE_GTAO_SHOW_NORMALS ) || defined( XE_GTAO_SHOW_EDGES ) || defined( XE_GTAO_SHOW_BENT_NORMALS )
//RWTexture2D<float4>         g_outputDbgImage    : register( u2 );
//#endif
//
//#include "XeGTAO.h"
#include "ffxiv_common.fxh"
#include "ReShade.fxh"

#define XE_GTAO_PI               	(3.1415926535897932384626433832795)
#define XE_GTAO_PI_HALF             (1.5707963267948966192313216916398)

#ifndef XE_GTAO_QUALITY_LEVEL
	#define XE_GTAO_QUALITY_LEVEL 2		// 0: low; 1: medium; 2: high; 3: ultra
#endif

#ifndef XE_GTAO_RESOLUTION_SCALE
	#define XE_GTAO_RESOLUTION_SCALE 0		// 0: full; 1: half; 2: quarter ...
#endif

#ifndef XE_GTAO_IL
	#define XE_GTAO_IL 0	// 0: disabled; 1: approximated GTAO multi bounce
#endif

#undef XE_GTAO_SLICE_COUNT
#undef XE_GTAO_SLICE_STEP_COUNT

#if XE_GTAO_QUALITY_LEVEL <= 0
	#define XE_GTAO_SLICE_COUNT 1
	#define XE_GTAO_SLICE_STEP_COUNT 2
#elif XE_GTAO_QUALITY_LEVEL == 1
	#define XE_GTAO_SLICE_COUNT 2
	#define XE_GTAO_SLICE_STEP_COUNT 2
#elif XE_GTAO_QUALITY_LEVEL == 2
	#define XE_GTAO_SLICE_COUNT 3
	#define XE_GTAO_SLICE_STEP_COUNT 3
#elif XE_GTAO_QUALITY_LEVEL >= 3
	#define XE_GTAO_SLICE_COUNT 9
	#define XE_GTAO_SLICE_STEP_COUNT 3
#endif

#define XE_GTAO_QWORD_BIT_WIDTH 32
#define XE_GTAO_BITMASK_NUM_BITS float(XE_GTAO_QWORD_BIT_WIDTH * 1)

#define XE_GTAO_DENOISE_BLUR_BETA (1.2)
//#define XE_GTAO_DENOISE_BLUR_BETA (1e4f)

#define XE_GTAO_DEPTH_MIP_LEVELS 5

#define CEILING_DIV(X, Y) (((X) + (Y) - 1) / (Y)) //(((X)/(Y)) + ((X) % (Y) != 0))
#define CS_DISPATCH_GROUPS 16

#define XE_GTAO_SCALED_BUFFER_WIDTH (BUFFER_WIDTH >> XE_GTAO_RESOLUTION_SCALE)
#define XE_GTAO_SCALED_BUFFER_HEIGHT (BUFFER_HEIGHT >> XE_GTAO_RESOLUTION_SCALE)
#define XE_GTAO_SCALED_BUFFER_RCP_WIDTH (1.0 / XE_GTAO_SCALED_BUFFER_WIDTH)
#define XE_GTAO_SCALED_BUFFER_RCP_HEIGHT (1.0 / XE_GTAO_SCALED_BUFFER_HEIGHT)
#define XE_GTAO_SCALED_BUFFER_PIXEL_SIZE float2(XE_GTAO_SCALED_BUFFER_RCP_WIDTH, XE_GTAO_SCALED_BUFFER_RCP_HEIGHT)
#define XE_GTAO_SCALED_BUFFER_SCREEN_SIZE float2(XE_GTAO_SCALED_BUFFER_WIDTH, XE_GTAO_SCALED_BUFFER_HEIGHT)

//#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uniform float FrameCount < source = "framecount"; >;
uniform float Time < source = "timer"; >;

uniform float constEffectRadius <
	ui_type = "drag";
	ui_min = 0.01; ui_max = 100.0;
	ui_step = 0.01;
	ui_label = "Effect Radius";
	ui_tooltip = "Effect Radius";
> = 0.5;

uniform float constRadiusMultiplier <
	ui_type = "drag";
	ui_min = 0.3; ui_max = 3.0;
	ui_step = 0.01;
	ui_label = "Effect Radius Multiplier";
	ui_tooltip = "Effect Radius Multiplier";
> = 1.457;

uniform float constEffectFalloffRange <
	ui_type = "drag";
	ui_min = 0.01; ui_max = 1.0;
	ui_step = 0.01;
	ui_label = "Effect Falloff Range";
	ui_tooltip = "Effect Falloff Range";
> = 0.615;

uniform float constFinalValuePower <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 5.0;
	ui_step = 0.01;
	ui_label = "Final Value Power";
	ui_tooltip = "Final Value Power";
> = 2.2;

uniform float constDepthMIPSamplingOffset <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 30.0;
	ui_step = 0.01;
	ui_label = "Depth Mip Sampling Offset";
	ui_tooltip = "Depth Mip Sampling Offset";
> = 3.30;

uniform float constSampleDistributionPower <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 3.0;
	ui_step = 0.01;
	ui_label = "Sample Distribution Power";
	ui_tooltip = "Sample Distribution Power";
> = 2.0;

uniform float constThinOccluderCompensation <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 5.0;
	ui_step = 0.01;
	ui_label = "Thin Occluder Compensation";
	ui_tooltip = "Thin Occluder Compensation";
> = 0.35;

#if XE_GTAO_IL > 0
//uniform float FFXIV_LIGHT_SOURCE_INTENSITY_MULTIPLIER <
//    ui_label = "Shiny Multiplier";
//     ui_tooltip = "Enhance light source intensity for GI calculation";
//	 ui_type = "drag";
//    ui_min = 0; ui_max = 1000.0;
//    ui_step = 1.0;
//> = 0;

uniform float FFXIV_IL_INTENSITY <
    ui_label = "Indirect Light Intensity";
     ui_tooltip = "Indirect Light Intensity";
	 ui_type = "drag";
    ui_min = 0; ui_max = 5.0;
    ui_step = 0.1;
> = 1.0;
#endif

uniform bool Debug <
	ui_type = "radio";
	ui_label = "Debug";
	ui_tooltip = "Debug AO";
> = 0;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
texture2D g_srcNDCDepth : DEPTH;
sampler2D g_sSrcNDCDepth { Texture = g_srcNDCDepth; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

texture2D g_srcWorkingDepth
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	MipLevels = XE_GTAO_DEPTH_MIP_LEVELS;
	Format = R32F;
};

storage2D g_outWorkingDepthMIP0 { Texture = g_srcWorkingDepth; MipLevel = 0; };
storage2D g_outWorkingDepthMIP1 { Texture = g_srcWorkingDepth; MipLevel = 1; };
storage2D g_outWorkingDepthMIP2 { Texture = g_srcWorkingDepth; MipLevel = 2; };
storage2D g_outWorkingDepthMIP3 { Texture = g_srcWorkingDepth; MipLevel = 3; };
storage2D g_outWorkingDepthMIP4 { Texture = g_srcWorkingDepth; MipLevel = 4; };
sampler2D g_sSrcWorkingDepth
{
	Texture = g_srcWorkingDepth;
	
#if XE_GTAO_RESOLUTION_SCALE <= 0
	MagFilter = POINT;
	MinFilter = POINT;
	MipFilter = POINT;
#endif
};

texture2D g_srcWorkingAOTerm
{
	Width = XE_GTAO_SCALED_BUFFER_WIDTH;
	Height = XE_GTAO_SCALED_BUFFER_HEIGHT;
	Format = R8;
};

storage2D g_outWorkingAOTerm { Texture = g_srcWorkingAOTerm; };
sampler2D g_sSrcWorkinAOTerm
{
	Texture = g_srcWorkingAOTerm;

#if XE_GTAO_RESOLUTION_SCALE <= 0
	MagFilter = POINT;
	MinFilter = POINT;
	MipFilter = POINT;
#endif
};

texture2D g_srcFilteredOutput0
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = R8;
};

sampler2D g_sSrcFilteredOutput0 {
	Texture = g_srcFilteredOutput0;

	MagFilter = POINT;
	MinFilter = POINT;
	MipFilter = POINT;
};

texture2D g_srcFilteredOutput1
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = R8;
};

sampler2D g_sSrcFilteredOutput1
{
	Texture = g_srcFilteredOutput1;

	MagFilter = POINT;
	MinFilter = POINT;
	MipFilter = POINT;
};

texture2D g_srcCurNomals
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA8;
};

sampler2D g_sSrcCurNomals
{
	Texture = g_srcCurNomals;

	MagFilter = POINT;
	MinFilter = POINT;
	MipFilter = POINT;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// From https://www.shadertoy.com/view/3tB3z3 - except we're using R2 here
#define XE_HILBERT_LEVEL    6U
#define XE_HILBERT_WIDTH    ( (1U << XE_HILBERT_LEVEL) )
#define XE_HILBERT_AREA     ( XE_HILBERT_WIDTH * XE_HILBERT_WIDTH )
uint HilbertIndex( uint posX, uint posY )
{   
    uint index = 0U;
    for( uint curLevel = XE_HILBERT_WIDTH/2U; curLevel > 0U; curLevel /= 2U )
    {
        uint regionX = ( posX & curLevel ) > 0U;
        uint regionY = ( posY & curLevel ) > 0U;
        index += curLevel * curLevel * ( (3U * regionX) ^ regionY);
        if( regionY == 0U )
        {
            if( regionX == 1U )
            {
                posX = uint( (XE_HILBERT_WIDTH - 1U) ) - posX;
                posY = uint( (XE_HILBERT_WIDTH - 1U) ) - posY;
            }

            uint temp = posX;
            posX = posY;
            posY = temp;
        }
    }
    return index;
}

float2 SpatioTemporalNoise( uint2 pixCoord )    
{
	uint temporalIndex = 0;//FrameCount % 64; // without TAA, temporalIndex is always 0
	// Hilbert curve driving R2 (see https://www.shadertoy.com/view/3tB3z3)
    uint index = HilbertIndex( pixCoord.x, pixCoord.y );
    index += 0*(temporalIndex%64); // why 288? tried out a few and that's the best so far (with XE_HILBERT_LEVEL 6U) - but there's probably better :)
    // R2 sequence - see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
	return float2( frac( 0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114) ) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Inputs are screen XY and viewspace depth, output is viewspace position
float3 XeGTAO_ComputeViewspacePosition( const float2 screenPos, const float viewspaceDepth )
{
    return FFXIV::get_position_from_uv(screenPos, FFXIV::delinearize_depth(FFXIV::view_z_to_depth(viewspaceDepth)));
}

float XeGTAO_ScreenSpaceToViewSpaceDepth( const float screenDepth )
{
	return FFXIV::linearize_depth(screenDepth) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

// http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
float XeGTAO_FastSqrt( float x )
{
    return (float)(asfloat( 0x1fbd1df5 + ( asint( x ) >> 1 ) ));
}
// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
float XeGTAO_FastACos( float inX )
{ 
    const float PI = 3.141593;
    const float HALF_PI = 1.570796;
    float x = abs(inX); 
    float res = -0.156583 * x + HALF_PI; 
    res *= XeGTAO_FastSqrt(1.0 - x); 
    return (inX >= 0) ? res : PI - res; 
}

void XeGTAO_OutputWorkingTerm( const uint2 pixCoord, float visibility )
{
    visibility = saturate( visibility );

	tex2Dstore(g_outWorkingAOTerm, pixCoord, visibility);
}

void XeGTAO_MainPass( const uint2 pixCoord, const float2 localNoise, float3 viewspaceNormal )
{                                                                       
    float2 normalizedScreenPos = (pixCoord + 0.5.xx) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;

    // viewspace Z at the center
	float viewspaceZ  = tex2Dlod(g_sSrcWorkingDepth, float4(normalizedScreenPos, 0, 0)).x; 

    // Move center pixel slightly towards camera to avoid imprecision artifacts due to depth buffer imprecision; offset depends on depth texture format used
    viewspaceZ *= 0.99999;     // this is good for FP32 depth buffer

    const float3 pixCenterPos   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
    const float3 viewVec      = normalize(-pixCenterPos);

    float visibility = 0;
	
    // see "Algorithm 1" in https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
    {
        const float noiseSlice  = (float)localNoise.x;
        const float noiseSample = (float)localNoise.y;

        // quality settings / tweaks / hacks
        const float pixelTooCloseThreshold  = 1.3;      // if the offset is under approx pixel size (pixelTooCloseThreshold), push it out to the minimum distance

        // approx viewspace pixel size at pixCoord; approximation of NDCToViewspace( normalizedScreenPos.xy + ReShade::PixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = XeGTAO_ComputeViewspacePosition(normalizedScreenPos + XE_GTAO_SCALED_BUFFER_PIXEL_SIZE, viewspaceZ).xy - pixCenterPos.xy;

        float screenspaceRadius   = (constEffectRadius * constRadiusMultiplier) / pixelDirRBViewspaceSizeAtCenterZ.x;

        // fade out for small screen radii 
        visibility += saturate((10 - screenspaceRadius)/100)*0.5;

        // this is the min distance to start sampling from to avoid sampling from the center pixel (no useful data obtained from sampling center pixel)
        const float minS = (float)pixelTooCloseThreshold / screenspaceRadius;
		const float thinness = (pixCenterPos.z / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
		
        [unroll]
        for( float slice = 0; slice < float(XE_GTAO_SLICE_COUNT); slice++ )
        {
            float sliceK = (slice + noiseSlice) / float(XE_GTAO_SLICE_COUNT);
            // lines 5, 6 from the paper
            float phi = sliceK * XE_GTAO_PI;
			float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            float2 omega = float2(cosPhi, -sinPhi);       //float2 on omega causes issues with big radii

            // convert to screen units (pixels) for later use
            omega *= screenspaceRadius;
			
			uint bi = 0;
			
            [unroll]
            for( float step = 0; step < float(XE_GTAO_SLICE_STEP_COUNT); step++ )
            {
                // R1 sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
                const float stepBaseNoise = float(slice + step * float(XE_GTAO_SLICE_STEP_COUNT)) * 0.6180339887498948482; // <- this should unroll
                float stepNoise = frac(noiseSample + stepBaseNoise);

                // approx line 20 from the paper, with added noise
				float s = (step+stepNoise) / float(XE_GTAO_SLICE_STEP_COUNT); // + (float2)1e-6f);

                // additional distribution modifier
                s       = pow( s, constSampleDistributionPower );

                // avoid sampling center pixel
                s       += minS;

				// approx lines 21-22 from the paper, unrolled
				float2 sampleOffset = s * omega; //* (1.0 + pixCenterPos.z / FFXIV::z_far() * 200.0);
				
				float sampleOffsetLength = length( sampleOffset );
	
				// note: when sampling, using point_point_point or point_point_linear sampler works, but linear_linear_linear will cause unwanted interpolation between neighbouring depth values on the same MIP level!
				float mipLevel    = (float)clamp( log2( sampleOffsetLength ) - constDepthMIPSamplingOffset, 0, XE_GTAO_DEPTH_MIP_LEVELS );
				
				sampleOffset = round(sampleOffset) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;

				float SZ0 = tex2Dlod( g_sSrcWorkingDepth, float4(normalizedScreenPos + sampleOffset, 0, mipLevel) ).x;
				float SZ1 = tex2Dlod( g_sSrcWorkingDepth, float4(normalizedScreenPos - sampleOffset, 0, mipLevel) ).x;
				
				float3 sf0 = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + sampleOffset, SZ0 );
				float3 sf1 = XeGTAO_ComputeViewspacePosition( normalizedScreenPos - sampleOffset, SZ1 );
				
				float3 sb0 = sf0 - viewVec * constThinOccluderCompensation;
				float3 sb1 = sf1 - viewVec * constThinOccluderCompensation;
				
				float phi_f0 = XeGTAO_FastACos(dot(normalize(sf0 - pixCenterPos), viewspaceNormal));
				float phi_b0 = XeGTAO_FastACos(dot(normalize(sb0 - pixCenterPos), viewspaceNormal));
				
				float phi_f1 = XeGTAO_FastACos(dot(normalize(sf1 - pixCenterPos), viewspaceNormal));
				float phi_b1 = XeGTAO_FastACos(dot(normalize(sb1 - pixCenterPos), viewspaceNormal));

				float2 min_max0 = float2(min(phi_f0, phi_b0), max(phi_f0, phi_b0));
				float2 min_max1 = float2(min(phi_f1, phi_b1), max(phi_f1, phi_b1));
				
				min_max0 = clamp(min_max0, 0, XE_GTAO_PI_HALF);
				min_max1 = clamp(min_max1, 0, XE_GTAO_PI_HALF);
				
				uint2 a_b0 = uint2(min_max0.x / XE_GTAO_PI * XE_GTAO_BITMASK_NUM_BITS, (min_max0.y - min_max0.x) / XE_GTAO_PI * XE_GTAO_BITMASK_NUM_BITS);
				uint2 a_b1 = uint2((min_max1.x + XE_GTAO_PI_HALF) / XE_GTAO_PI * XE_GTAO_BITMASK_NUM_BITS, (min_max1.y - min_max1.x) / XE_GTAO_PI * XE_GTAO_BITMASK_NUM_BITS);
				
				bi |= ((1 << a_b0.y) - 1) << a_b0.x;
				bi |= ((1 << a_b1.y) - 1) << a_b1.x;
            }
			
			visibility += 1.0 - countbits(bi) / XE_GTAO_BITMASK_NUM_BITS;
        }
		
        visibility /= float(XE_GTAO_SLICE_COUNT);
        visibility = pow( visibility, constFinalValuePower );
        visibility = max( 0.03, visibility ); // disallow total occlusion (which wouldn't make any sense anyhow since pixel is visible but also helps with packing bent normals)
    }

    XeGTAO_OutputWorkingTerm( pixCoord, visibility );
}

// weighted average depth filter
float XeGTAO_DepthMIPFilter( float depth0, float depth1, float depth2, float depth3 )
{
    float maxDepth = max( max( depth0, depth1 ), max( depth2, depth3 ) );

    const float depthRangeScaleFactor = 0.75; // found empirically :)
    const float effectRadius              = depthRangeScaleFactor * (float)constEffectRadius * (float)constRadiusMultiplier;
    const float falloffRange              = (float)constEffectFalloffRange * effectRadius;
    const float falloffFrom       = effectRadius * ((float)1-(float)constEffectFalloffRange);
    // fadeout precompute optimisation
    const float falloffMul        = (float)-1.0 / ( falloffRange );
    const float falloffAdd        = falloffFrom / ( falloffRange ) + (float)1.0;

    float weight0 = saturate( (maxDepth-depth0) * falloffMul + falloffAdd );
    float weight1 = saturate( (maxDepth-depth1) * falloffMul + falloffAdd );
    float weight2 = saturate( (maxDepth-depth2) * falloffMul + falloffAdd );
    float weight3 = saturate( (maxDepth-depth3) * falloffMul + falloffAdd );

    float weightSum = weight0 + weight1 + weight2 + weight3;
    return (weight0 * depth0 + weight1 * depth1 + weight2 * depth2 + weight3 * depth3) / weightSum;
}

// This is also a good place to do non-linear depth conversion for cases where one wants the 'radius' (effectively the threshold between near-field and far-field GI), 
// is required to be non-linear (i.e. very large outdoors environments).
float XeGTAO_ClampDepth( float depth )
{
    return clamp( depth, 0.0, 3.402823466e+38 );
}

groupshared float g_scratchDepths[8 * 8];
static const uint2 g_scratchSize = uint2(8, 8);

void XeGTAO_PrefilterDepths16x16( uint2 dispatchThreadID /*: SV_DispatchThreadID*/, uint2 groupThreadID /*: SV_GroupThreadID*/ )
{
	const uint soffset = groupThreadID.x * g_scratchSize.y + groupThreadID.y;
	
    // MIP 0
    const uint2 baseCoord = dispatchThreadID;
    const uint2 pixCoord = baseCoord * 2;
	float4 depths4 = tex2DgatherR( g_sSrcNDCDepth, float2( pixCoord * ReShade::PixelSize ), int2(1,1) );
    float depth0 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.w ) );
    float depth1 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.z ) );
    float depth2 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.x ) );
    float depth3 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.y ) );
	tex2Dstore(g_outWorkingDepthMIP0, pixCoord + uint2(0, 0), (float)depth0);
	tex2Dstore(g_outWorkingDepthMIP0, pixCoord + uint2(1, 0), (float)depth1);
	tex2Dstore(g_outWorkingDepthMIP0, pixCoord + uint2(0, 1), (float)depth2);
	tex2Dstore(g_outWorkingDepthMIP0, pixCoord + uint2(1, 1), (float)depth3);

    // MIP 1
    float dm1 = XeGTAO_DepthMIPFilter( depth0, depth1, depth2, depth3 );
	tex2Dstore(g_outWorkingDepthMIP1, baseCoord, (float)dm1);
	g_scratchDepths[soffset] = dm1;

	barrier();

    // MIP 2
    [branch]
    if( all( ( groupThreadID.xy % (2).xx ) == 0 ) )
    {
		float inTL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inTR = g_scratchDepths[(groupThreadID.x + 1) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inBL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 1)];
		float inBR = g_scratchDepths[(groupThreadID.x + 1) * g_scratchSize.y + (groupThreadID.y + 1)];

        float dm2 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR );
		tex2Dstore(g_outWorkingDepthMIP2, baseCoord / 2, (float)dm2);
		g_scratchDepths[soffset] = dm2;
    }

	barrier();

    // MIP 3
    [branch]
    if( all( ( groupThreadID.xy % (4).xx ) == 0 ) )
    {
		float inTL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inTR = g_scratchDepths[(groupThreadID.x + 2) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inBL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 2)];
		float inBR = g_scratchDepths[(groupThreadID.x + 2) * g_scratchSize.y + (groupThreadID.y + 2)];

        float dm3 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR );
		tex2Dstore(g_outWorkingDepthMIP3, baseCoord / 4, (float)dm3);
		g_scratchDepths[soffset] = dm3;
    }

	barrier();

    // MIP 4
    [branch]
    if( all( ( groupThreadID.xy % (8).xx ) == 0 ) )
    {
		float inTL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inTR = g_scratchDepths[(groupThreadID.x + 4) * g_scratchSize.y + (groupThreadID.y + 0)];
		float inBL = g_scratchDepths[(groupThreadID.x + 0) * g_scratchSize.y + (groupThreadID.y + 4)];
		float inBR = g_scratchDepths[(groupThreadID.x + 4) * g_scratchSize.y + (groupThreadID.y + 4)];

        float dm4 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR );
		tex2Dstore(g_outWorkingDepthMIP4, baseCoord / 8, (float)dm4);
    }
}

#undef AOTermType
#define AOTermType float

float3 LoadNormal(const uint2 id)
{
	float2 texcoord = (float2(id) + 0.5) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;
	float3 normal = tex2Dlod(g_sSrcCurNomals, float4(texcoord, 0, 0)).rgb;
	return normalize(normal - 0.5);
}

float3 LoadNormalUV(const float2 texcoord)
{
	float3 normal = tex2Dlod(g_sSrcCurNomals, float4(texcoord, 0, 0)).rgb;
	return normalize(normal - 0.5);
}

float4 XeGTAO_PSAtrous( const float2 viewcoord, const float stepwidth, sampler visibilitySampler )
{
	static const float2 offset[9] = {float2(-1.0, -1.0), float2(0.0, -1.0), float2(1.0, -1.0), float2(-1.0, 0.0), float2(0.0, 0.0), float2(1.0, 0.0), float2(-1.0, 1.0), float2(0.0, 1.0), float2(1.0, 1.0)};
    static const float kernel[9] = {1.0f/16.0f, 3.0f/32.0f, 1.0f/16.0f, 3.0f/32.0f, 9.0f/64.0f, 3.0f/32.0f, 1.0f/16.0f, 3.0f/32.0f, 1.0f/16.0f};
    
    float sum = 0.0;
    float cum_w = 0.0;
	
    float c_phi = 1.0;
    float n_phi = 0.01;
    //float p_phi = 0.5;
	
	float2 uv = viewcoord;
    
	float cval = tex2Dlod(visibilitySampler, float4(uv, 0, 0)).r;
	float3 nval = LoadNormalUV(uv);
	
	[unroll]
    for(int i=0; i<9; i++)
    {
        float2 uvOffset = uv + offset[i] * stepwidth * ReShade::PixelSize;
        
        float ctmp = tex2Dlod(visibilitySampler, float4(uvOffset, 0, 0)).r;
        float t = cval - ctmp;
        float dist2 = t*t;
        float c_w = min(exp(-(dist2)/c_phi), 1.0);
        
        float3 ntmp = LoadNormalUV(uvOffset);
        float3 tt = nval - ntmp;
        dist2 = max(dot(tt,tt) / (stepwidth * stepwidth), 0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);
        
        float weight = c_w * n_w; //* p_w;
        sum += ctmp * weight * kernel[i];
        cum_w += weight * kernel[i];
    }
    
    float res = sum / cum_w;
	
	return float4(res, 0, 0, 1.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSPrefilterDepths16x16(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
	XeGTAO_PrefilterDepths16x16( id.xy, tid.xy );
}

void CSGTAO(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
	XeGTAO_MainPass( id.xy, SpatioTemporalNoise(id.xy), LoadNormal(id.xy) );
}

#if XE_GTAO_IL > 0
float3 reinhard(float3 v)
{
    return v / (1.0 + v);
}

float3 raushard(float3 v)
{
    return -v / (v - 1.0);
}

float3 GTAOMultiBounce(float visibility, float3 albedo)
{
	float3 a = 2.0404 * albedo - 0.3324;
	float3 b = -4.7951 * albedo + 0.6417;
	float3 c = 2.7552 * albedo + 0.6903;
	
	return max(visibility, ((visibility * a + b) * visibility + c) * visibility);
}
#endif

void PS_Atrous0(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output: SV_Target0)
{
	output = XeGTAO_PSAtrous(texcoord, 4.0, g_sSrcWorkinAOTerm);
}

void PS_Atrous1(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output: SV_Target0)
{
	output = XeGTAO_PSAtrous(texcoord, 2.0, g_sSrcFilteredOutput0);
}

void PS_Atrous2(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output: SV_Target0)
{
	output = XeGTAO_PSAtrous(texcoord, 1.0, g_sSrcFilteredOutput1);
}

void PS_CurGBuf(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 normals : SV_Target0)
{
	float3 normal = FFXIV::get_normal(texcoord);
	normals = float4(normal, 1.0);
}

void PS_ApplyAO(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, in float4 orgColor, in float multi, out float4 output : SV_Target0)
{
	#if XE_GTAO_IL == 0
	float aoTerm = tex2D(g_sSrcFilteredOutput0, texcoord).r;
    aoTerm = lerp(aoTerm, 1.0, multi);
	
	orgColor.rgb *= aoTerm;
	
	output = Debug ? float4((aoTerm).rrr, orgColor.a) : float4(orgColor);
	#else
	float aoTerm = tex2D(g_sSrcFilteredOutput0, texcoord).r;
	
	float3 albedo = float3(orgColor.rgb);
	//albedo.rgb = reinhard(albedo.rgb) * FFXIV_IL_INTENSITY;
	
	float3 coloredAO = GTAOMultiBounce(aoTerm, albedo);
	
	orgColor.rgb *= coloredAO;
	
	output = Debug ? float4((coloredAO).rgb, orgColor.a) : float4(orgColor);
	#endif
}

void PS_Out(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	float decal = tex2D(FFXIV::sDecalNormals, texcoord).a;
    float decalWater = tex2D(FFXIV::sDecalWaterNormals, texcoord).a;
	float4 orgColor = tex2D(ReShade::BackBuffer, texcoord);

	//if(decal > FFXIV::MAGIC_ALPHA && decal != FFXIV::MAGIC_WATER_ALPHA && decal != FFXIV::MAGIC_WATER_ALPHA2)
    if (decal >= 0.004 && decalWater < 0.004 )
	{
		output = orgColor;
		return;
	}
	
	PS_ApplyAO(position, texcoord, orgColor, 0, output);
}

void PS_Out_Decals(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	float decal = tex2D(FFXIV::sDecalNormals, texcoord).a;
    float decalWater = tex2D(FFXIV::sDecalWaterNormals, texcoord).a;
	float4 orgColor = tex2D(ReShade::BackBuffer, texcoord);

	//if(decal <= FFXIV::MAGIC_ALPHA || decal == FFXIV::MAGIC_WATER_ALPHA || decal == FFXIV::MAGIC_WATER_ALPHA2)
    if (decal < 0.004 || decalWater >= 0.004)
	{
		output = orgColor;
		return;
	}
	
	PS_ApplyAO(position, texcoord, orgColor, 0, output);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
technique FFXIV_XeGTAO
{
pass
{
	ComputeShader = CSPrefilterDepths16x16<8, 8>;
	DispatchSizeX = ((BUFFER_WIDTH) + ((16) - 1)) / (16);
	DispatchSizeY = ((BUFFER_HEIGHT) + ((16) - 1)) / (16);
	GenerateMipMaps = false;
}

pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_CurGBuf;
	RenderTarget0 = g_srcCurNomals;
}

pass
{
	ComputeShader = CSGTAO<CS_DISPATCH_GROUPS, CS_DISPATCH_GROUPS>;
	DispatchSizeX = CEILING_DIV(XE_GTAO_SCALED_BUFFER_WIDTH, CS_DISPATCH_GROUPS);
	DispatchSizeY = CEILING_DIV(XE_GTAO_SCALED_BUFFER_HEIGHT, CS_DISPATCH_GROUPS);
}

pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_Atrous0;
	RenderTarget0 = g_srcFilteredOutput0;
}

pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_Atrous1;
	RenderTarget0 = g_srcFilteredOutput1;
}

pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_Atrous2;
	RenderTarget0 = g_srcFilteredOutput0;
}

pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_Out;
}
}

technique FFXIV_XeGTAO_Decals
{
pass
{
	VertexShader = PostProcessVS;
	PixelShader  = PS_Out_Decals;
}
}