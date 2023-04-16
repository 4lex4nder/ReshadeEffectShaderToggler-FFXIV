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

#define XE_GTAO_OCCLUSION_TERM_SCALE (1.5f)


#define XE_GTAO_DENOISE_BLUR_BETA (1.2)
//#define XE_GTAO_DENOISE_BLUR_BETA (1e4f)

#define XE_GTAO_DEPTH_MIP_LEVELS 5

#define CEILING_DIV(X, Y) (((X) + (Y) - 1) / (Y)) //(((X)/(Y)) + ((X) % (Y) != 0))
#define CS_DISPATCH_GROUPS 8

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
	ui_min = 0.0; ui_max = 0.7;
	ui_step = 0.01;
	ui_label = "Thin Occluder Compensation";
	ui_tooltip = "Thin Occluder Compensation";
> = 0.0;

uniform bool LightAvoidance <
	ui_type = "radio";
	ui_label = "LightAvoidance";
	ui_tooltip = "LightAvoidance";
> = 0;

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
	Format = R16F;
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
	Format = R16F;
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
	Format = R16F;
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
    //float3 ret;
    //ret.xy = (consts.NDCToViewMul * screenPos.xy + consts.NDCToViewAdd) * viewspaceDepth;
    //ret.z = viewspaceDepth;
    //return ret;
	
	return FFXIV::Compat::get_position_from_uv(screenPos, viewspaceDepth);
}

float XeGTAO_ScreenSpaceToViewSpaceDepth( const float screenDepth )
{
	return FFXIV::linearize_depth(screenDepth) * FFXIV::z_far();
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

void XeGTAO_OutputWorkingTerm( const uint2 pixCoord, float visibility, float3 bentNormal )
{
    visibility = saturate( visibility / float(XE_GTAO_OCCLUSION_TERM_SCALE) );

    //outWorkingAOTerm[pixCoord] = uint(visibility * 255.0 + 0.5);
	tex2Dstore(g_outWorkingAOTerm, pixCoord, visibility);
	//tex2Dstore(g_outWorkingAOTerm, pixCoord, visibility);
}

// "Efficiently building a matrix to rotate one vector to another"
// http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf / https://dl.acm.org/doi/10.1080/10867651.1999.10487509
// (using https://github.com/assimp/assimp/blob/master/include/assimp/matrix3x3.inl#L275 as a code reference as it seems to be best)
float3x3 XeGTAO_RotFromToMatrix( float3 from, float3 to )
{
    const float e       = dot(from, to);
    const float f       = abs(e); //(e < 0)? -e:e;

    // WARNING: This has not been tested/worked through, especially not for 16bit floats; seems to work in our special use case (from is always {0, 0, -1}) but wouldn't use it in general
    if( f > float( 1.0 - 0.0003 ) )
        return float3x3( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    const float3 v      = cross( from, to );
    /* ... use this hand optimized version (9 mults less) */
    const float h       = (1.0)/(1.0 + e);      /* optimization by Gottfried Chen */
    const float hvx     = h * v.x;
    const float hvz     = h * v.z;
    const float hvxy    = hvx * v.y;
    const float hvxz    = hvx * v.z;
    const float hvyz    = hvz * v.y;

    float3x3 mtx;
    mtx[0][0] = e + hvx * v.x;
    mtx[0][1] = hvxy - v.z;
    mtx[0][2] = hvxz + v.y;

    mtx[1][0] = hvxy + v.z;
    mtx[1][1] = e + h * v.y * v.y;
    mtx[1][2] = hvyz - v.x;

    mtx[2][0] = hvxz - v.y;
    mtx[2][1] = hvyz + v.x;
    mtx[2][2] = e + hvz * v.z;

    return mtx;
}

void XeGTAO_MainPass( const uint2 pixCoord, float sliceCount, float stepsPerSlice, const float2 localNoise, float3 viewspaceNormal )
{                                                                       
    float2 normalizedScreenPos = (pixCoord + 0.5.xx) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;

    //float4 valuesUL   = sourceViewspaceDepth.GatherRed( depthSampler, float2( pixCoord * ReShade::PixelSize )               );
    //float4 valuesBR   = sourceViewspaceDepth.GatherRed( depthSampler, float2( pixCoord * ReShade::PixelSize ), int2( 1, 1 ) );
	float4 valuesUL   = tex2DgatherR( g_sSrcWorkingDepth, float2( pixCoord * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE ) );
    //float4 valuesBR   = tex2DgatherR( g_sSrcWorkingDepth, float2( pixCoord * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE ), int2( 1, 1 ) );

    // viewspace Z at the center
    float viewspaceZ  = valuesUL.y; //sourceViewspaceDepth.SampleLevel( depthSampler, normalizedScreenPos, 0 ).x; 

    // viewspace Zs left top right bottom
    //const float pixLZ = valuesUL.x;
    //const float pixTZ = valuesUL.z;
    //const float pixRZ = valuesBR.z;
    //const float pixBZ = valuesBR.x;

    //float4 edgesLRTB  = XeGTAO_CalculateEdges( (float)viewspaceZ, (float)pixLZ, (float)pixRZ, (float)pixTZ, (float)pixBZ );
    //outWorkingEdges[pixCoord] = XeGTAO_PackEdges(edgesLRTB);
	//tex2Dstore(g_outWorkingEdges, pixCoord, XeGTAO_PackEdges(edgesLRTB));

	// Generating screen space normals in-place is faster than generating normals in a separate pass but requires
	// use of 32bit depth buffer (16bit works but visibly degrades quality) which in turn slows everything down. So to
	// reduce complexity and allow for screen space normal reuse by other effects, we've pulled it out into a separate
	// pass.
	// However, we leave this code in, in case anyone has a use-case where it fits better.
//#ifdef XE_GTAO_GENERATE_NORMALS_INPLACE
//    float3 CENTER   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
//    float3 LEFT     = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2(-1,  0) * ReShade::PixelSize, pixLZ );
//    float3 RIGHT    = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 1,  0) * ReShade::PixelSize, pixRZ );
//    float3 TOP      = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0, -1) * ReShade::PixelSize, pixTZ );
//    float3 BOTTOM   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0,  1) * ReShade::PixelSize, pixBZ );
//    viewspaceNormal = (float3)XeGTAO_CalculateNormal( edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM );
//#endif

    // Move center pixel slightly towards camera to avoid imprecision artifacts due to depth buffer imprecision; offset depends on depth texture format used
    viewspaceZ *= 0.99999;     // this is good for FP32 depth buffer

    const float3 pixCenterPos   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
    const float3 viewVec      = (float3)normalize(-pixCenterPos);
    
    // prevents normals that are facing away from the view vector - xeGTAO struggles with extreme cases, but in Vanilla it seems rare so it's disabled by default
    // viewspaceNormal = normalize( viewspaceNormal + max( 0, -dot( viewspaceNormal, viewVec ) ) * viewVec );

//#ifdef XE_GTAO_SHOW_NORMALS
//    g_outputDbgImage[pixCoord] = float4( DisplayNormalSRGB( viewspaceNormal.xyz ), 1 );
//#endif
//
//#ifdef XE_GTAO_SHOW_EDGES
//    g_outputDbgImage[pixCoord] = 1.0 - float4( edgesLRTB.x, edgesLRTB.y * 0.5 + edgesLRTB.w * 0.5, edgesLRTB.z, 1.0 );
//#endif

//#if XE_GTAO_USE_DEFAULT_CONSTANTS != 0
//    const float effectRadius              = (float)consts.EffectRadius * (float)XE_GTAO_DEFAULT_RADIUS_MULTIPLIER;
//    const float sampleDistributionPower   = (float)XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER;
//    const float thinOccluderCompensation  = (float)XE_GTAO_DEFAULT_THIN_OCCLUDER_COMPENSATION;
//    const float falloffRange              = (float)XE_GTAO_DEFAULT_FALLOFF_RANGE * effectRadius;
//#else
    const float effectRadius              = (float)constEffectRadius * (float)constRadiusMultiplier;
    const float sampleDistributionPower   = (float)constSampleDistributionPower;
    const float thinOccluderCompensation  = (float)constThinOccluderCompensation;
    const float falloffRange              = (float)constEffectFalloffRange * effectRadius;
//#endif

    const float falloffFrom       = effectRadius * ((float)1-(float)constEffectFalloffRange);

    // fadeout precompute optimisation
    const float falloffMul        = (float)-1.0 / ( falloffRange );
    const float falloffAdd        = falloffFrom / ( falloffRange ) + (float)1.0;

    float visibility = 0;
    float3 bentNormal = viewspaceNormal;

    // see "Algorithm 1" in https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
    {
        const float noiseSlice  = (float)localNoise.x;
        const float noiseSample = (float)localNoise.y;

        // quality settings / tweaks / hacks
        const float pixelTooCloseThreshold  = 1.3;      // if the offset is under approx pixel size (pixelTooCloseThreshold), push it out to the minimum distance

        // approx viewspace pixel size at pixCoord; approximation of NDCToViewspace( normalizedScreenPos.xy + ReShade::PixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;
        const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * float2(FFXIV::matProjInv[0][0], FFXIV::matProjInv[1][1]) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE; //viewspaceZ.xx * consts.NDCToViewMul_x_PixelSize;

        float screenspaceRadius   = effectRadius / (float)pixelDirRBViewspaceSizeAtCenterZ.x;

        // fade out for small screen radii 
        visibility += saturate((10 - screenspaceRadius)/100)*0.5;

#if 0   // sensible early-out for even more performance; disabled because not yet tested
        [branch]
        if( screenspaceRadius < pixelTooCloseThreshold )
        {
            XeGTAO_OutputWorkingTerm( pixCoord, 1, viewspaceNormal );
            return;
        }
#endif

        // this is the min distance to start sampling from to avoid sampling from the center pixel (no useful data obtained from sampling center pixel)
        const float minS = (float)pixelTooCloseThreshold / screenspaceRadius;
		
        //[unroll]
        for( float slice = 0; slice < sliceCount; slice++ )
        {
            float sliceK = (slice+noiseSlice) / sliceCount;
            // lines 5, 6 from the paper
            float phi = sliceK * XE_GTAO_PI;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            float2 omega = float2(cosPhi, -sinPhi);       //float2 on omega causes issues with big radii

            // convert to screen units (pixels) for later use
            omega *= screenspaceRadius;

            // line 8 from the paper
            const float3 directionVec = float3(cosPhi, sinPhi, 0);

            // line 9 from the paper
            const float3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);

            // line 10 from the paper
            //axisVec is orthogonal to directionVec and viewVec, used to define projectedNormal
            const float3 axisVec = normalize( cross(orthoDirectionVec, viewVec) );

            // alternative line 9 from the paper
            // float3 orthoDirectionVec = cross( viewVec, axisVec );

            // line 11 from the paper
            float3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

            // line 13 from the paper
            float signNorm = (float)sign( dot( orthoDirectionVec, projectedNormalVec ) );

            // line 14 from the paper
            float projectedNormalVecLength = length(projectedNormalVec);
            float cosNorm = (float)saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);

            // line 15 from the paper
            float n = signNorm * XeGTAO_FastACos(cosNorm);

            // this is a lower weight target; not using -1 as in the original paper because it is under horizon, so a 'weight' has different meaning based on the normal
            const float lowHorizonCos0  = cos(n+XE_GTAO_PI_HALF);
            const float lowHorizonCos1  = cos(n-XE_GTAO_PI_HALF);

            // lines 17, 18 from the paper, manually unrolled the 'side' loop
            float horizonCos0           = lowHorizonCos0; //-1;
            float horizonCos1           = lowHorizonCos1; //-1;

            [unroll]
            for( float step = 0; step < stepsPerSlice; step++ )
            {
                // R1 sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
                const float stepBaseNoise = float(slice + step * stepsPerSlice) * 0.6180339887498948482; // <- this should unroll
                float stepNoise = frac(noiseSample + stepBaseNoise);

                // approx line 20 from the paper, with added noise
                float s = (step+stepNoise) / (stepsPerSlice); // + (float2)1e-6f);

                // additional distribution modifier
                s       = (float)pow( s, (float)sampleDistributionPower );

                // avoid sampling center pixel
                s       += minS;

                // approx lines 21-22 from the paper, unrolled
                float2 sampleOffset = s * omega;

                float sampleOffsetLength = length( sampleOffset );

                // note: when sampling, using point_point_point or point_point_linear sampler works, but linear_linear_linear will cause unwanted interpolation between neighbouring depth values on the same MIP level!
                const float mipLevel    = (float)clamp( log2( sampleOffsetLength ) - constDepthMIPSamplingOffset, 0, XE_GTAO_DEPTH_MIP_LEVELS );

                // Snap to pixel center (more correct direction math, avoids artifacts due to sampling pos not matching depth texel center - messes up slope - but adds other 
                // artifacts due to them being pushed off the slice). Also use full precision for high res cases.
                sampleOffset = round(sampleOffset) * (float2)XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;

                float2 sampleScreenPos0 = normalizedScreenPos + sampleOffset;
                //float  SZ0 = sourceViewspaceDepth.SampleLevel( depthSampler, sampleScreenPos0, mipLevel ).x;
				float  SZ0 = tex2Dlod( g_sSrcWorkingDepth, float4(sampleScreenPos0, 0, mipLevel) ).x;
                float3 samplePos0 = XeGTAO_ComputeViewspacePosition( sampleScreenPos0, SZ0 );

                float2 sampleScreenPos1 = normalizedScreenPos - sampleOffset;
                //float  SZ1 = sourceViewspaceDepth.SampleLevel( depthSampler, sampleScreenPos1, mipLevel ).x;
				float  SZ1 = tex2Dlod( g_sSrcWorkingDepth, float4(sampleScreenPos1, 0, mipLevel) ).x;
                float3 samplePos1 = XeGTAO_ComputeViewspacePosition( sampleScreenPos1, SZ1 );

                float3 sampleDelta0     = (samplePos0 - float3(pixCenterPos)); // using float for sampleDelta causes precision issues
                float3 sampleDelta1     = (samplePos1 - float3(pixCenterPos)); // using float for sampleDelta causes precision issues
                float sampleDist0     = (float)length( sampleDelta0 );
                float sampleDist1     = (float)length( sampleDelta1 );

                // approx lines 23, 24 from the paper, unrolled
                float3 sampleHorizonVec0 = (float3)(sampleDelta0 / sampleDist0);
                float3 sampleHorizonVec1 = (float3)(sampleDelta1 / sampleDist1);

                // any sample out of radius should be discarded - also use fallof range for smooth transitions; this is a modified idea from "4.3 Implementation details, Bounding the sampling area"
//#if XE_GTAO_USE_DEFAULT_CONSTANTS != 0 && XE_GTAO_DEFAULT_THIN_OBJECT_HEURISTIC == 0
//                float weight0         = saturate( sampleDist0 * falloffMul + falloffAdd );
//                float weight1         = saturate( sampleDist1 * falloffMul + falloffAdd );
//#else
                // this is our own thickness heuristic that relies on sooner discarding samples behind the center
                float falloffBase0    = length( float3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1+thinOccluderCompensation) ) );
                float falloffBase1    = length( float3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1+thinOccluderCompensation) ) );
                float weight0         = saturate( falloffBase0 * falloffMul + falloffAdd );
                float weight1         = saturate( falloffBase1 * falloffMul + falloffAdd );
//#endif

                // sample horizon cos
                float shc0 = (float)dot(sampleHorizonVec0, viewVec);
                float shc1 = (float)dot(sampleHorizonVec1, viewVec);

                // discard unwanted samples
                shc0 = lerp( lowHorizonCos0, shc0, weight0 ); // this would be more correct but too expensive: cos(lerp( acos(lowHorizonCos0), acos(shc0), weight0 ));
                shc1 = lerp( lowHorizonCos1, shc1, weight1 ); // this would be more correct but too expensive: cos(lerp( acos(lowHorizonCos1), acos(shc1), weight1 ));

                // thickness heuristic - see "4.3 Implementation details, Height-field assumption considerations"
                horizonCos0 = max( horizonCos0, shc0 );
                horizonCos1 = max( horizonCos1, shc1 );
            }

#if 1       // I can't figure out the slight overdarkening on high slopes, so I'm adding this fudge - in the training set, 0.05 is close (PSNR 21.34) to disabled (PSNR 21.45)
            projectedNormalVecLength = lerp( projectedNormalVecLength, 1, 0.05 );
#endif

            // line ~27, unrolled
            float h0 = -XeGTAO_FastACos((float)horizonCos1);
            float h1 = XeGTAO_FastACos((float)horizonCos0);

            float iarc0 = ((float)cosNorm + (float)2 * (float)h0 * (float)sin(n)-(float)cos((float)2 * (float)h0-n))/(float)4;
            float iarc1 = ((float)cosNorm + (float)2 * (float)h1 * (float)sin(n)-(float)cos((float)2 * (float)h1-n))/(float)4;
            float localVisibility = (float)projectedNormalVecLength * (float)(iarc0+iarc1);
            visibility += localVisibility;

        }
		
        visibility /= (float)sliceCount;
        visibility = pow( visibility, (float)constFinalValuePower );
        visibility = max( (float)0.03, visibility ); // disallow total occlusion (which wouldn't make any sense anyhow since pixel is visible but also helps with packing bent normals)
    }

    XeGTAO_OutputWorkingTerm( pixCoord, visibility, bentNormal );
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

void XeGTAO_AddSample( AOTermType ssaoValue, float edgeValue, inout AOTermType sum, inout float sumWeight )
{
    float weight = edgeValue;    

    sum += (weight * ssaoValue);
    sumWeight += weight;
}

void XeGTAO_OutputPong( uint2 pixCoord, AOTermType outputValue, const bool finalApply )
{
    outputValue *=  (finalApply)?((float)XE_GTAO_OCCLUSION_TERM_SCALE):(1);

	tex2Dstore(g_outWorkingAOTerm, pixCoord, uint(outputValue * 255.0 + 0.5) );
}

float3 LoadNormal(const uint2 id)
{
	float2 texcoord = (float2(id) + 0.5) * XE_GTAO_SCALED_BUFFER_PIXEL_SIZE;
	float3 normal = tex2Dlod(g_sSrcCurNomals, float4(texcoord, 0, 0));//FFXIV::get_normal(texcoord);
	//normal.b = 1.0 - normal.b;
	return normalize(normal - 0.5);
}

float3 LoadNormalUV(const float2 texcoord)
{
	float3 normal = tex2Dlod(g_sSrcCurNomals, float4(texcoord, 0, 0));//FFXIV::get_normal(texcoord);
	//normal.b = 1.0 - normal.b;
	//return normal;
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
	
	float2 uv = viewcoord; //+ ReShade::PixelSize / 2;
    
	//float depth = tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).r;
	float cval = tex2Dlod(visibilitySampler, float4(uv, 0, 0)).r;
	float3 nval = LoadNormalUV(uv);
    //float3 pval = normalize(FFXIV::get_world_position_from_uv(uv, depth));
    
    for(int i=0; i<9; i++)
    {
        float2 uvOffset = uv + offset[i] * stepwidth * ReShade::PixelSize;
		
		//float depthtmp = tex2Dlod(ReShade::DepthBuffer, float4(uvOffset, 0, 0)).r;
        
        float ctmp = tex2Dlod(visibilitySampler, float4(uvOffset, 0, 0)).r;
        float t = cval - ctmp;
        float dist2 = t*t;//dot(t,t);
        float c_w = min(exp(-(dist2)/c_phi), 1.0);
        
        float3 ntmp = LoadNormalUV(uvOffset);
        float3 tt = nval - ntmp;
        dist2 = max(dot(tt,tt) / (stepwidth * stepwidth), 0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);
        
        //float3 ptmp = normalize(FFXIV::get_world_position_from_uv(uvOffset, depthtmp));
        //tt = pval - ptmp;
        //dist2 = dot(tt,tt);
        //float p_w = min(exp(-(dist2)/p_phi), 1.0);
        
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

void CSGTAO(uint3 tid : SV_DispatchThreadID)
{
#if XE_GTAO_QUALITY_LEVEL <= 0
	XeGTAO_MainPass( tid.xy, 1, 2, SpatioTemporalNoise(tid.xy), LoadNormal(tid.xy) );
#elif XE_GTAO_QUALITY_LEVEL == 1
	XeGTAO_MainPass( tid.xy, 2, 2, SpatioTemporalNoise(tid.xy), LoadNormal(tid.xy) );
#elif XE_GTAO_QUALITY_LEVEL == 2
	XeGTAO_MainPass( tid.xy, 3, 3, SpatioTemporalNoise(tid.xy), LoadNormal(tid.xy) );
#else
	XeGTAO_MainPass( tid.xy, 9, 3, SpatioTemporalNoise(tid.xy), LoadNormal(tid.xy) );
#endif
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
	normal.b = 1.0 - normal.b;
	normals = float4(normal, 1.0);
}

void PS_ApplyAO(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, in float4 orgColor, in bool avoidLights, out float4 output : SV_Target0)
{
	#if XE_GTAO_IL == 0
	float aoTerm = saturate(tex2D(g_sSrcFilteredOutput0, texcoord).r * XE_GTAO_OCCLUSION_TERM_SCALE * (1.0 + orgColor.a * avoidLights));
	
	orgColor.rgb *= aoTerm;
	
	output = Debug ? float4((aoTerm).rrr, orgColor.a) : float4(orgColor);
	#else
	float aoTerm = tex2D(g_sSrcFilteredOutput0, texcoord).r * XE_GTAO_OCCLUSION_TERM_SCALE * (1.0 + orgColor.a * avoidLights);
	
	float3 albedo = float4(orgColor.rgb * (1.0 + orgColor.a * 0/*FFXIV_LIGHT_SOURCE_INTENSITY_MULTIPLIER*/), 1.0);
	albedo.rgb = reinhard(albedo.rgb) * FFXIV_IL_INTENSITY;
	
	float3 coloredAO = GTAOMultiBounce(aoTerm, albedo);
	
	orgColor.rgb *= coloredAO;
	
	output = Debug ? float4((coloredAO).rgb, orgColor.a) : float4(orgColor);
	#endif
}

void PS_Out(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	float decal = tex2D(FFXIV::DecalNormalMap, texcoord).a;
	float4 orgColor = tex2D(ReShade::BackBuffer, texcoord);

	if(decal > FFXIV::DECAL_BACKGROUND && decal != FFXIV::MAGIC_WATER_ALPHA && decal != FFXIV::MAGIC_WATER_ALPHA2)
	{
		output = orgColor;
		return;
	}
	
	PS_ApplyAO(position, texcoord, orgColor, LightAvoidance, output);
}

void PS_Out_Decals(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	float decal = tex2D(FFXIV::DecalNormalMap, texcoord).a;
	float4 orgColor = tex2D(ReShade::BackBuffer, texcoord);

	if(decal <= FFXIV::DECAL_BACKGROUND || decal == FFXIV::MAGIC_WATER_ALPHA || decal == FFXIV::MAGIC_WATER_ALPHA2)
	{
		output = orgColor;
		return;
	}
	
	PS_ApplyAO(position, texcoord, orgColor, false, output);
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