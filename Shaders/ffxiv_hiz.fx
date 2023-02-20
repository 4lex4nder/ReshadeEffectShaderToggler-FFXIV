/*
	Original article: GPU Pro 5, Hi-Z Screen-Space Cone-Traced Reflections by Yasin Uludag
	Construction using CS scratchpad taken from XeGTAO: https://github.com/GameTechDev/XeGTAO, https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Shaders/XeGTAO.hlsli
*/

#include "Reshade.fxh"
#include "ffxiv_hiz.fxh"

#if NUM_MIPMAPS > 4
	#define FIRST_PASS_MIPMAPS 4
#else
	#define FIRST_PASS_MIPMAPS NUM_MIPMAPS
#endif

#define SECOND_PASS_MIPMAPS (NUM_MIPMAPS - 4)

#if FIRST_PASS_MIPMAPS < 2
	#define FIRST_PASS_NUM_THREADS_HEIGHT 1
	#define FIRST_PASS_NUM_THREADS_WIDTH 1
#else
	#define FIRST_PASS_NUM_THREADS_HEIGHT (1 << (FIRST_PASS_MIPMAPS - 1))
	#define FIRST_PASS_NUM_THREADS_WIDTH (1 << (FIRST_PASS_MIPMAPS - 1))
#endif

#if SECOND_PASS_MIPMAPS < 2
	#define SECOND_PASS_NUM_THREADS_HEIGHT 1
	#define SECOND_PASS_NUM_THREADS_WIDTH 1
#else
	#define SECOND_PASS_NUM_THREADS_HEIGHT (1 << (SECOND_PASS_MIPMAPS - 1))
	#define SECOND_PASS_NUM_THREADS_WIDTH (1 << (SECOND_PASS_MIPMAPS - 1))
#endif

//#undef SECOND_PASS_NUM_THREADS
//#define SECOND_PASS_NUM_THREADS 2

//----------------

uniform bool Debug <
	ui_type = "radio";
	ui_label = "Debug";
	ui_tooltip = "Draw HiZ mip layers";
> = 0;

uniform float Mip_Layer <
	ui_type = "drag";
	ui_label = "Debug Mip Layer";
	ui_min = 0; ui_max = NUM_MIPMAPS;
	ui_step = 1;
> = 0;

//----------------

texture FFXIV_SSR_Depth : DEPTH;
sampler sFFXIV_SSR_Depth_Point { Texture = FFXIV_SSR_Depth; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

storage2D stFFXIV_Hi_Z_0 { Texture = FFXIV_Hi_Z; MipLevel = 0; };

#if NUM_MIPMAPS >= 2
	storage2D stFFXIV_Hi_Z_1 { Texture = FFXIV_Hi_Z; MipLevel = 1; };
#endif
#if NUM_MIPMAPS >= 3
	storage2D stFFXIV_Hi_Z_2 { Texture = FFXIV_Hi_Z; MipLevel = 2; };
#endif
#if NUM_MIPMAPS >= 4
	storage2D stFFXIV_Hi_Z_3 { Texture = FFXIV_Hi_Z; MipLevel = 3; };
#endif
#if NUM_MIPMAPS >= 5
	storage2D stFFXIV_Hi_Z_4 { Texture = FFXIV_Hi_Z; MipLevel = 4; };
#endif
#if NUM_MIPMAPS >= 6
	storage2D stFFXIV_Hi_Z_5 { Texture = FFXIV_Hi_Z; MipLevel = 5; };
#endif
#if NUM_MIPMAPS >= 7
	storage2D stFFXIV_Hi_Z_6 { Texture = FFXIV_Hi_Z; MipLevel = 6; };
#endif
#if NUM_MIPMAPS >= 8
	storage2D stFFXIV_Hi_Z_7 { Texture = FFXIV_Hi_Z; MipLevel = 7; };
#endif

//----------------

static const uint2 g_scratchSize = uint2(FIRST_PASS_NUM_THREADS_WIDTH, FIRST_PASS_NUM_THREADS_HEIGHT);
groupshared float2 g_scratchDepths[g_scratchSize.x * g_scratchSize.y];

float2 GetMinMax(float2 depth0, float2 depth1, float2 depth2, float2 depth3)
{
	float min_depth = min(depth0.r, min(depth1.r, min(depth2.r, depth3.r)));
	float max_depth = max(depth0.g, max(depth1.g, max(depth2.g, depth3.g)));
	
	return float2(min_depth, max_depth);
}

void SetMinMaxMapScratch(uint iteration, uint2 tid, uint2 baseCoord, storage2D store, out float2 minMax)
{
	barrier();
	
	uint soffset = (1 << iteration);
	
	[branch]
	if(all((tid.xy % soffset.xx) == 0))
	{
		uint offset = (1 << (iteration - 1));
		float2 inTL = g_scratchDepths[tid.x * g_scratchSize.y + tid.y];
		float2 inTR = g_scratchDepths[(tid.x + offset) * g_scratchSize.y + tid.y];
		float2 inBL = g_scratchDepths[tid.x * g_scratchSize.y + (tid.y + offset)];
		float2 inBR = g_scratchDepths[(tid.x + offset) * g_scratchSize.y + (tid.y + offset)];
	
		minMax = GetMinMax(inTL, inTR, inBL, inBR);
		
		tex2Dstore(store, baseCoord / soffset, minMax.rgrg);
	}
}

//----------------

void CS_CreateHiZ_0_3(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
	const uint2 baseCoord = id.xy;
	const uint2 pixCoord = baseCoord * 2;
	uint soffset = tid.x * g_scratchSize.y + tid.y;
	
	float depth0 = tex2Dfetch(sFFXIV_SSR_Depth_Point, pixCoord + uint2(0, 0)).r;
	float depth1 = tex2Dfetch(sFFXIV_SSR_Depth_Point, pixCoord + uint2(1, 0)).r;
	float depth2 = tex2Dfetch(sFFXIV_SSR_Depth_Point, pixCoord + uint2(0, 1)).r;
	float depth3 = tex2Dfetch(sFFXIV_SSR_Depth_Point, pixCoord + uint2(1, 1)).r;
	
	float2 minMax = GetMinMax(depth0.rr, depth1.rr, depth2.rr, depth3.rr);

	tex2Dstore(stFFXIV_Hi_Z_0, baseCoord, minMax.rgrg);
	
	#if NUM_MIPMAPS > 1
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(1, tid.xy, baseCoord, stFFXIV_Hi_Z_1, minMax);
	#endif
	
	#if NUM_MIPMAPS > 2
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(2, tid.xy, baseCoord, stFFXIV_Hi_Z_2, minMax);
	#endif
	
	#if NUM_MIPMAPS > 3
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(3, tid.xy, baseCoord, stFFXIV_Hi_Z_3, minMax);
	#endif
}

#if NUM_MIPMAPS > 4
void CS_CreateHiZ_4_7(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
	const uint2 baseCoord = id.xy;
	const uint2 pixCoord = baseCoord * 2;
	uint soffset = tid.x * g_scratchSize.y + tid.y;
	
	float2 depth0 = tex2Dfetch(stFFXIV_Hi_Z_3, pixCoord + uint2(0, 0)).rg;
	float2 depth1 = tex2Dfetch(stFFXIV_Hi_Z_3, pixCoord + uint2(1, 0)).rg;
	float2 depth2 = tex2Dfetch(stFFXIV_Hi_Z_3, pixCoord + uint2(0, 1)).rg;
	float2 depth3 = tex2Dfetch(stFFXIV_Hi_Z_3, pixCoord + uint2(1, 1)).rg;

	float2 minMax = GetMinMax(depth0, depth1, depth2, depth3);
	
	tex2Dstore(stFFXIV_Hi_Z_4, baseCoord, minMax.rgrg);
	
	#if NUM_MIPMAPS > 5
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(1, tid.xy, baseCoord, stFFXIV_Hi_Z_5, minMax);
	#endif
	
	#if NUM_MIPMAPS > 6
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(2, tid.xy, baseCoord, stFFXIV_Hi_Z_6, minMax);
	#endif
	
	#if NUM_MIPMAPS > 7
	g_scratchDepths[soffset] = minMax;
	SetMinMaxMapScratch(3, tid.xy, baseCoord, stFFXIV_Hi_Z_7, minMax);
	#endif
}
#endif

void PS_Out(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	output = 0;
	if(Debug)
	{
		if(Mip_Layer > 0)
		{
			output = float4(tex2Dlod(sFFXIV_Hi_Z, float4(texcoord.xy, 0, (Mip_Layer - 1))).rg, 1, 1);
			return;
		}
		else
		{
			output = float4(tex2Dlod(sFFXIV_SSR_Depth_Point, float4(texcoord.xy, 0, 0)).rr, 1, 1);
			return;
		}
	}
	discard;
}

//----------------

technique FFXIV_HIZ
{
	pass
	{
		ComputeShader = CS_CreateHiZ_0_3<FIRST_PASS_NUM_THREADS_WIDTH, FIRST_PASS_NUM_THREADS_HEIGHT>;
		DispatchSizeX = ((BUFFER_WIDTH << 1) + ((FIRST_PASS_NUM_THREADS_WIDTH << 1) - 1)) / (FIRST_PASS_NUM_THREADS_WIDTH << 1);
		DispatchSizeY = ((BUFFER_HEIGHT << 1) + ((FIRST_PASS_NUM_THREADS_HEIGHT << 1) - 1)) / (FIRST_PASS_NUM_THREADS_HEIGHT << 1);
		GenerateMipMaps = false;
	}
	
	#if NUM_MIPMAPS > 4
	pass
	{
		ComputeShader = CS_CreateHiZ_4_7<SECOND_PASS_NUM_THREADS_WIDTH, SECOND_PASS_NUM_THREADS_HEIGHT>;
		DispatchSizeX = ((BUFFER_WIDTH >> 4) + ((SECOND_PASS_NUM_THREADS_WIDTH << 1) - 1)) / (SECOND_PASS_NUM_THREADS_WIDTH << 1);
		DispatchSizeY = ((BUFFER_HEIGHT >> 4) + ((SECOND_PASS_NUM_THREADS_HEIGHT << 1) - 1)) / (SECOND_PASS_NUM_THREADS_HEIGHT << 1);
		GenerateMipMaps = false;
	}
	#endif
	
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_Out;
	}
}