#pragma once

namespace FFXIV {
	static const float MAGIC_WATER_ALPHA = 0.050980390;
	static const float MAGIC_WATER_ALPHA2 = MAGIC_WATER_ALPHA + 0.2;
	//static const float MAGIC_ALPHA = 0.0039215684790;
	static const float MAGIC_ALPHA = MAGIC_WATER_ALPHA - 0.000000002;
	static const float Z_NEAR = 0.1;
	
	// Needs sane initialization, otherwise RTGI with IBL enabled will break on init and never recover (?)
	uniform float4x4 matProj < source = "mat_Proj"; > = float4x4(1.520478, 0,        0, 		0,
																 0,        2.432765, 0, 		0,
																 0,        0,        -1.000050, -0.100005,
																 0,        0,        -1,        0);
	uniform float4x4 matProjInv < source = "mat_InvProj"; > = float4x4(0.657688, 0,        0, 		  0,
																	   0,        0.411055, 0, 		  0,
																	   0,        0,        0, 		  -1,
																	   0,		 0, 	   -9.999500, 10);
	uniform float4x4 matViewProj < source = "mat_ViewProj"; >;
	uniform float4x4 matViewProjInv < source = "mat_InvViewProj"; >;
	uniform float3 camPos < source = "vec_CameraWorldPos"; >;
	uniform float3 camDir < source = "vec_CameraViewDir"; >;

	texture NormalMapTex : NORMALMAP;
	sampler NormalMap { Texture = NormalMapTex; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT; };

	texture DecalNormalMapTex : HAIR;
	sampler DecalNormalMap { Texture = DecalNormalMapTex; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT; };

	float3 get_normal(float2 texcoord)
	{
		float4 normal = tex2Dlod(NormalMap, float4(texcoord, 0, 0));
		float4 decal = tex2Dlod(DecalNormalMap, float4(texcoord, 0, 0));
		normal.xyz = lerp(normal.xyz, float3(0.5, 0.5, 1), normal.w == 1);
		normal.xyz = lerp(normal.xyz, decal.xyz, decal.a > MAGIC_ALPHA);
		return normal.xyz;
	}
	
	// Annoying, but didn't see correct z_far in the constant buffers (and it differs between over and inner world)
	float z_far()
	{
		return (Z_NEAR * (-matProj[2][2])) * rcp((-matProj[2][2]) - 1);
	}
	
	float linearize_depth(float depth)
	{
		return rcp((depth * matProjInv[3][2] + matProjInv[3][3]) * z_far());
	}

	float4 linearize_depths(float4 depths)
	{
		return rcp((depths * matProjInv[3][2] + matProjInv[3][3]) * z_far());
	}
	
	// Used with separate linearization component (linearize_depth) for compatibility with various shaders
	float3 get_position_from_uv_depth(float2 uv, float depth)
	{
		float4 pos = (float4(uv.x, uv.y, depth, 1) * float4(2, 2, 1, 1)) - float4(1, 1, 0, 0);
		float4 res = pos * float4(matProjInv[0][0] * pos.z, matProjInv[1][1] * pos.z, 1, 1);
		
		return res.xyz;
	}
	
	// Used with raw depth buffer values
	float3 get_position_from_uv_raw_depth(float2 uv, float depth)
	{
		float4 pos = (float4(uv.x, uv.y, depth, 1) * float4(2, 2, 1, 1)) - float4(1, 1, 0, 0);
		float4 res = mul(matProjInv, pos);
		res /= res.w;
		
		return res.xyz * float3(1, 1, -1); // ffxiv uses negative z -> negate
	}
	
	float2 get_uv_from_position(float3 pos)
	{
		return (pos.xy * float2(matProj[0][0], matProj[1][1]) * rcp(pos.z) + float2(1, 1)) * float2(0.5, 0.5);
	}
	
	float depth_to_view_z(in float depth)
	{
		return depth * z_far();
	}
	
	float view_z_to_depth(in float z)
	{
		return z * rcp(z_far());
	}
}