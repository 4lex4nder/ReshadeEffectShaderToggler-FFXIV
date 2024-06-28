#pragma once

namespace FFXIV {
    texture texNormals : NORMALS;
    sampler2D<float4> sNormals {
        Texture = texNormals; 
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };
    
    texture texDecalNormals : DECALS;
    sampler2D<float4> sDecalNormals {
        Texture = texDecalNormals; 
        MagFilter = POINT;
        MinFilter = POINT;
        MipFilter = POINT;
    };
    
    texture texNativeMotionVectors : MOTIONVECTORS;
    sampler sNativeMotionVectorTex { Texture = texNativeMotionVectors; };
    
    //texture texAlbedo : ALBEDO;
    //sampler sAlbedo { Texture = texAlbedo; };
    //
    texture texMaterial : MATERIAL;
    sampler sMaterial { Texture = texMaterial; };
    
    uniform float4x4 matProj < source = "mat_Proj"; >;
    uniform float4x4 matProjInv < source = "mat_ProjInv"; >;
    uniform float4x4 matViewProj < source = "mat_ViewProj"; >;
    uniform float4x4 matViewProjInv < source = "mat_ViewProjInv"; >;
    uniform float4x4 matView < source = "mat_View"; >;
    uniform float4x4 matViewInv < source = "mat_ViewInv"; >;
   
    // https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
    float2 _octWrap( float2 v )
    {
        //return ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );
        return float2((1.0 - abs( v.y ) ) * ( v.x >= 0.0 ? 1.0 : -1.0),
            (1.0 - abs( v.x ) ) * ( v.y >= 0.0 ? 1.0 : -1.0));
    }
    
    float2 _encode( float3 n )
    {
        n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
        n.xy = n.z >= 0.0 ? n.xy : _octWrap( n.xy );
        n.xy = n.xy * 0.5 + 0.5;
        return n.xy;
    }
    
    float3 _decode( float2 f )
    {
        f = f * 2.0 - 1.0;
        float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
        float t = saturate( -n.z );
        n.xy += n.xy >= 0.0 ? -t : t;
        return normalize( n );
    }
    
    float3 get_normal(float2 texcoord)
    {
        float4 tnormal = tex2Dlod(sNormals, float4(texcoord, 0, 0));
        float4 tdecal_normal = tex2Dlod(sDecalNormals, float4(texcoord, 0, 0));
        
        // Blend with decal normals
        float4 normal = float4(lerp(tnormal.rgb, tdecal_normal.rgb, tdecal_normal.a), 1);
        
        // Convert from world space to screen space
        float3x3 matV = float3x3(FFXIV::matView[0].xyz, FFXIV::matView[1].xyz, FFXIV::matView[2].xyz);
        normal = float4(mul(matV, normal.xyz - 0.5), 1);
        
        return normal.rgb + 0.5;
    }
    
    float2 get_motion(float2 texcoord)
    {
        return tex2Dlod(sNativeMotionVectorTex, float4(texcoord, 0, 0)).rg;
    }
    
    //float3 get_albedo(float2 texcoord)
    //{
    //    return tex2Dlod(sAlbedo, float4(texcoord, 0, 0)).rgb;
    //}
    //
    float get_roughness(float2 texcoord)
    {
        return tex2Dlod(sMaterial, float4(texcoord, 0, 0)).g;
    }
}