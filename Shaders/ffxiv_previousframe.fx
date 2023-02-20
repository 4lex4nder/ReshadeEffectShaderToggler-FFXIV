#include "Reshade.fxh"
#include "ffxiv_common.fxh"

texture2D FFXIV_PrevFrame { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture2D FFXIV_PrevNormals { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture2D FFXIV_PrevFrameDepth { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R32F; };

void PS_WritePrevFrame(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 color : SV_Target0, out float depth: SV_Target1, out float4 normal: SV_Target2)
{
	color = float4(tex2D(ReShade::BackBuffer, texcoord).rgb, 1);
	normal = FFXIV::get_normal(texcoord);
	depth = tex2D(ReShade::DepthBuffer, texcoord).r;
}


void PS_Out(in float4 position : SV_Position, in float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	output = 0;
	discard;
}

technique FFXIV_PREVIOUS_FRAME
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_WritePrevFrame;
		RenderTarget0 = FFXIV_PrevFrame;
		RenderTarget1 = FFXIV_PrevFrameDepth;
		RenderTarget2 = FFXIV_PrevNormals;
	}
	
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader  = PS_Out;
	}
}