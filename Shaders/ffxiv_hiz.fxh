#pragma once

#include "Reshade.fxh"

#undef NUM_MIPMAPS
#define NUM_MIPMAPS 1

//#undef NPOT_WIDTH
//#define NPOT_WIDTH false
//
//#undef NPOT_WIDTH_LAST_MIP
//#define NPOT_WIDTH_LAST_MIP 0
//
//#undef NPOT_HEIGHT_LAST_MIP
//#define NPOT_HEIGHT_LAST_MIP 0
//
//#undef NPOT_HEIGHT
//#define NPOT_HEIGHT false

#if ((BUFFER_WIDTH >> 2) >= 8 || (BUFFER_HEIGHT >> 2) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 2
#endif
#if ((BUFFER_WIDTH >> 3) >= 8 || (BUFFER_HEIGHT >> 3) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 3
#endif
#if ((BUFFER_WIDTH >> 4) >= 8 || (BUFFER_HEIGHT >> 4) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 4
#endif
#if ((BUFFER_WIDTH >> 5) >= 8 || (BUFFER_HEIGHT >> 5) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 5
#endif
#if ((BUFFER_WIDTH >> 6) >= 8 || (BUFFER_HEIGHT >> 6) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 6
#endif
#if ((BUFFER_WIDTH >> 7) >= 8 || (BUFFER_HEIGHT >> 7) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 7
#endif
#if ((BUFFER_WIDTH >> 8) >= 8 || (BUFFER_HEIGHT >> 8) >= 8)
	#undef NUM_MIPMAPS
	#define NUM_MIPMAPS 8
#endif

//#undef NUM_MIPMAPS
//	#define NUM_MIPMAPS 8
//
//#if (BUFFER_WIDTH % (1 << NUM_MIPMAPS)) != 0
//	#undef NPOT_WIDTH_LAST_MIP
//	#define NPOT_WIDTH_LAST_MIP 1
//#endif
//
//#if (BUFFER_HEIGHT % (1 << NUM_MIPMAPS)) != 0
//	#undef NPOT_HEIGHT_LAST_MIP
//	#define NPOT_HEIGHT_LAST_MIP 1
//#endif

texture2D FFXIV_Hi_Z
{
	Width = BUFFER_WIDTH >> 1;
	Height = BUFFER_HEIGHT >> 1;
	MipLevels = NUM_MIPMAPS;
	Format = RG32F;
};

sampler sFFXIV_Hi_Z { Texture = FFXIV_Hi_Z; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };