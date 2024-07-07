# ReshadeEffectShaderToggler-FFXIV
FFXIV specific ReshadeEffectShaderToggler configuration. 

# Requirements
* [REST v1.3.20](https://github.com/4lex4nder/ReshadeEffectShaderToggler/releases/tag/v1.3.20), note that the ReShade installer may provide a different version. The configuration is only tested against this version
* [ReShade 6.1+ with Addon support](https://reshade.me/)

# Instructions
1. Quit the game
2. Click on the green "Code" button in the top right of this page then on "Download ZIP"
3. From the downloaded archive, put ReshadeEffectShaderToggler.ini into C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. (Optionally) Put Shader\ to where all your other ReShade shaders are
5. Copy ReshadeEffectShaderToggler.addon64 from the REST release next to the ReshadeEffectShaderToggler.ini at, e.g., C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. Start the game

# Notes
* Disable dynamic resolution or anything that entails running an internal buffer that doesn't match your output resolution. This means running FSR1 at 100% scale and, in case of DLSS, use a mod to turn it into DLAA. Otherwise you're on your own
* If you have "FFKeepUI" and/or "FFRestoreUI" effects enabled, disable them
* Try to avoid having multiples of the same effects hidden somewhere in nested folders in ReShade's Shader directory
* Set game to DirectX 11 and Settings to Max, except for Anti-Aliasing and Ambient Occlusion, which can be whatever. You can further adjust after making sure everything works to verify nothing breaks on changing the settings

# Shaders (optional)
* FFXIV_Crashpad: Drop-in replacement for iMMERSE Launchpad, providing game engine normals and motion vectors instead of approximations
* FFXIV_XeGTAO: Port of original XeGTAO (modified with visibility bitmasks). Consists of two effects `FFXIV_XeGTAO` and `FFXIV_XeGTAO_Decals`. The former is the main effect, the latter is for decals, Both are required for the effect to work properly
* ReBlur: Motion blur making use of the game's motion vectors