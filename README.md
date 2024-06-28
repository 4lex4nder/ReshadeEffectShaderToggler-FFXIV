# ReshadeEffectShaderToggler-FFXIV
FFXIV specific ReshadeEffectShaderToggler configuration. 

# Requirements
* [REST v1.3.17](https://github.com/4lex4nder/ReshadeEffectShaderToggler/releases/tag/v.1.3.17)
* [ReShade 6.1+ with Addon support](https://reshade.me/)

# Instructions
1. Quit the game
2. Click on the green "Code" button in the top right of this page then on "Download ZIP"
3. From the downloaded archive, put ReshadeEffectShaderToggler.ini into C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. (Optionally) Put Shader\ to where all your other ReShade shaders are
5. Copy ReshadeEffectShaderToggler.addon64 from the REST release next to the ReshadeEffectShaderToggler.ini at, e.g., C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. Start the game

# Notes
* Disable dynamic resolution
* Motion vectors through ffxiv_crashpad only available with DLSS and TSCMAA AA modes (make sure Frame Rate Threshold is set to "Always enabled" for DLSS)
* Set gamma to exactly 50
* If you have "FFKeepUI" and/or "FFRestoreUI" effects enabled, disable them
* Try to avoid having multiples of the same effects hidden somewhere in nested folders in ReShade's Shader directory
* Set game to DirectX 11 and Settings to Max, except for Anti-Aliasing and Ambient Occlusion, which can be whatever. You can further adjust after making sure everything works to verify nothing breaks on changing the settings

# Shaders (optional)
TODO: write more things