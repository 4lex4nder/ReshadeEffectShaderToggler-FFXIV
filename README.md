# ReshadeEffectShaderToggler-FFXIV
FFXIV specific ReshadeEffectShaderToggler configuration. 

# Requirements
* [REST v1.3.2](https://github.com/4lex4nder/ReshadeEffectShaderToggler/releases/tag/v1.3.2)
* [ReShade 5.9+ with Addon support](https://reshade.me/)

# Instructions
1. Quit the game
2. Click on the green "Code" button in the top right of this page then on "Download ZIP"
3. From the downloaded archive, put ReshadeEffectShaderToggler.ini into C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. (Optionally) Put Shader\ to where all your other ReShade shaders are
5. Copy ReshadeEffectShaderToggler.addon64 from the REST release next to the ReshadeEffectShaderToggler.ini at, e.g., C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game
4. Start the game

# Notes
* Disable dynamic resolution
* Set gamma to exactly 50
* If you have "FFKeepUI" and/or "FFRestoreUI" effects enabled, disable them
* Try to avoid having multiples of the same effects hidden somewhere in nested folders in ReShade's Shader directory
* Set game to DirectX 11 and Settings to Max, except for Anti-Aliasing and Ambient Occlusion, which can be whatever. You can further adjust after making sure everything works to verify nothing breaks on changing the settings

# Troubleshooting
## Everything is completely messed up, e.g., trees stretch across the horizon
Open the ReShade menu and navigate to the `Add-ons` tab. Expand the `Reshade Effect Shader Toggler` addon by click on the arrow on the left. Expand `Options` and set `Constant Buffer copy method` to:
* `gpu_readback` if you use the optional shaders provided here
* `none` otherwise

Click `Save All Toggle Groups` on the bottom and restart the game

# Shaders (optional)
TODO: write more things