{
  pkgs,
  lib,
  mkShell,

  tomet,
  tomet-shelf,
  ...
}:
mkShell {
  buildInputs = with pkgs; [
    tomet
    tomet-shelf
    #= GPU
    vulkan-tools
  ];

  LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.vulkan-loader
  ];

  shellHook = ''
    echo "📖 Tomet"
  '';
}
