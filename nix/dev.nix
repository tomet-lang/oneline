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
    #= Benchmarks (simdjson comparison)
    cmake # simdjson-sys builds the bundled simdjson with CMake
    simdjson # reference C++ library for the standalone driver
  ];

  LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.vulkan-loader
  ];

  shellHook = ''
    echo "📖 Tomet"
  '';
}
