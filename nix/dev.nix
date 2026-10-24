{
  pkgs,
  lib,
  mkShell,

  tools,
  ...
}:
mkShell {
  buildInputs = with pkgs; [
    tools
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
