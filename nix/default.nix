{
  flake-parts,
  ...
}@inputs:
flake-parts.lib.mkFlake { inherit inputs; } {
  systems = [
    "x86_64-linux"
    "aarch64-linux"
    "aarch64-darwin"
  ];
  imports = [
    inputs.treefmt-nix.flakeModule
  ];

  perSystem =
    { pkgs, ... }:{
      devShells.default = pkgs.callPackage ./dev.nix {
        tomet = inputs.tomet.packages.${pkgs.system}.tomet;
        tomet-shelf = inputs.tomet-shelf.packages.${pkgs.system}.tomet-shelf;
      };

      treefmt = import ./formatter.nix { inherit tomet; };
    };
}
