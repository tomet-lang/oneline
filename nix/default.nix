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
    { pkgs, ... }:
    let
      inherit (inputs.tomet.packages.${pkgs.system}) tomet;
      inherit (inputs.tomet-shelf.packages.${pkgs.system}) tomet-shelf;
    in
    {
      devShells.default = pkgs.callPackage ./dev.nix { inherit tomet tomet-shelf; };

      treefmt = import ./formatter.nix { inherit tomet; };
    };
}
