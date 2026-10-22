{
  description = "Tomet documentation site (Astro, rendered by the tomet CLI)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";

    #[ Tool ]
    tomet.url = "github:tomet-lang/tomet";
    tomet-shelf.url = "github:tomet-lang/tomet-shelf";
  };

  outputs = inputs: import ./nix inputs;
}
