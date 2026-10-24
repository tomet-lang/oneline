{ tomet, ... }:
{
  projectRootFile = "flake.nix";
  programs = {
    #= Nix
    nixfmt.enable = true;
    statix.enable = true;
    deadnix.enable = true;

    #= Shell
    shfmt.enable = true;
    shellcheck.enable = true;

    #= Main
    rustfmt.enable = true;
    taplo.enable = true;
  };

  settings = {
    global.excludes = [
      "*.lock"
    ];

    formatter = {
      tomet = {
        command = "${tomet}/bin/tomet";
        options = [
          "format"
          "-i"
        ];
        includes = [ "*.tmt" ];
      };
    };

    shfmt = {
      includes = [ "*.sh" ];
    };

    rustfmt = {
      includes = [ "*.rs" ];
    };
  };
}
