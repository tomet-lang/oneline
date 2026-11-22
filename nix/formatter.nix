{ tomet, ... }:
{
  projectRootFile = "flake.nix";
  programs = {
    #--- Nix ---
    nixfmt.enable = true;
    statix.enable = true;
    deadnix.enable = true;
    #--- Shell ---
    shfmt.enable = true;
    shellcheck.enable = true;
    #--- Rust ---
    rustfmt.enable = true;
    taplo = {
      enable = true;
      settings = {
        formatting = {
          align_entries = true;
          align_comments = true;
          reorder_keys = false;
        };
        rule = [
          {
            keys = [
              "dependencies"
              "dev-dependencies"
              "build-dependencies"
              "workspace.dependencies"
            ];
            formatting = {
              reorder_keys = true;
            };
          }
        ];
      };
    };

    #--- Biome ---
    biome = {
      enable = true;
      formatCommand = "format";
      settings = {
        formatter = {
          enabled = true;
          useEditorconfig = true;
        };
        linter.enabled = false;
        assist.enabled = false;
      };
      includes = [
        "*.json"
        "*.jsonc"
        #--- Web ---
        "*.js"
        "*.ts"
        "*.mjs"
        "*.mts"
        "*.cjs"
        "*.cts"
        "*.jsx"
        "*.tsx"
      ];
    };
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
