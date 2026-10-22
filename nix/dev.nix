{
  pkgs,
  mkShell,

  tomet,
  tomet-shelf,
  ...
}:
mkShell {
  buildInputs = with pkgs; [
    tomet
    tomet-shelf
  ];

  shellHook = ''
    echo "📖 Tomet"
  '';
}
