# flake.nix (Final Simplified Version)
{
  description = "A development environment using uv for Python dependencies.";

  inputs = {
    # We only need one modern nixpkgs input now.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # The outputs are simple again, no legacy input.
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        iqbart-cpp = pkgs.stdenv.mkDerivation {
          pname = "iqbart-cpp";
          version = "0.1.0";
          src = ./iqbart;
          nativeBuildInputs = [ pkgs.cmake pkgs.python311Packages.pybind11];
          buildInputs = [pkgs.python311 ];
          installPhase = ''
            mkdir -p $out/${pkgs.python311.sitePackages}
            cp *.so $out/${pkgs.python311.sitePackages}/
          '';
        };
        iqbartPythonPath = "${iqbart-cpp}/${pkgs.python311.sitePackages}";
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python311
            pkgs.uv
            pkgs.R
            pkgs.gcc
            pkgs.git
            # We're back to using the modern LLVM.
            pkgs.llvm
          ];

          shellHook = ''
            echo "Nix environment activated. Setting up Python virtual environment with uv..."
            test -d .venv || uv venv
            source .venv/bin/activate
            uv pip install -e . --quiet
            echo "Done. Python environment is ready."
            export PYTHONPATH="${iqbartPythonPath}:$PYTHONPATH"
            echo "Custom C++ module 'iqbart_cpp' is now available to Python."
          '';
        };
      }
    );
}
