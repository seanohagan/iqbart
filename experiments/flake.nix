{
  description = "A development environment using uv for Python dependencies.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        iqbart-cpp = pkgs.stdenv.mkDerivation {
          pname = "iqbart-cpp";
          version = "0.1.0";
          src = pkgs.lib.cleanSource ../iqbart;
          unpackPhase = ''
            cp -r $src/. ./
          '';
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
        devShells.default =
          let
            drbart = pkgs.rPackages.buildRPackage {
              name="drbart";
              version="0.0.0.9";
              src = pkgs.fetchFromGitHub {
                  owner = "vittorioorlandi";
                  repo = "drbart";
                  rev = "main";

                  sha256 = "sha256-D/ft7CkUiiBx+218rWy354IFsfHdAx2rxRyR/ENvn8A=";
                };

              unpackPhase = ''
                cp -r $src/. ./
                chmod -R u+w .
              '';

              nativeBuildInputs = [ pkgs.gfortran pkgs.gfortran.cc ];

                propagatedBuildInputs = with pkgs.rPackages; [
                  Rcpp
                  RColorBrewer
                ];
            };
          in
            pkgs.mkShell {
              buildInputs = [
                pkgs.python311
                pkgs.uv
                pkgs.R
                pkgs.gcc
                pkgs.git
                pkgs.llvm
                pkgs.stdenv.cc.cc.lib
                pkgs.bashInteractive
                pkgs.rPackages.gamlss
                pkgs.gfortran
                pkgs.gfortran.cc

                drbart
              ];

              #export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

              shellHook = ''
                echo "Nix environment activated. Setting up Python virtual environment with uv..."
                test -d .venv || uv venv
                source .venv/bin/activate
                uv pip install -e . --quiet --no-binary=true
                echo "Done. Python environment is ready."
                export LD_LIBRARY_PATH="$(python -m rpy2.situation LD_LIBRARY_PATH)":$LD_LIBRARY_PATH
                export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                export PYTHONPATH="${iqbartPythonPath}:$PYTHONPATH"
                echo "Custom C++ module 'iqbart_cpp' is now available to Python."
              '';
            };
      }
    );
}
