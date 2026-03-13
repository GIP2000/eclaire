{
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
        rust-overlay.url = "github:oxalica/rust-overlay";
    };
    outputs = { self, nixpkgs, rust-overlay } : 
    let 
        system = "x86_64-linux";
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {inherit system overlays;};
    in {
        devShells.${system}.default = pkgs.mkShell {
            packages = with pkgs; [
                (rust-bin.nightly.latest.default.override {
                    extensions = ["rust-analyzer" "rust-src"];
                })
                zlib
                libxml2
                llvmPackages_21.llvm
                llvmPackages_21.clang
            ];

        };

    };
}
