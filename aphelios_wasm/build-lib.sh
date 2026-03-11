#这是官方示例打包指令, 但是有问题
# cargo build --target wasm32-unknown-unknown --release
# wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web

wasm-pack build ./ --target web --out-dir ./build --features wgpu


#cargo tree -p aphelios_wasm -no-default-features --features "wgpu" > aaa.txt

#cargo tree -p aphelios_wasm --edges features --target wasm32-unknown-unknown

