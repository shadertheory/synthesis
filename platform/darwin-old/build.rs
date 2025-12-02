use std::io::{self, Read};
use std::path::*;
use std::process::Command;
use std::{env, fs};

fn visit(path: &Path, cb: &mut dyn FnMut(PathBuf)) -> io::Result<()> {
    for e in fs::read_dir(path)? {
        let e = e?;
        let path = e.path();
        if path.is_dir() {
            visit(&path, cb)?;
        } else if path.is_file() {
            cb(path);
        }
    }
    Ok(())
}

fn main() {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Must be built with cargo");
    let env = env::var("CARGO_CFG_TARGET_ENV").expect("Must be built with cargo");
    let out = Path::new(&*env::var("OUT_DIR").expect("Must be run from cargo")).to_path_buf();
    let manifest =
        std::path::Path::new(&*env::var("CARGO_MANIFEST_DIR").expect("Must be built with cargo"))
            .to_path_buf();
    let src = manifest.join("src");
    let target = if env.is_empty() {
        os.to_string()
    } else {
        env.to_string()
    };

    let sdk = match (os.as_str(), env.as_str()) {
        ("macos", _) => "macosx",
        ("ios", "sim") => "iphonesimulator",
        ("ios", _) => "iphoneos",
        _ => panic!("Unsupported target OS for Metal compilation: {target}"),
    };

    println!("cargo::warning=Using Metal SDK: '{sdk}' for target: '{target}'");

    let mut files = vec![];
    visit(&src, &mut |path: PathBuf| {
        let Some(ext) = path.extension() else {
            return;
        };

        println!("cargo::warning={ext:?}");
        if ext != "metal" {
            return;
        }

        let Some(name) = path.file_name() else {
            return;
        };

        // Read the file content to check for shader entry points
        let mut content = String::new();
        let Ok(mut file) = std::fs::File::open(&path) else {
            return;
        };
        if file.read_to_string(&mut content).is_err() {
            return;
        }
        let is_source_file = ["kernel", "vertex", "fragment"]
            .iter()
            .map(|x| format!("{x} "))
            .any(|x| content.contains(&x));
        if !is_source_file {
            println!("cargo::warning=Skipping header: {:?}", name);
            return;
        }
        files.push(path);
    })
    .expect("Failed to traverse");
    if files.len() == 0 {
        return;
    }

    let mut ir = vec![];
    for file in files {
        let name = file.file_name().expect("failed to get name");

        println!("cargo::warning=Compiling shader '{name:?}'");

        let mut out = out.clone();
        out.push(name);
        let out = out.with_extension("ir");

        let args = [
            "-sdk",
            sdk,
            "metal",
            "-c",
            "-I",
            src.as_os_str().to_str().expect("failed to get src"),
            file.as_os_str().to_str().expect("failed to get file"),
            "-o",
            out.as_os_str().to_str().expect("failed to get out"),
        ];
        let compiler = Command::new("xcrun")
            .args(args)
            .spawn()
            .expect("Failed to execute 'xcrun'");

        let output = compiler
            .wait_with_output()
            .expect("Failed to retrieve 'xcrun' output");

        let stderr = output.stderr;

        if !output.status.success() {
            let err = String::from_utf8(stderr).unwrap();
            panic!("Metal compilation failed for {name:?}: {err}");
        }

        ir.push(out);
    }

    let lib = out.join("shaders.metallib");

    let mut args = vec!["-sdk", sdk, "metallib"];

    // Add all the IR file paths to the arguments
    let ir_strings: Vec<String> = ir
        .iter()
        .map(|p| p.to_str().expect("Invalid path").to_string())
        .collect();

    for ir_path in &ir_strings {
        args.push(ir_path.as_str());
    }

    args.push("-o");
    args.push(lib.to_str().unwrap());

    let output = Command::new("xcrun")
        .args(args)
        .output()
        .expect("Failed to execute 'xcrun metallib'");

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        panic!("Metal linking failed:\n{err}");
    }

    let count = ir.len();
    println!("cargo::warning=Completed Metal library with {count} shaders.");

    let mut file = std::fs::File::open(&lib).expect("Failed to open metallib");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("Failed to read metallib");

    let shader_bin_path = out.join("shaders.bin");
    std::fs::write(&shader_bin_path, bytes).expect("Failed to write shader binary");

    println!(
        "cargo::rustc-env=METAL_SHADER_LIB_PATH={}",
        shader_bin_path.display()
    );
}
