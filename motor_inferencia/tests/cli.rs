use std::process::{Command, Output};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_motor_inferencia"))
        .args(args)
        .output()
        .expect("can execute research binary")
}

#[test]
fn help_describes_spatial_cpu_and_cuda_gate() {
    let r = invoke(&["--help"]);
    assert!(r.status.success());
    let help = String::from_utf8_lossy(&r.stdout);
    assert!(help.contains("--cpu | --cuda"));
    assert!(help.contains("--export-prefix"));
    assert!(help.contains("spatial"));
}

#[test]
fn no_required_arguments_fails() {
    let r = invoke(&["--cpu"]);
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("Usage:"));
}

#[test]
fn contradictory_device_flags_are_rejected() {
    let r = invoke(&["--cpu", "--cuda"]);
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("choose exactly one device"));
}

#[test]
fn cuda_full_forward_is_explicitly_unavailable_not_silent_cpu() {
    let r = invoke(&["--image", "x.png", "--weights", "weights.safetensors", "--cuda"]);
    assert!(!r.status.success());
    let msg = String::from_utf8_lossy(&r.stderr);
    assert!(msg.contains("native CUDA Rust full-model inference is not implemented yet"));
    assert!(msg.contains("No silent CPU fallback"));
}
