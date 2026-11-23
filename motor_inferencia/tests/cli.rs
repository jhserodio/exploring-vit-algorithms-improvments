use std::process::{Command, Output};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_motor_inferencia"))
        .args(args)
        .output()
        .expect("can run the research CLI")
}

#[test]
fn help_explains_device_and_reference_export() {
    let result = invoke(&["--help"]);
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("--cpu | --cuda"));
    assert!(stdout.contains("--export-prefix"));
}

#[test]
fn missing_required_args_returns_nonzero() {
    let result = invoke(&["--cpu"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("Usage:"));
}

#[test]
fn contradictory_device_options_are_rejected() {
    let result = invoke(&["--cpu", "--cuda"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("choose exactly one device"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_without_feature_does_not_silently_fallback_to_cpu() {
    let result = invoke(&[
        "--image",
        "missing.png",
        "--weights",
        "missing.safetensors",
        "--cuda",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("CUDA is disabled"));
}
