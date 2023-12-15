import turnkeyml.build.sequences as sequences
from .runtime import OnnxRTOpenVino

implements = {
    "runtimes": {
        "ort_openvino": {
            "build_required": True,
            "RuntimeClass": OnnxRTOpenVino,
            "supported_devices": {"x86"},
            "default_sequence": sequences.optimize_fp32,
        }
    }
}
