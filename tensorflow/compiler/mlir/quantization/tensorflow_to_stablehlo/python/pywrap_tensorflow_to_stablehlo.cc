/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11   // IWYU pragma: keep
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil    // IWYU pragma: keep
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow_to_stablehlo/tf_to_stablehlo.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"

namespace py = pybind11;

namespace mlir::pywrap {

absl::StatusOr<std::string> ModuleToBytecode(ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

py::bytes PywrapSavedModelToStablehlo(
    absl::string_view input_path,
    const std::vector<std::string>& exported_model_signatures =
        {"serving_default"},
    const std::vector<std::string>& tag_names = {"serve"},
    absl::string_view input_arg_shapes_str = "") {
  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      TfToStablehlo(input_path, &context, exported_model_signatures, tag_names,
                    input_arg_shapes_str, /*is_input_mlir_module=*/false);

  if (!module.ok()) {
    PyErr_SetString(PyExc_ValueError,
                    "failed to converted TensorFlow to StableHLO");
    return {};
  }

  auto bytecode = ModuleToBytecode(module.value().get());
  if (!bytecode.ok()) {
    PyErr_SetString(PyExc_ValueError, "failed to write module to bytecode");
    return {};
  }

  return bytecode.value();
}

py::bytes PywrapTfMlirToStablehlo(absl::string_view input_path,
                                  absl::string_view input_arg_shapes_str = "") {
  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = TfToStablehlo(
      input_path, &context, /*exported_model_signatures=*/{}, /*tag_names=*/{},
      input_arg_shapes_str, /*is_input_mlir_module=*/true);

  if (!module.ok()) {
    PyErr_SetString(PyExc_ValueError,
                    "failed to converted TensorFlow to StableHLO");
    return {};
  }

  auto bytecode = ModuleToBytecode(module.value().get());
  if (!bytecode.ok()) {
    PyErr_SetString(PyExc_ValueError, "failed to write module to bytecode");
    return {};
  }

  return bytecode.value();
}

}  // namespace mlir::pywrap

PYBIND11_MODULE(pywrap_tensorflow_to_stablehlo, m) {
  m.doc() = "TenforFlow to StableHLO APIs.";

  // LINT.IfChange(savedmodel_to_stablehlo)
  m.def("savedmodel_to_stablehlo", &mlir::pywrap::PywrapSavedModelToStablehlo,
        R"pbdoc(
        This tool converts TensorFlow SavedModel to StableHLO.

        * input-path: The path to the input TensorFlow SavedModel.
        * exported-model-signatures: Comma-separated list of exported model
          signatures to convert. Ignored for MLIR input.
        * tag_names: Comma-separated list of tags for loading SavedModel. Ignored for MLIR
          input.
        * input-arg-shapes: A string representation of input argument shapes for
          'main' entry-point, separating tensors with ':', dimension with ',', and
          using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
          expresses argument shapes [1,2], [] and [1,?].
        )pbdoc",
        py::arg("input_path"),
        py::arg("exported_model_signatures") =
            std::vector<std::string>{"serving_default"},
        py::arg("tag_names") = std::vector<std::string>{"serve"},
        py::arg("input_arg_shapes_str") = "");
  // LINT.ThenChange(pywrap_tensorflow_to_stablehlo.pyi:tensorflow_to_stablehlo)
  //
  // LINT.IfChange(tensorflow_mlir_to_stablehlo)
  m.def("tensorflow_mlir_to_stablehlo", &mlir::pywrap::PywrapTfMlirToStablehlo,
        R"pbdoc(
        This tool converts TensorFlow MLIR file to StableHLO.

        * input-path: The path to the input TensorFlow MLIR module with .mlir
          extension.
        * input-arg-shapes: A string representation of input argument shapes for
          'main' entry-point, separating tensors with ':', dimension with ',', and
          using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
          expresses argument shapes [1,2], [] and [1,?].
        )pbdoc",
        py::arg("input_path"), py::arg("input_arg_shapes_str") = "");
  // LINT.ThenChange(pywrap_tensorflow_to_stablehlo.pyi:tensorflow_mlir_to_stablehlo)
}
