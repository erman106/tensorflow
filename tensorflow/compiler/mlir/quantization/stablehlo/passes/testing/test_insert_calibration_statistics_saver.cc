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

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

// NOLINTNEXTLINE - Automatically generated.
#define GEN_PASS_DEF_TESTINSERTCALIBRATIONSTATISTICSSAVERWITHSKIPPINGPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

namespace {

class TestInsertCalibrationStatisticsSaverWithSkippingPass
    : public impl::TestInsertCalibrationStatisticsSaverWithSkippingPassBase<
          TestInsertCalibrationStatisticsSaverWithSkippingPass> {
 public:
  using impl::TestInsertCalibrationStatisticsSaverWithSkippingPassBase<
      TestInsertCalibrationStatisticsSaverWithSkippingPass>::
      TestInsertCalibrationStatisticsSaverWithSkippingPassBase;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestInsertCalibrationStatisticsSaverWithSkippingPass)

 private:
  void runOnOperation() override;
};

void TestInsertCalibrationStatisticsSaverWithSkippingPass::runOnOperation() {
  PassManager pass_manager{&getContext()};

  pass_manager.addPass(stablehlo::CreateInsertCalibrationStatisticsSaverPass(
      /*calibration_data_dir=*/"",
      /*skipping_aggregator_ops=*/{"skipping_id"}));

  if (failed(pass_manager.run(getOperation()))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo::testing
