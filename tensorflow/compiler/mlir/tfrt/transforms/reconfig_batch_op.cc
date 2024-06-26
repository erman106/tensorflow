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

#include <algorithm>
#include <cstdint>
#include <memory>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

class ReconfigBatchOpPass
    : public mlir::PassWrapper<ReconfigBatchOpPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit ReconfigBatchOpPass(ReconfigBatchOpPassOptions options)
      : mlir::PassWrapper<ReconfigBatchOpPass,
                          mlir::OperationPass<mlir::ModuleOp>>() {
    min_num_batch_threads_ = options.min_num_batch_threads;
  }
  ReconfigBatchOpPass()
      : mlir::PassWrapper<ReconfigBatchOpPass,
                          mlir::OperationPass<mlir::ModuleOp>>() {}
  ReconfigBatchOpPass(const ReconfigBatchOpPass& other)
      : mlir::PassWrapper<ReconfigBatchOpPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other) {}

  ReconfigBatchOpPass& operator=(const ReconfigBatchOpPass& other) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReconfigBatchOpPass)

 private:
  llvm::StringRef getArgument() const final { return "tfrt-reconfig-batch-op"; }

  llvm::StringRef getDescription() const final {
    return "Reconfig batch op such as num_batch_threads.";
  }

  void runOnOperation() override {
    if (min_num_batch_threads_ > 0) {
      mlir::ModuleOp module = getOperation();
      module.walk([&](mlir::TF::BatchFunctionOp batch_op) {
        int64_t num_batch_threads = batch_op.getNumBatchThreads();
        num_batch_threads =
            std::max(num_batch_threads, min_num_batch_threads_.getValue());
        batch_op.setNumBatchThreads(num_batch_threads);
      });
    }
  }

 protected:
  mlir::Pass::Option<int64_t> min_num_batch_threads_{
      *this, "tfrt-min-num-batch-threads", llvm::cl::init(1),
      llvm::cl::desc("Minimum number of batch threads")};
  ;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateReconfigBatchOpPass(
    ReconfigBatchOpPassOptions options) {
  return std::make_unique<ReconfigBatchOpPass>(options);
}

static mlir::PassRegistration<ReconfigBatchOpPass> register_pass;

}  // namespace tfrt_compiler
}  // namespace tensorflow
