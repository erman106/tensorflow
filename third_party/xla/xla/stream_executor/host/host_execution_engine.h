/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTION_ENGINE_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTION_ENGINE_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"

namespace stream_executor::host {

class RuntimeExecutionEngine {
 public:
  // Pointer to a function exported to the Xla executable.
  //
  // Xla exported function expects all arguments to be passed as an array of
  // opaque pointers to the actual values. In C++ it would look like this:
  //
  //   void compute(int32_t arg0, float arg1, ...);
  //
  //   void __xla_compute(void** args) {
  //      int32_t arg0 = *reinterpret_cast<int32_t*>(args[0]);
  //      float arg1 = *reinterpret_cast<float*>(args[1]);
  //      ...
  //      compute(arg0, arg1, ...);
  //   }
  //
  // This is required to avoid dealing with ABI of the compiled function. See
  // `SetUpExportedFunction` for implementation details.
  using ExportedFunctionPtr = void (*)(void **);

  // Callback to register symbols with the execution engine (e.g. to register
  // custom runtime intrinsics for Gpu integration).
  using SymbolsBinding =
      std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>;

  // Callback to run optimization passes on the compiled LLVM module.
  using OptimizingTransformer = std::function<llvm::Error(llvm::Module *)>;

  // Callback to construct an optimizing transformer for the given options.
  using MakeOptimizingTransformer =
      std::function<OptimizingTransformer(llvm::TargetMachine *targetMachine)>;

  // Compose multiple symbol bindings into a single symbol binding function.
  static SymbolsBinding BindAll(std::vector<SymbolsBinding> bindings);

  //------------------------------------------------------------------------- //
  // Options for creating execution engine from an LLVM module.
  //------------------------------------------------------------------------- //

  struct JitOptions {
    // User-provided codegen optimization level.
    llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::Default;

    // User-provided target machine specification.
    std::shared_ptr<llvm::TargetMachine> target_machine = nullptr;

    // User-provided builder for the optimizing transformer.
    MakeOptimizingTransformer make_optimizing_transformer;

    // User-provided memory mapper for allocating memory for executables.
    llvm::SectionMemoryManager::MemoryMapper *section_memory_mapper = nullptr;

    // User-provided bindings for symbols.
    SymbolsBinding symbols_binding = nullptr;

    // Notify the llvm's global GDB notifications listener.
    bool enable_gdb_listener = false;

    // Notify the llvm's global Perf notifications listener.
    bool enable_perf_listener = false;

    // Save compiled object file.
    bool save_compiled_obj_file = true;
  };

  // Creates a new execution engine by compiling the provided LLVM module to
  // a native executable using LLVM ORC stack.
  static absl::StatusOr<std::unique_ptr<RuntimeExecutionEngine>>
  CreateFromModule(std::unique_ptr<llvm::LLVMContext> ctx,
                   std::unique_ptr<llvm::Module> module, JitOptions options,
                   absl::Span<const std::string_view> exported);

  //------------------------------------------------------------------------- //
  // Options for creating execution engine from an AOT compiled object file.
  //------------------------------------------------------------------------- //

  struct AotOptions {
    // User-provided memory mapper for allocating memory for executables.
    llvm::SectionMemoryManager::MemoryMapper *section_memory_mapper = nullptr;

    // User-provided bindings for symbols.
    SymbolsBinding symbols_binding = nullptr;

    // Notify the llvm's global GDB notifications listener.
    bool enable_gdb_listener = true;

    // Notify the llvm's global Perf notifications listener.
    bool enable_perf_listener = true;
  };

  // Creates a new execution engine by loading AOT compiled XLA executable
  // object file.
  // static absl::StatusOr<std::unique_ptr<RuntimeExecutionEngine>>
  // CreateFromObjFile(
  //     std::unique_ptr<llvm::MemoryBuffer>, AotOptions options,
  //     absl::Span<const std::string_view> exported);

  //------------------------------------------------------------------------- //

  // Returns a pointer to the exported function.
  absl::Span<const ExportedFunctionPtr> exported() const { return exported_; }

  ExportedFunctionPtr exported(unsigned ordinal) const {
    return exported_[ordinal];
  }

  // Return a memory buffer with a object file behind this execution engine. Can
  // be null if execution engine didn't save the compiled object file.
  std::unique_ptr<llvm::MemoryBuffer> obj_file() const;

 private:
  RuntimeExecutionEngine(bool enable_gdb_listener, bool enable_perf_listener);

  // We build execution engine on top of the ORC LLJIT API, which owns all
  // compiled/loaded object files and does the linking at run time.
  //
  // TODO(ezhulenev): Instead of keeping LLJIT alive we should be able to keep
  // only llvm::orc::JITDylibSP owning main dylib and the object layer owning
  // memory-mapped regions holding object files. Once we are done with
  // executable compilation this jit is defunct because it holds an expired
  // weak_ptr to an llvm::orc::TargetMachine instance.
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  // Pointers to resolved exported functions. Indexed by function ordinal.
  std::vector<ExportedFunctionPtr> exported_;

  // Object file behind the compiled executable. Can be null.
  std::unique_ptr<llvm::MemoryBuffer> obj_file_;

  llvm::JITEventListener *gdb_listener_ = nullptr;
  llvm::JITEventListener *perf_listener_ = nullptr;
};

// Emits an interface function ('exported_name') that wraps all arguments
// of a function ('original_name') into a single pointer to a ptr**,
// thereby exposing a trivial ABI. The original function is also inlined,
// if possible.
absl::Status ExportWithXlaRuntimeAbi(llvm::Module &module,
                                     std::string_view original_name,
                                     std::string_view exported_name);

// Virtual base class that owns jit-compiled function.
class HostExecutionEngine {
 public:
  virtual ~HostExecutionEngine() {};
  virtual SE_HOST_Kernel *kernel() const = 0;
};

class LlvmExecutionEngine : public HostExecutionEngine {
 public:
  SE_HOST_Kernel *kernel() const override { return kernel_; }
  ~LlvmExecutionEngine() override {
    // TODO(tsilytskyi): clean up kernel_
  }

  static absl::StatusOr<std::unique_ptr<LlvmExecutionEngine>> CreateFromLlvmIr(
      absl::string_view name, absl::string_view entry, absl::string_view ir,
      absl::Span<const std::string> options);

 private:
  explicit LlvmExecutionEngine(
      std::unique_ptr<RuntimeExecutionEngine> exec_engine)
      : engine_(std::move(exec_engine)) {
    kernel_ = reinterpret_cast<SE_HOST_Kernel *>(engine_->exported(0));
  };
  std::unique_ptr<RuntimeExecutionEngine> engine_;
  SE_HOST_Kernel *kernel_;
};

class CppExecutionEngine : public HostExecutionEngine {
 public:
  ~CppExecutionEngine() override = default;
  SE_HOST_Kernel *kernel() const override { return kernel_; }

 private:
  CppExecutionEngine() = default;
  SE_HOST_Kernel *kernel_ = nullptr;
};

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_EXECUTION_ENGINE_H_
