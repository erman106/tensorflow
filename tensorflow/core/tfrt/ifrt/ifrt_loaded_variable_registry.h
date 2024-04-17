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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

// This class is thread safe.
class IfrtLoadedVariableRegistry {
 public:
  struct Key {
    absl::flat_hash_set<int> device_ids;
    std::string input_name;
    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      h = H::combine(std::move(h), key.input_name, key.device_ids);
      return h;
    }

    friend bool operator==(const Key& x, const Key& y) {
      return x.input_name == y.input_name && x.device_ids == y.device_ids;
    }
  };

  struct LoadedVariable {
    xla::ifrt::Future<absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>> array;
  };
  using LoadedVariableConstructor =
      absl::AnyInvocable<absl::StatusOr<LoadedVariable>() const>;

  // Tries to register a loaded variable with the given name.
  // Returns an error if the named array does not already exists and
  // loaded_variable_constructor failed to create an array. Note that it returns
  // OK if the named array already exists.
  // loaded_variable_constructor is invoked in the caller thread.
  absl::Status TryRegisterLoadedVariable(
      Key key, LoadedVariableConstructor&& loaded_variable_constructor)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<LoadedVariable> GetLoadedVariable(Key key) const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, LoadedVariable> loaded_variable_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
