/* Copyright 2021 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/activity_watcher/activity.h"

#include <memory>

namespace tensorflow {
namespace activity_watcher {
void MaybeEnableMultiWorkersWatching(CoordinationServiceAgent* agent) {}

namespace internal {

std::atomic<int> g_watcher_level(kWatcherDisabled);
ActivityId RecordActivityStart(std::unique_ptr<Activity>) {
  return kActivityNotRecorded;
}
void RecordActivityEnd(ActivityId id) {}

}  // namespace internal

}  // namespace activity_watcher
}  // namespace tensorflow
